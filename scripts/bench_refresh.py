"""Measured cost of the cheap KV refresh components (one A100, batch 1, CUDA graphs, as bench_overlap.py).

Times, after a ctx-token prefix in the KV cache:
  target verify       the full target over q = 16 / 32 / 48 / 64 tokens               (the reference)
  shallow bf16        the target's first --shallow layers over the 16-token block (exact layer-1 tap; no LM head)
  shallow 4-bit       the same layers of the 4-bit copy
  copy                the full 4-bit copy over the block                           (today's refresh)
  map                 the ridge map, 16 rows x (3H+1) -> 5H taps (fp32 and bf16), and -> one fused H vector (bf16)
  drafter             one drafter block, and a batched call with --restarts blocks (the restart call)
Each component is also given as a fraction of the target's 16-token verify in the same harness, so it can be
carried over to other cost tables (e.g. vLLM).

  python scripts/bench_refresh.py --ctx 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_overlap import KEEP, capture_draft, capture_model, timed  # noqa: E402

from hspec.models import free, load_draft, load_mid, load_target  # noqa: E402
from hspec.pipeline import crop  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


@torch.inference_mode()
def capture_shallow(model, n_layers, ctx, q):
    """CUDA graph of the base model truncated to its first n_layers, over q tokens after a ctx-token prefix."""
    base = model.model
    full_layers, full_n = base.layers, model.config.num_hidden_layers
    base.layers = torch.nn.ModuleList(list(full_layers)[:n_layers])
    model.config.num_hidden_layers = n_layers
    try:
        dev = next(model.parameters()).device
        vocab = model.config.vocab_size
        cache = DynamicCache()
        base(torch.randint(0, vocab, (1, ctx), device=dev), past_key_values=cache, use_cache=True)
        ids = torch.randint(0, vocab, (1, q), device=dev)
        pos = torch.arange(ctx, ctx + q, device=dev).unsqueeze(0)

        def step():
            base(ids, position_ids=pos, past_key_values=cache, use_cache=True)
            crop(cache, ctx)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        KEEP.append([(layer.keys, layer.values) for layer in cache.layers])
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            base(ids, position_ids=pos, past_key_values=cache, use_cache=True)
        crop(cache, ctx)
        KEEP.extend([ids, pos, cache])
        return g
    finally:
        base.layers = full_layers
        model.config.num_hidden_layers = full_n


@torch.inference_mode()
def capture_matmul(rows, din, dout, dtype, dev):
    x = torch.randn(rows, din, device=dev, dtype=dtype)
    w = torch.randn(din, dout, device=dev, dtype=dtype) / din ** 0.5
    out = torch.empty(rows, dout, device=dev, dtype=dtype)

    def step():
        torch.matmul(x, w, out=out)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    KEEP.extend([x, w, out])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--shallow", type=int, default=2, help="target layers for the exact layer-1 tap")
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--out", default="bench_refresh")
    args = ap.parse_args()
    dev = "cuda"

    target = load_target(args.target)
    draft = load_draft(args.draft)
    H = target.config.hidden_size
    bs = draft.block_size
    rows = []

    def add(name, ms, ref=None):
        rows.append({"component": name, "ms": ms, "frac_of_verify16": (ms / ref) if ref else float("nan")})
        print(rows[-1], flush=True)

    gv = {q: capture_model(target, args.ctx, q)[0] for q in (16, 32, 48, 64)}
    t_v16 = timed(gv[16].replay)
    for q, g in gv.items():
        add(f"target verify {q} tokens", t_v16 if q == 16 else timed(g.replay), t_v16)
    add(f"shallow bf16: target layers 0..{args.shallow - 1}, {bs} tokens",
        timed(capture_shallow(target, args.shallow, args.ctx, bs).replay), t_v16)
    gd = capture_draft(draft, target)
    add("drafter, one block", timed(gd.replay), t_v16)
    add(f"drafter, {args.restarts} blocks batched (restart call)",
        timed(capture_draft(draft, target, batch=args.restarts).replay), t_v16)
    for nm, din, dout, dt in (("map fp32 (3H+1 -> 5H)", 3 * H + 1, 5 * H, torch.float32),
                              ("map bf16 (3H+1 -> 5H)", 3 * H + 1, 5 * H, torch.bfloat16),
                              ("map bf16 (3H+1 -> fused H)", 3 * H + 1, H, torch.bfloat16)):
        add(nm + f", {bs} rows", timed(capture_matmul(bs, din, dout, dt, dev).replay), t_v16)
    try:
        copy = load_mid(args.copy)
        add(f"shallow 4-bit: copy layers 0..{args.shallow - 1}, {bs} tokens",
            timed(capture_shallow(copy, args.shallow, args.ctx, bs).replay), t_v16)
        add(f"copy, full 4-bit pass over {bs} tokens", timed(capture_model(copy, args.ctx, bs)[0].replay), t_v16)
        free(copy)
    except Exception as e:  # noqa: BLE001
        print(f"[copy] failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print_table(rows, list(rows[0]), f"measured component costs, one A100, batch 1, CUDA graphs, ctx {args.ctx}")
    save_json(args.out, {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
