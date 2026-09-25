"""Forward latency of the target, middle models and drafter at the shapes the pipeline uses.

Each forward processes q new tokens on top of a cached context of --ctx tokens, which is
what a verification pass does (q = 16 for a drafter block, q = P+1 for a target check).
This gives the cost ratio r = c_T / c that decides the optimal window P*.

Two modes:
  eager     plain HF forward (what the experiments currently run). At batch 1 this is
            dominated by Python / kernel-launch overhead, not by weight bandwidth.
  compiled  static KV cache + torch.compile(mode="reduce-overhead") (CUDA graphs), which is
            close to what a serving engine does. Quantization only pays off here.

  python scripts/bench_forward.py --mode eager compiled --mids ao4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import _make_cache  # noqa: E402
from transformers import StaticCache  # noqa: E402

from hspec.models import free, load_draft, load_mid, load_target  # noqa: E402
from hspec.pipeline import crop  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

WARMUP = 5


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


@torch.inference_mode()
def time_eager(model, ctx: int, q: int, reps: int, vocab: int) -> float:
    device = next(model.parameters()).device
    cache = _make_cache(model.config)
    prompt = torch.randint(0, vocab, (1, ctx), device=device)
    model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
    block = torch.randint(0, vocab, (1, q), device=device)
    pos = torch.arange(ctx, ctx + q, device=device).unsqueeze(0)
    times = []
    for i in range(reps + WARMUP):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(block, position_ids=pos, past_key_values=cache, use_cache=True,
              output_hidden_states=True)
        torch.cuda.synchronize()
        if i >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
        crop(cache, ctx)
    return _median(times)


@torch.inference_mode()
def time_compiled(model, ctx: int, qs: list[int], reps: int, vocab: int) -> dict[int, float]:
    device = next(model.parameters()).device
    cache = StaticCache(config=model.config, max_cache_len=ctx + max(qs) + 8)
    prompt = torch.randint(0, vocab, (1, ctx), device=device)
    model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1,
          cache_position=torch.arange(ctx, device=device))
    fwd = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False, dynamic=False)
    out = {}
    for q in qs:
        block = torch.randint(0, vocab, (1, q), device=device)
        cpos = torch.arange(ctx, ctx + q, device=device)
        times = []
        for i in range(reps + WARMUP):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # same positions every time: the static cache slots are simply overwritten
            fwd(block, position_ids=cpos.unsqueeze(0), cache_position=cpos,
                past_key_values=cache, use_cache=True, output_hidden_states=True)
            torch.cuda.synchronize()
            if i >= WARMUP:
                times.append((time.perf_counter() - t0) * 1000)
        out[q] = _median(times)
    return out


@torch.inference_mode()
def time_draft(draft, target, ctx: int, new_ctx: int, reps: int) -> float:
    """One drafter call: new_ctx fresh context features plus a 16-token masked block."""
    device = next(draft.parameters()).device
    bs = draft.block_size
    width = len(draft.target_layer_ids) * target.config.hidden_size
    cache = _make_cache(draft.config)
    pos = torch.arange(ctx + new_ctx + bs, device=device).unsqueeze(0)
    h0 = torch.randn(1, ctx, width, device=device, dtype=torch.bfloat16)
    noise = torch.randn(1, bs, target.config.hidden_size, device=device, dtype=torch.bfloat16)
    draft(target_hidden=h0, noise_embedding=noise, position_ids=pos[:, : ctx + bs],
          past_key_values=cache, use_cache=True)
    crop(cache, ctx)
    h = torch.randn(1, new_ctx, width, device=device, dtype=torch.bfloat16)
    head = target.lm_head
    times = []
    for i in range(reps + WARMUP):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hid = draft(target_hidden=h, noise_embedding=noise, position_ids=pos[:, ctx:],
                    past_key_values=cache, use_cache=True)
        draft.compute_logits(hid[:, 1:], head).argmax(-1)
        torch.cuda.synchronize()
        if i >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
        crop(cache, ctx)
    return _median(times)


def bench_model(name, model, args, vocab, rows):
    for mode in args.mode:
        row = {"model": name, "mode": mode}
        try:
            if mode == "eager":
                for q in args.qs:
                    row[f"q={q}"] = time_eager(model, args.ctx, q, args.reps, vocab)
            else:
                row.update({f"q={q}": t for q, t in
                            time_compiled(model, args.ctx, args.qs, args.reps, vocab).items()})
        except Exception as e:  # noqa: BLE001
            print(f"[{name} | {mode}] failed: {type(e).__name__}: {str(e)[:300]}")
            continue
        rows.append(row)
        print(row, flush=True)
        torch._dynamo.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mids", nargs="+", default=["bnb4:Qwen/Qwen3-8B", "ao4:Qwen/Qwen3-8B"])
    ap.add_argument("--mode", nargs="+", default=["eager", "compiled"], choices=["eager", "compiled"])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--qs", nargs="+", type=int, default=[1, 17, 33, 65, 129])
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    rows = []
    target = load_target(args.target)
    vocab = target.config.vocab_size
    bench_model(f"bf16:{args.target}", target, args, vocab, rows)

    draft = load_draft(args.draft)
    t_draft = time_draft(draft, target, args.ctx, 7, args.reps)
    print(f"drafter (eager): {t_draft:.2f} ms per block", flush=True)
    del draft, target
    free()

    for spec in args.mids:
        try:
            m = load_mid(spec)
        except Exception as e:  # noqa: BLE001
            print(f"could not load {spec}: {e}")
            continue
        bench_model(spec, m, args, vocab, rows)
        del m
        free()

    print_table(rows, ["model", "mode"] + [f"q={q}" for q in args.qs],
                f"median forward latency (ms), ctx={args.ctx}")
    print(f"drafter (eager): {t_draft:.2f} ms per block")
    save_json("bench_forward", {"args": vars(args), "rows": rows, "draft_ms": t_draft})


if __name__ == "__main__":
    main()
