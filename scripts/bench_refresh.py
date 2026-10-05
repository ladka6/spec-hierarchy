"""Stage latencies of feature-refreshed decoding under CUDA graphs (no torch.compile), batch 1.

Each stage is captured once with torch.cuda.CUDAGraph at the exact shape the loop uses and then
replayed, which removes HF's per-call Python / launch overhead (the dominant cost in eager mode):

  verify   bf16 target, q new tokens on a cached context, explicit 4-D attention mask (tree-shaped
           masks cost the same as any other mask of that shape), hidden states returned
  copy     the cheap copy (torchao int4 by default; bnb4 for comparison) over q tokens, hidden
           states + logits for the last 16 positions

StaticCache with fixed cache_position, so every replay overwrites the same slots. Eager timings of
the same calls are printed next to the graph timings.

  python scripts/bench_refresh.py --copies ao4:Qwen/Qwen3-8B bnb4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import StaticCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import free, load_mid, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def stage(model, ctx, q, reps, keep=None):
    """(eager ms, graph ms) for one forward of q tokens on a ctx-token static cache."""
    dev = next(model.parameters()).device
    V = model.config.vocab_size
    max_len = ctx + q + 8
    cache = StaticCache(config=model.config, max_cache_len=max_len)
    prompt = torch.randint(0, V - 10, (1, ctx), device=dev)
    with torch.inference_mode():
        model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1,
              cache_position=torch.arange(ctx, device=dev))
    ids = torch.randint(0, V - 10, (1, q), device=dev)
    cpos = torch.arange(ctx, ctx + q, device=dev)
    mask = torch.zeros(1, 1, q, max_len, dtype=torch.bool, device=dev)
    mask[..., :ctx] = True
    mask[0, 0, :, ctx : ctx + q] = torch.tril(torch.ones(q, q, dtype=torch.bool, device=dev))
    kw = dict(position_ids=cpos[None], cache_position=cpos, attention_mask=mask, past_key_values=cache,
              use_cache=True, output_hidden_states=True, logits_to_keep=keep or q)

    def run():
        # this transformers StaticCache writes at its own counter (cumulative_length), not at
        # cache_position: reset it so every call overwrites the same q slots after the context
        for layer in cache.layers:
            layer.cumulative_length.fill_(ctx)
        out = model(ids, **kw)
        return out.logits, out.hidden_states[-1]

    with torch.inference_mode():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            run()
        torch.cuda.synchronize()
        eager = (time.perf_counter() - t0) * 1000 / reps
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        graph = st.elapsed_time(en) / reps
    del g, cache
    return eager, graph


def draft_stage(draft, target, ctx, new_ctx, n_blocks, reps):
    """(eager ms, graph ms) for one drafter call: new_ctx fresh context features plus n_blocks masked
    blocks of 16 (n_blocks > 1 = batched restarts with per-block masks) on a ctx-position cache."""
    from dflash.model import _make_cache

    from hspec.pipeline import crop

    dev = next(draft.parameters()).device
    bs = draft.block_size
    W = len(draft.target_layer_ids) * target.config.hidden_size
    H = target.config.hidden_size
    cache = _make_cache(draft.config)
    q = n_blocks * bs
    pos = torch.arange(ctx + new_ctx + q, device=dev)[None]
    with torch.inference_mode():
        draft(target_hidden=torch.randn(1, ctx, W, device=dev, dtype=torch.bfloat16),
              noise_embedding=torch.randn(1, bs, H, device=dev, dtype=torch.bfloat16),
              position_ids=pos[:, : ctx + bs], past_key_values=cache, use_cache=True)
        crop(cache, ctx)
    h = torch.randn(1, new_ctx, W, device=dev, dtype=torch.bfloat16)
    noise = torch.randn(1, q, H, device=dev, dtype=torch.bfloat16)
    mask = None
    if n_blocks > 1:
        m = torch.zeros(q, ctx + new_ctx + q, dtype=torch.bool, device=dev)
        m[:, :ctx] = True
        for r in range(n_blocks):
            m[r * bs : (r + 1) * bs, ctx : ctx + 1 + r * (new_ctx - 1) // max(n_blocks - 1, 1)] = True
            m[r * bs : (r + 1) * bs, ctx + new_ctx + r * bs : ctx + new_ctx + (r + 1) * bs] = True
        mask = m[None, None]
    head = target.lm_head

    def run():
        hid = draft(target_hidden=h, noise_embedding=noise, position_ids=pos[:, ctx:], attention_mask=mask,
                    past_key_values=cache, use_cache=True)
        return draft.compute_logits(hid, head).argmax(-1)

    with torch.inference_mode():
        for _ in range(3):
            run()
            crop(cache, ctx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            run()
            crop(cache, ctx)
        torch.cuda.synchronize()
        eager = (time.perf_counter() - t0) * 1000 / reps
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()
                crop(cache, ctx)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):      # the cache grows to ctx + new_ctx + q inside the graph; replays
            run()                      # recompute that same step, which is what we want to time
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        graph = st.elapsed_time(en) / reps
    del g, cache
    return eager, graph


def bench(name, model, args, qs, keep, rows):
    for q in qs:
        row = {"stage": name, "q": q}
        try:
            row["eager_ms"], row["graph_ms"] = stage(model, args.ctx, q, args.reps, keep)
        except Exception as e:  # noqa: BLE001
            print(f"[{name} q={q}] failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            row["eager_ms"] = row["graph_ms"] = float("nan")
        rows.append(row)
        print(row, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copies", nargs="*", default=["ao4:Qwen/Qwen3-8B", "bnb4:Qwen/Qwen3-8B"])
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--verify-qs", nargs="+", type=int, default=[1, 16, 32, 48, 64])
    ap.add_argument("--copy-qs", nargs="+", type=int, default=[16, 26, 32])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--only-copies", action="store_true")
    args = ap.parse_args()
    rows = []
    if args.only_copies:
        for spec in args.copies:
            try:
                m = load_mid(spec)
            except Exception as e:  # noqa: BLE001
                print(f"[{spec}] load failed: {type(e).__name__}: {e}", flush=True)   # full text (build logs)
                continue
            print(f"[{spec}] linear layer class: {type(m.model.layers[0].mlp.down_proj).__module__}."
                  f"{type(m.model.layers[0].mlp.down_proj).__name__}", flush=True)
            bench("copy_" + spec.split(":")[0], m, args, args.copy_qs, 16, rows)
            free(m)
            del m
        print_table(rows, ["stage", "q", "eager_ms", "graph_ms"], f"copy latency, batch 1, ctx {args.ctx}")
        save_json("bench_refresh_copies", {"args": vars(args), "rows": rows})
        return
    target = load_target(args.target)
    bench("verify_bf16", target, args, args.verify_qs, None, rows)
    from hspec.models import load_draft
    draft = load_draft(args.draft)
    for name, new_ctx, nb in (("draft_block1", 10, 1), ("draft_restarts3", 16, 3), ("draft_restarts1", 16, 1)):
        row = {"stage": name, "q": nb * draft.block_size}
        try:
            row["eager_ms"], row["graph_ms"] = draft_stage(draft, target, args.ctx, new_ctx, nb, args.reps)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            row["eager_ms"] = row["graph_ms"] = float("nan")
        rows.append(row)
        print(row, flush=True)
    free(draft)
    del draft
    free(target)
    del target
    for spec in args.copies:
        try:
            m = load_mid(spec)
        except Exception as e:  # noqa: BLE001
            print(f"[{spec}] load failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        bench("copy_" + spec.split(":")[0], m, args, args.copy_qs, 16, rows)
        free(m)
        del m
    print_table(rows, ["stage", "q", "eager_ms", "graph_ms"],
                f"stage latency, batch 1, ctx {args.ctx} (graph = CUDA-graph replay)")
    save_json("bench_refresh", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
