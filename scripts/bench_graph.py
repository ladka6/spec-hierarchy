"""Per-forward cost of target and middle-model variants under CUDA graphs, same framework.

Each model gets a static KV cache holding a `ctx`-token prefix; for every query length q
a forward over q new tokens (at fixed cache positions) is captured once as a CUDA graph
and replayed. This removes Python / launch overhead and the subtraction tricks of the
vLLM benchmark, so c_mid(q) / c_target(q) is a like-for-like ratio.

Also measures the drafter at batch 1, 2, 4, 8 (hedging extends several branches in one
batched drafter call).

  python scripts/bench_graph.py --mids ao4:Qwen/Qwen3-8B bnb4:Qwen/Qwen3-8B
Writes results/bench_graph.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import free, load_draft, load_mid, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _time(fn, reps):
    out = []
    for i in range(reps + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if i >= 5:
            out.append((time.perf_counter() - t0) * 1000)
    return _median(out)


@torch.inference_mode()
def bench_model(name, model, ctx, qs, reps):
    from transformers import StaticCache

    dev = next(model.parameters()).device
    vocab = model.config.vocab_size
    row = {"model": name}
    cache = StaticCache(config=model.config, max_cache_len=ctx + max(qs) + 8)
    prefix = torch.randint(0, vocab, (1, ctx), device=dev)
    model(prefix, cache_position=torch.arange(ctx, device=dev), past_key_values=cache, use_cache=True,
          logits_to_keep=1)
    for q in qs:
        ids = torch.randint(0, vocab, (1, q), device=dev)
        cpos = torch.arange(ctx, ctx + q, device=dev)
        pos = cpos.unsqueeze(0)

        def step():
            return model(ids, position_ids=pos, cache_position=cpos, past_key_values=cache, use_cache=True)

        row[f"eager_q{q}"] = _time(step, reps)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step()
            row[f"q{q}"] = _time(g.replay, reps)
            del g
        except Exception as e:  # noqa: BLE001
            row[f"q{q}"] = float("nan")
            print(f"[{name} q={q}] graph capture failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print(row, flush=True)
    return row


@torch.inference_mode()
def bench_draft_batch(draft, target, batches, reps):
    dev = next(draft.parameters()).device
    hidden = target.config.hidden_size
    width = len(draft.target_layer_ids) * hidden
    bs, ctx = draft.block_size, 8
    rows = []
    for b in batches:
        h = torch.randn(b, ctx, width, device=dev, dtype=torch.bfloat16)
        noise = torch.randn(b, bs, hidden, device=dev, dtype=torch.bfloat16)
        pos = torch.arange(1000, 1000 + ctx + bs, device=dev).unsqueeze(0).expand(b, -1)

        def step():
            hid = draft(target_hidden=h, noise_embedding=noise, position_ids=pos, past_key_values=None,
                        use_cache=False)
            return draft.compute_logits(hid[:, 1:], target.lm_head).argmax(-1)

        row = {"batch": b, "eager_ms": _time(step, reps)}
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step()
            row["graph_ms"] = _time(g.replay, reps)
        except Exception as e:  # noqa: BLE001
            row["graph_ms"] = float("nan")
            print(f"[draft batch {b}] graph capture failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        rows.append(row)
        print(row, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mids", nargs="+", default=["ao4:Qwen/Qwen3-8B", "bnb4:Qwen/Qwen3-8B"])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--qs", nargs="+", type=int, default=[1, 8, 17, 33, 49, 65, 97, 129])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    rows = []
    target = load_target(args.target)
    rows.append(bench_model("bf16", target, args.ctx, args.qs, args.reps))
    draft = load_draft(args.draft)
    drows = bench_draft_batch(draft, target, args.batches, args.reps)
    del draft, target
    free()
    for spec in args.mids:
        try:
            m = load_mid(spec)
            rows.append(bench_model(spec, m, args.ctx, args.qs, args.reps))
            del m
        except Exception as e:  # noqa: BLE001
            print(f"[{spec}] failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
        free()

    cols = ["model"] + [f"q{q}" for q in args.qs]
    print_table(rows, cols, f"forward cost with CUDA graphs (ms), static cache, ctx {args.ctx}")
    print_table(rows, ["model"] + [f"eager_q{q}" for q in args.qs], "same, eager (ms)")
    base = rows[0]
    ratio = [{"model": r["model"], **{f"q{q}": r[f"q{q}"] / base[f"q{q}"] for q in args.qs}} for r in rows[1:]]
    print_table(ratio, cols, "middle model cost / bf16 target cost (graphs)")
    d1 = drows[0]["graph_ms"]
    for r in drows:
        r["beta"] = (r["graph_ms"] / d1 - 1) / (r["batch"] - 1) if r["batch"] > 1 else 0.0
    print_table(drows, ["batch", "eager_ms", "graph_ms", "beta"], "drafter cost vs batch (beta = extra cost per extra branch)")
    save_json("bench_graph", {"args": vars(args), "rows": rows, "ratio": ratio, "draft": drows})


if __name__ == "__main__":
    main()
