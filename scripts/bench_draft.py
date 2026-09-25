"""Drafter cost per call with CUDA graphs (no Python / launch overhead).

One drafter call = drafter forward over C fresh context features + a masked block of
size bs, then the LM head on the bs-1 draft positions. Captured without a KV cache on
static shapes, so C = small (e.g. 8) measures the memory-bound cost of a real call
(weights + LM head); C = 512 is an upper bound (re-projects 512 context features every
call). Attention over the cached prefix in a real call is small at these lengths.

  python scripts/bench_draft.py
Writes results/bench_draft.json {rows: [{bs, ctx, eager_ms, graph_ms}]}.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--block-sizes", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--ctxs", nargs="+", type=int, default=[8, 64, 512])
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    target = load_target(args.target)
    head = target.lm_head
    hidden = target.config.hidden_size
    draft = load_draft(args.draft)
    width = len(draft.target_layer_ids) * hidden
    dev = next(draft.parameters()).device
    rows = []
    for bs in args.block_sizes:
        for c in args.ctxs:
            h = torch.randn(1, c, width, device=dev, dtype=torch.bfloat16)
            noise = torch.randn(1, bs, hidden, device=dev, dtype=torch.bfloat16)
            pos = torch.arange(1000, 1000 + c + bs, device=dev).unsqueeze(0)

            def step():
                hid = draft(target_hidden=h, noise_embedding=noise, position_ids=pos,
                            past_key_values=None, use_cache=False)
                return draft.compute_logits(hid[:, 1:], head).argmax(-1)

            with torch.inference_mode():
                eager = []
                for i in range(args.reps + 5):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    step()
                    torch.cuda.synchronize()
                    if i >= 5:
                        eager.append((time.perf_counter() - t0) * 1000)
                graph_ms = float("nan")
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
                    gt = []
                    for i in range(args.reps + 5):
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        g.replay()
                        torch.cuda.synchronize()
                        if i >= 5:
                            gt.append((time.perf_counter() - t0) * 1000)
                    graph_ms = _median(gt)
                    del g
                except Exception as e:  # noqa: BLE001
                    print(f"graph capture failed (bs={bs}, ctx={c}): {type(e).__name__}: {str(e)[:200]}")
            row = {"bs": bs, "ctx": c, "eager_ms": _median(eager), "graph_ms": graph_ms}
            rows.append(row)
            print(row, flush=True)
    print_table(rows, ["bs", "ctx", "eager_ms", "graph_ms"], "drafter cost per call")
    save_json("bench_draft", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
