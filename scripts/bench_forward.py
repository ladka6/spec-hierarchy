"""Forward latency of the target and candidate middle models at the shapes the pipeline uses.

Each forward processes q new tokens on top of a cached context of --ctx tokens, which is
what a verification pass does (q = 16 for a drafter block, q = P+1 for a target check).
This gives the cost ratio r = c_T / c that decides the optimal window P*.

  python scripts/bench_forward.py --mids bnb4:Qwen/Qwen3-8B ao4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import _make_cache  # noqa: E402

from hspec.models import free, load_mid, load_target  # noqa: E402
from hspec.pipeline import crop  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


@torch.inference_mode()
def time_forward(model, ctx: int, q: int, reps: int, vocab: int) -> float:
    device = next(model.parameters()).device
    cache = _make_cache(model.config)
    prompt = torch.randint(0, vocab, (1, ctx), device=device)
    model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
    block = torch.randint(0, vocab, (1, q), device=device)
    pos = torch.arange(ctx, ctx + q, device=device).unsqueeze(0)
    times = []
    for i in range(reps + 3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(block, position_ids=pos, past_key_values=cache, use_cache=True,
              output_hidden_states=True)
        end.record()
        torch.cuda.synchronize()
        crop(cache, ctx)
        if i >= 3:
            times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--mids", nargs="+", default=["bnb4:Qwen/Qwen3-8B", "ao4:Qwen/Qwen3-8B"])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--qs", nargs="+", type=int, default=[1, 16, 33, 65, 129])
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    rows = []
    target = load_target(args.target)
    vocab = target.config.vocab_size
    row = {"model": f"bf16:{args.target}"}
    for q in args.qs:
        row[f"q={q}"] = time_forward(target, args.ctx, q, args.reps, vocab)
    rows.append(row)
    print(row, flush=True)
    del target
    free()

    for spec in args.mids:
        try:
            m = load_mid(spec)
        except Exception as e:  # noqa: BLE001
            print(f"could not load {spec}: {e}")
            continue
        row = {"model": spec}
        for q in args.qs:
            row[f"q={q}"] = time_forward(m, args.ctx, q, args.reps, vocab)
        rows.append(row)
        print(row, flush=True)
        del m
        free()

    print_table(rows, ["model"] + [f"q={q}" for q in args.qs], f"median forward latency (ms), ctx={args.ctx}")
    save_json("bench_forward", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
