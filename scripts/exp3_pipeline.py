"""Experiment 3: the full three-stage pipeline vs baselines.

Configs per prompt (all greedy):
  ar          target alone, autoregressive
  dflash      vanilla DFlash (drafter -> target)
  3s-P        drafter -> mid (+ mid features for the drafter) -> target checks every >= P
              pending tokens, for each P in --windows
  3s-adapt    same, window chosen online from P* = ln(1 + eps c_T/c) / eps

Reports tokens/s, speedup over ar and dflash, target calls per token, tokens accepted per
target check, and whether the output is identical to dflash (it must be, up to numerical
noise: all configs are lossless w.r.t. the target's greedy decoding).

Usage:
  python scripts/exp3_pipeline.py --n 10 --mids bnb4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import free, gpu_mem_gb, load_draft, load_mid, load_target, load_tokenizer, same_hidden_space  # noqa: E402
from hspec.pipeline import WindowPolicy, ar_generate, three_stage_generate, two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def match(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of the shorter sequence that is an identical prefix."""
    k = min(len(a), len(b))
    if k == 0:
        return 1.0
    diff = (a[:k] != b[:k]).nonzero()
    return 1.0 if diff.numel() == 0 else diff[0].item() / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mids", nargs="+", default=["bnb4:Qwen/Qwen3-8B"])
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 8, 16, 32, 64, 128])
    ap.add_argument("--no-ar", action="store_true", help="skip the slow autoregressive baseline")
    ap.add_argument("--block-size", type=int, default=None)
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    prompts = {d: load_prompts(d, args.n) for d in args.datasets}
    bs = args.block_size

    # warmup
    w = encode(tok, prompts[args.datasets[0]][0])
    two_stage_generate(draft, target, target, target, w, 64, stops, bs)

    records = []
    ref = {}
    for d in args.datasets:
        for i, p in enumerate(prompts[d]):
            ids = encode(tok, p)
            r = two_stage_generate(draft, target, target, target, ids, args.max_new, stops, bs)
            ref[(d, i)] = r.generated.cpu()
            records.append({"config": "dflash", "mid": "-", "dataset": d, "i": i, **r.summary(),
                            "match": 1.0})
            if not args.no_ar:
                a = ar_generate(target, ids, args.max_new, stops)
                records.append({"config": "ar", "mid": "-", "dataset": d, "i": i, **a.summary(),
                                "match": match(a.generated.cpu(), ref[(d, i)])})
        print(f"[baselines | {d}] done", flush=True)

    for spec in args.mids:
        mid = load_mid(spec)
        if not same_hidden_space(target, mid):
            print(f"skip {spec}: the drafter needs features in the target's hidden space")
            free(mid)
            continue
        policies = [(f"3s-{p}", lambda p=p: WindowPolicy("fixed", window=p)) for p in args.windows]
        policies.append(("3s-adapt", lambda: WindowPolicy("adaptive", window=32)))
        three_stage_generate(draft, target, mid, w, 64, stops, WindowPolicy("fixed", window=16), bs)
        for name, make_policy in policies:
            policy = make_policy()   # adaptive state is shared across prompts of one run
            for d in args.datasets:
                for i, p in enumerate(prompts[d]):
                    r = three_stage_generate(draft, target, mid, encode(tok, p), args.max_new,
                                             stops, policy, bs)
                    records.append({"config": name, "mid": spec, "dataset": d, "i": i, **r.summary(),
                                    "match": match(r.generated.cpu(), ref[(d, i)]),
                                    "final_window": policy.current_window(), "eps": policy.eps()})
            print(f"[{name} | {spec}] done  window_now={policy.current_window()}", flush=True)
        free(mid)

    # aggregate
    rows = []
    keys = sorted({(r["config"], r["mid"], r["dataset"]) for r in records},
                  key=lambda k: (k[2], k[1], k[0]))
    for cfg, mid, d in keys:
        rs = [r for r in records if (r["config"], r["mid"], r["dataset"]) == (cfg, mid, d)]
        tot_tok = sum(r["num_output_tokens"] for r in rs)
        tot_t = sum(r["decode_time"] for r in rs)
        row = {"dataset": d, "config": cfg, "mid": mid, "tok/s": tot_tok / tot_t,
               "tgt_calls/tok": mean(r["target_calls_per_token"] for r in rs),
               "mid_calls/tok": mean(r["mid_calls_per_token"] for r in rs),
               "tau": mean(r["mean_round_len"] for r in rs),
               "acc/check": mean(r.get("mean_accepted_per_check", float("nan")) for r in rs),
               "full_acc": mean(r.get("full_accept_rate", float("nan")) for r in rs),
               "match": mean(r["match"] for r in rs)}
        rows.append(row)
    for row in rows:
        base = {r["config"]: r["tok/s"] for r in rows if r["dataset"] == row["dataset"]}
        row["x_dflash"] = row["tok/s"] / base["dflash"]
        if "ar" in base:
            row["x_ar"] = row["tok/s"] / base["ar"]

    print_table(rows, ["dataset", "config", "mid", "tok/s", "x_ar", "x_dflash", "tgt_calls/tok",
                       "mid_calls/tok", "tau", "acc/check", "full_acc", "match"],
                "three-stage pipeline")
    print(f"peak GPU memory: {gpu_mem_gb():.1f} GB")
    save_json("exp3_pipeline", {"args": vars(args), "rows": rows, "records": records})


if __name__ == "__main__":
    main()
