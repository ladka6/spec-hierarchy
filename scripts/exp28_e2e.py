"""Experiment 28: feature-refreshed decoding end to end (greedy, real loop).

Runs plain DFlash (chain) and the refreshed loop (hspec/refresh.py) on the same prompts and reports
mean accepted tokens per target verify (= tokens per round), target calls per token, wall-clock
tok/s and the per-stage time split (drafter pass 1, copy pass, restart passes, verify). Also
checks that the refreshed output matches plain DFlash (both are greedy-lossless; bf16 numerics can
make long outputs drift, so this is reported as a match rate, not asserted).

HF eager timing is dominated by per-call overhead, so tok/s here is a correctness / relative
number; the per-stage split plus the vLLM cost table give the realistic estimate.

  python scripts/exp28_e2e.py --copy bnb4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import ddtree_generate, two_stage_generate  # noqa: E402
from hspec.refresh import refresh_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--ddtree", type=int, default=0, help="also run DDTree with this budget (0 = off)")
    ap.add_argument("--out", default="exp28_e2e")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    copy = load_mid(args.copy)
    stops = stop_ids(target, tok)
    methods = {
        "dflash": lambda ids: two_stage_generate(draft, target, target, target, ids, args.max_new, stops),
        "refresh_pc": lambda ids: refresh_generate(draft, target, copy, ids, args.max_new, stops, ks=0,
                                                   use_pc=True, profile=True),
        "refresh_pc+conf2": lambda ids: refresh_generate(draft, target, copy, ids, args.max_new, stops, ks=2,
                                                         use_pc=True, profile=True),
        "refresh_conf2": lambda ids: refresh_generate(draft, target, copy, ids, args.max_new, stops, ks=2,
                                                      use_pc=False, profile=True),
    }
    if args.ddtree:
        methods[f"ddtree{args.ddtree}"] = lambda ids: ddtree_generate(draft, target, ids, args.max_new, stops,
                                                                     budget=args.ddtree)
    # warm-up (kernels, allocator)
    w = encode(tok, load_prompts(args.datasets[0], 1)[0])
    for f in methods.values():
        f(w)

    per = {m: [] for m in methods}
    rows = []
    for d in args.datasets:
        dm = {m: [] for m in methods}
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            ref = None
            for m, f in methods.items():
                r = f(ids)
                g = r.generated
                if ref is None:
                    ref = g
                k = min(len(g), len(ref))
                first_diff = next((i for i in range(k) if g[i] != ref[i]), k)
                rec = {"tokens": r.num_output_tokens, "time": r.decode_time, "rounds": len(r.rounds),
                       "target_calls": r.target_calls, "draft_calls": r.draft_calls,
                       "match_prefix": first_diff / max(len(ref), 1),
                       "timing": getattr(r, "timing", None), "tq": mean(r.tq) if r.tq else 0.0}
                dm[m].append(rec)
                per[m].append(rec)
        for m, recs in dm.items():
            rows.append(summ(d, m, recs))
        print(f"[{d}] done", flush=True)
    for m, recs in per.items():
        rows.append(summ("all", m, recs))
    base = {r["dataset"]: r["tok_s"] for r in rows if r["method"] == "dflash"}
    for r in rows:
        r["x_vs_dflash"] = r["tok_s"] / base[r["dataset"]] if base.get(r["dataset"]) else float("nan")
    print_table(rows, ["dataset", "method", "tau", "tok_s", "x_vs_dflash", "verify_q", "draft_calls_per_round",
                       "ms_draft1", "ms_copy", "ms_draft2", "ms_verify", "match"],
                "end-to-end greedy (HF eager): tau = tokens per target verify; ms_* = mean ms per round per stage")
    save_json(args.out, {"args": vars(args), "rows": rows})


def summ(d, m, recs):
    tokens = sum(r["tokens"] for r in recs)
    rounds = sum(r["rounds"] for r in recs)
    out = {"dataset": d, "method": m, "tau": tokens / max(rounds, 1),
           "tok_s": tokens / max(sum(r["time"] for r in recs), 1e-9),
           "verify_q": mean(r["tq"] for r in recs),
           "draft_calls_per_round": sum(r["draft_calls"] for r in recs) / max(rounds, 1),
           "match": mean(r["match_prefix"] for r in recs)}
    for k in ("draft1", "copy", "draft2", "verify"):
        tt = [r["timing"][k] for r in recs if r["timing"]]
        out[f"ms_{k}"] = 1000 * sum(tt) / max(rounds, 1) if tt else float("nan")
    return out


if __name__ == "__main__":
    main()
