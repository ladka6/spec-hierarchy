"""Experiment 24: how much could drafting ahead (async / PEARL-style) gain at most?

Runs the official DFlash / DDTree loops and splits each round's time into drafter pass, tree
build (+compile), target verify, commit and the rest. Drafting ahead can at best overlap the
drafter side (draft + tree) with verification:

  ceil_all     T / (T - draft - tree)          every round's drafting hidden (perfect guess)
  ceil_full    T / (T - f_full*(draft + tree)) only rounds that accept the whole block let a
                                               PEARL-style "assume full acceptance" guess survive
  slack        verify / draft                  how many drafter passes fit into one verify for free

  python scripts/exp24_ceiling.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

from ddtree_official.ddtree import ddtree_generate as off_ddtree, maybe_enable_cpp_compact  # noqa: E402
from ddtree_official.dflash import dflash_generate as off_dflash  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--budgets", nargs="+", type=int, default=[32, 64, 128, 256])
    args = ap.parse_args()

    maybe_enable_cpp_compact(True)
    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, mask = draft.block_size, draft.mask_token_id
    methods = ["ar", "dflash"] + [f"ddtree_tb{b}" for b in args.budgets]

    def run(m, ids):
        if m == "ar":
            return off_dflash(draft, target, ids, mask, args.max_new, 1, stops)
        if m == "dflash":
            return off_dflash(draft, target, ids, mask, args.max_new, bs, stops)
        return off_ddtree(draft, target, ids, mask, args.max_new, bs, stops, tree_budget=int(m[9:]))

    warm = encode(tok, load_prompts(args.datasets[0], 1)[0])
    for m in methods:
        run(m, warm)

    acc = {m: defaultdict(float) for m in methods}
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            for m in methods:
                r = run(m, ids)
                a = acc[m]
                a["T"] += r.time_per_output_token * r.num_output_tokens
                a["tokens"] += r.num_output_tokens
                a["rounds"] += len(r.acceptance_lengths) if getattr(r, "acceptance_lengths", None) else r.num_output_tokens
                a["full"] += sum(x >= bs for x in (r.acceptance_lengths or []))
                for k, v in r.stage_times.items():
                    a[k] += v
        print(f"[{d}] done", flush=True)

    ar_tpot = acc["ar"]["T"] / acc["ar"]["tokens"]
    rows = []
    for m in methods[1:]:
        a = acc[m]
        R = a["rounds"]
        T = a["T"]
        drf = a["draft"]
        tree = a.get("tree_build", 0.0) + a.get("tree_compile", 0.0)
        ver, com = a["verify"], a["commit"]
        f_full = a["full"] / R
        rows.append({
            "method": m, "speedup": ar_tpot / (T / a["tokens"]), "tau": a["tokens"] / R,
            "ms_round": 1e3 * T / R, "draft_ms": 1e3 * drf / R, "tree_ms": 1e3 * tree / R,
            "verify_ms": 1e3 * ver / R, "commit_ms": 1e3 * com / R,
            "other_ms": 1e3 * (T - drf - tree - ver - com) / R,
            "draft_share": (drf + tree) / T, "f_full": f_full,
            "ceil_all": T / (T - drf - tree), "ceil_full": T / (T - f_full * (drf + tree)),
            "slack": ver / max(drf, 1e-9),
        })
    print_table(rows, ["method", "speedup", "tau", "ms_round", "draft_ms", "tree_ms", "verify_ms", "commit_ms",
                       "other_ms", "draft_share", "f_full", "ceil_all", "ceil_full", "slack"],
                "per-round time split and the ceiling for drafting ahead of verification")
    save_json("exp24_ceiling", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
