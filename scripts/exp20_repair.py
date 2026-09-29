"""Experiment 20: does a draft survive a boundary correction? (bounds "verified-prefix repair")

For each anchor s: draft a block (fresh features), find the first position where the drafter's
top-1 path leaves the target trajectory (index a: the drafter guessed spine[a], the target
wrote x* = truth[a] at position p = s+1+a). Then compare continuations after x*:

  reuse      the old draft's tokens after the boundary, unchanged (free: what a repair
             module has to beat)
  fresh      a normal DFlash redraft from x* with target features up to p (today's DFlash)
  lag        a redraft from x* with features only up to s and the accepted tokens + x* as
             inputs (lag-trained drafter): a redraft that has no new target features, i.e.
             roughly the best a repair from stale state could do with a full drafter pass
  lag-orig   the same with the original drafter (not trained for lag)

Accepted tokens are counted within the old draft's remaining horizon (D-1-a positions) so all
four are comparable; `fresh_full` counts the whole new block. `survival` = share of those
positions where the old draft's token equals the fresh redraft's token.

  python scripts/exp20_repair.py --lag-draft /scratch-shared/$USER/hspec_lagft/drafter_lag16
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def prefix_match(pred, truth):
    a = 0
    while a < len(pred) and a < len(truth) and pred[a] == truth[a]:
        a += 1
    return a


def bucket(a):
    return "a=0" if a == 0 else ("a=1-3" if a <= 3 else ("a=4-7" if a <= 7 else "a>=8"))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--lag-draft", required=True)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    lagd = load_draft(args.lag_draft)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    D = bs - 1

    rec = defaultdict(list)
    n_rounds = n_rej = 0
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            for s in range(n0, len(x) - 2 * bs - 2, args.stride):
                truth = x[s + 1 : s + 1 + 2 * D].tolist()
                spine = draft_logits_lagged(draft, target, feats, x, s, 0, bs).argmax(-1).tolist()
                n_rounds += 1
                a = prefix_match(spine, truth)
                if a >= D - 1:                         # no rejection, or nothing left after it
                    continue
                n_rej += 1
                pos = s + 1 + a                        # position of the correction x*
                after = truth[a + 1 :]                 # truth after x*
                h = D - 1 - a                          # old draft's remaining horizon
                reuse = spine[a + 1 :]
                fresh = draft_logits_lagged(draft, target, feats, x, pos, 0, bs).argmax(-1).tolist()
                lag = draft_logits_lagged(lagd, target, feats, x, pos, a + 1, bs).argmax(-1).tolist()
                lag_o = draft_logits_lagged(draft, target, feats, x, pos, a + 1, bs).argmax(-1).tolist()
                r = {"dataset": d, "a": a, "h": h,
                     "reuse": prefix_match(reuse, after[:h]),
                     "fresh": prefix_match(fresh[:h], after[:h]),
                     "lag": prefix_match(lag[:h], after[:h]),
                     "lag_orig": prefix_match(lag_o[:h], after[:h]),
                     "fresh_full": prefix_match(fresh, after),
                     "survival": mean(float(u == v) for u, v in zip(reuse, fresh[:h]))}
                rec[bucket(a)].append(r)
                rec["ALL"].append(r)
        print(f"[{d}] done", flush=True)

    rows = []
    for b in ["ALL", "a=0", "a=1-3", "a=4-7", "a>=8"]:
        rs = rec[b]
        if not rs:
            continue
        row = {"boundary": b, "share": len(rs) / max(n_rej, 1), "horizon": mean(r["h"] for r in rs)}
        for k in ("reuse", "lag_orig", "lag", "fresh", "fresh_full", "survival"):
            row[k] = mean(r[k] for r in rs)
        row["reuse/fresh"] = row["reuse"] / row["fresh"] if row["fresh"] else float("nan")
        row["lag/fresh"] = row["lag"] / row["fresh"] if row["fresh"] else float("nan")
        rows.append(row)
    print(f"{n_rej} of {n_rounds} rounds have a rejection with positions left after it", flush=True)
    print_table(rows, ["boundary", "share", "horizon", "reuse", "lag_orig", "lag", "fresh", "fresh_full", "survival",
                       "reuse/fresh", "lag/fresh"],
                "accepted tokens after the correction, within the old draft's horizon (a = accepted before the boundary)")
    save_json("exp20_repair", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
