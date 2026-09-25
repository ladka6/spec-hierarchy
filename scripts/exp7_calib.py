"""Experiment 7: does the middle model know where the target will disagree?

On target-greedy trajectories (teacher forcing), for every generated position compare the
middle model's greedy token with the target's. Reports:
  eps          per-token disagreement rate
  h            P(target token == mid's 2nd choice | disagreement)   (hedging catch ceiling)
  reliability  disagreement rate by mid top-1 probability bin
  fork table   for a fork threshold thr: share of positions below thr (fork rate), share of
               disagreements below thr (recall) and share caught by the 2nd choice (catch)

  python scripts/exp7_calib.py --n 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

BINS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.97, 0.995, 1.0001]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mid", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--thrs", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    mid = load_mid(args.mid)
    stops = stop_ids(target, tok)

    rel_rows, fork_rows, summary = [], [], []
    for d in args.datasets:
        p1s, dis, top2hit = [], [], []
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            seq = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids
            if seq.shape[1] - n0 < 2:
                continue
            logits = mid(seq).logits[0, n0 - 1 : -1].float()      # predicts positions n0 .. T-1
            truth = seq[0, n0:]
            top = torch.softmax(logits, -1).topk(2, -1)
            p1s.append(top.values[:, 0].cpu())
            dis.append((top.indices[:, 0] != truth).cpu())
            top2hit.append((top.indices[:, 1] == truth).cpu())
            del logits
        p1, dz, t2 = torch.cat(p1s), torch.cat(dis), torch.cat(top2hit)
        nd = int(dz.sum())
        summary.append({"dataset": d, "positions": len(p1), "eps": float(dz.float().mean()),
                        "h_top2_given_dis": float((t2 & dz).sum()) / max(nd, 1)})
        for lo, hi in zip(BINS, BINS[1:]):
            m = (p1 >= lo) & (p1 < hi)
            rel_rows.append({"dataset": d, "p1_bin": f"[{lo:.3f},{min(hi, 1):.3f})", "share": float(m.float().mean()),
                             "dis_rate": float(dz[m].float().mean()) if m.any() else float("nan"),
                             "share_of_dis": float((dz & m).sum()) / max(nd, 1)})
        for thr in args.thrs:
            m = p1 < thr
            fork_rows.append({"dataset": d, "thr": thr, "fork_rate": float(m.float().mean()),
                              "recall": float((dz & m).sum()) / max(nd, 1),
                              "catch": float((dz & m & t2).sum()) / max(nd, 1)})
        print(f"[{d}] done", flush=True)

    print_table(summary, ["dataset", "positions", "eps", "h_top2_given_dis"],
                "mid vs target disagreement (teacher forced on target trajectories)")
    print_table(rel_rows, ["dataset", "p1_bin", "share", "dis_rate", "share_of_dis"],
                "reliability: disagreement rate by mid top-1 probability")
    print_table(fork_rows, ["dataset", "thr", "fork_rate", "recall", "catch"],
                "hedging ceiling: fork where top-1 < thr")
    save_json("exp7_calib", {"args": vars(args), "summary": summary, "reliability": rel_rows, "fork": fork_rows})


if __name__ == "__main__":
    main()
