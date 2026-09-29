"""Experiment 18 (idea A): conditional re-drafting at the uncertain spot, offline.

DDTree builds its tree from DFlash's independent per-position distributions: a sibling token
at depth j gets the same (wrong) guesses for depths > j as the top-1 path. Here, after the
normal drafter pass, find the first depth j* on the top-1 path (spine) whose confidence is
below tau, take the top-M tokens there and re-draft the rest of the block for each of them
with the lag-tolerant drafter (tokens mode: spine prefix + branch token given as inputs), in
one batched call. The tree is built from the original candidates plus these conditional
continuations (path log-prob scores, best B prefix-closed), as in exp15.

Offline on target-greedy trajectories (Qwen3-8B), anchors every `stride` positions; accepted
tokens per target pass (incl. bonus) at equal node budget vs DDTree and DDTree at 2B.
Cost of the idea: one extra batched drafter pass per round (~0.17-0.27 of a target pass),
so it needs a clear acceptance gain to pay off.

  python scripts/exp18_redraft.py --lag-draft /scratch-shared/$USER/hspec_lagft/drafter_lag16
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp15_blocktree import accepted, select, tree_paths  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.tree import build_ddtree  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


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
    ap.add_argument("--budgets", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--branches", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--penalties", nargs="+", type=float, default=[0.0, -1.0])
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    lagd = load_draft(args.lag_draft)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    D = bs - 1
    Mmax = max(args.branches)

    acc = defaultdict(list)
    stats = defaultdict(list)
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            for s in range(n0, len(x) - 2 * bs - 1, args.stride):
                truth = x[s + 1 : s + 1 + 2 * D].tolist()
                lg0 = draft_logits_lagged(draft, target, feats, x, s, 0, bs).float()
                lq0 = torch.log_softmax(lg0, -1).cpu()
                spine = lg0.argmax(-1).tolist()
                conf = lq0.max(-1).values.exp().tolist()
                cum = torch.cumsum(lq0[torch.arange(D), torch.tensor(spine)], 0).tolist()
                base = {B: tree_paths(build_ddtree(lg0, B), lq0) for B in set(args.budgets + [2 * b for b in args.budgets])}
                for B in args.budgets:
                    acc[("ddtree", B, d)].append(accepted(set(base[B]), truth))
                    acc[("ddtree2x", B, d)].append(accepted(set(base[2 * B]), truth))
                spine_ok = 0
                while spine_ok < D and spine[spine_ok] == truth[spine_ok]:
                    spine_ok += 1
                for tau in args.taus:
                    j = next((i for i in range(D) if conf[i] < tau), None)   # 0-based depth index of the branch
                    if j is None:
                        for B in args.budgets:
                            for M in args.branches:
                                for pen in args.penalties:
                                    acc[(f"redraft t{tau} M{M} p{pen:g}", B, d)].append(accepted(set(base[B]), truth))
                        stats[(tau, "branched")].append(0.0)
                        continue
                    stats[(tau, "branched")].append(1.0)
                    stats[(tau, "spine_wrong_at_branch")].append(float(spine_ok == j))
                    top = lq0[j].topk(Mmax)
                    btoks, blq = top.indices.tolist(), top.values.tolist()
                    pre = tuple(spine[:j])
                    pre_score = cum[j - 1] if j > 0 else 0.0
                    stats[(tau, "truth_in_topM")].append(float(truth[j] in btoks) if spine_ok >= j else float("nan"))
                    conts = []
                    for b in btoks:                                   # re-draft after spine[:j] + b
                        xs = x.clone()
                        xs[s + 1 : s + 1 + j] = torch.tensor(spine[:j], device=x.device)
                        xs[s + 1 + j] = b
                        lgb = draft_logits_lagged(lagd, target, feats, xs, s + j + 1, j + 1, bs).float()
                        lqb = torch.log_softmax(lgb, -1).cpu()
                        conts.append({B: tree_paths(build_ddtree(lgb, B), lqb) for B in args.budgets})
                    for B in args.budgets:
                        for M in args.branches:
                            for pen in args.penalties:
                                cands = dict(base[B])
                                for i in range(D):
                                    cands.setdefault(tuple(spine[: i + 1]), cum[i])
                                for bi in range(M):
                                    head = pre + (btoks[bi],)
                                    hs = pre_score + blq[bi]
                                    cands[head] = max(cands.get(head, float("-inf")), hs)
                                    for path, sc in conts[bi][B].items():
                                        full = head + path
                                        v = hs + sc + pen
                                        if v > cands.get(full, float("-inf")):
                                            cands[full] = v
                                acc[(f"redraft t{tau} M{M} p{pen:g}", B, d)].append(accepted(select(cands, B), truth))
        print(f"[{d}] done", flush=True)

    rows = []
    names = sorted({k[0] for k in acc}, key=lambda n: (n != "ddtree", n != "ddtree2x", n))
    for B in args.budgets:
        ref = mean(sum((acc[("ddtree", B, dd)] for dd in args.datasets), []))
        for nm in names:
            allv, row = [], {"method": nm, "budget": B}
            for dd in args.datasets:
                v = acc[(nm, B, dd)]
                row[dd] = mean(v)
                allv += v
            row["ALL"] = mean(allv)
            row["vs_ddtree"] = row["ALL"] / ref
            rows.append(row)
    print_table(rows, ["method", "budget", "ALL", "vs_ddtree"] + args.datasets,
                "accepted tokens per target pass at equal node budget (incl. bonus)")
    srows = [{"tau": t, "branched": mean(stats[(t, "branched")]),
              "spine_wrong_at_branch": mean(stats[(t, "spine_wrong_at_branch")]),
              "truth_in_topM(max)": mean(v for v in stats[(t, "truth_in_topM")] if v == v)} for t in args.taus]
    print_table(srows, ["tau", "branched", "spine_wrong_at_branch", "truth_in_topM(max)"],
                "where the re-draft branches: share of rounds, and how often the spine fails exactly there")
    save_json("exp18_redraft", {"args": vars(args), "rows": rows, "stats": srows})


if __name__ == "__main__":
    main()
