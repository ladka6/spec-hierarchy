"""Experiment 15: block trees. Can one target pass verify deeper than one drafter block?

DDTree's tree is limited to one drafter block (depth bs-1). A block tree adds continuation
blocks hanging off the drafter's top-1 path (the spine) at cut depths c: a lag-tolerant
drafter (train_lag.py) drafts positions s+c+1 .. s+c+bs-1 without target features for the
lagged span (tokens mode: the spine tokens are given as inputs). All candidates are scored
by their path log-probability (continuation paths: spine prefix + continuation, plus an
optional per-continuation penalty) and the best B form the tree, as in DDTree.

Offline on target-greedy trajectories (the target accepts exactly the longest tree path
that matches the trajectory), anchors every `stride` positions. Reports accepted tokens
per target pass (incl. the bonus token) at equal node budget B, plus DDTree at 2B.

  python scripts/exp15_blocktree.py --lag-draft /scratch-shared/$USER/hspec_lagft/drafter_lag16
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
from hspec.tree import build_ddtree  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def tree_paths(tree, logq):
    """{token path tuple: log-prob} for every node of a DDTree built from logq [D, V]."""
    paths, out = [], {}
    for i, (t, p) in enumerate(zip(tree.tokens, tree.parents)):
        path = (paths[p] if p >= 0 else ()) + (t,)
        paths.append(path)
        prev = out[paths[p]] if p >= 0 else 0.0
        out[path] = prev + float(logq[len(path) - 1, t])
    return out


def select(cands: dict, budget: int) -> set:
    """Best `budget` paths by score, keeping the set prefix-closed."""
    chosen = set()
    for path, _ in sorted(cands.items(), key=lambda kv: -kv[1]):
        if len(chosen) >= budget:
            break
        if len(path) == 1 or path[:-1] in chosen:
            chosen.add(path)
    return chosen


def accepted(chosen: set, truth: list) -> int:
    """Tokens the target accepts from this tree (longest matching path) + its bonus token."""
    a = 0
    while a < len(truth) and tuple(truth[: a + 1]) in chosen:
        a += 1
    return a + 1


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
    ap.add_argument("--cuts", nargs="+", default=["15", "11,15", "7,11,15"],
                    help="cut-depth sets (comma separated); depth bs-1 = end of the first block")
    ap.add_argument("--penalties", nargs="+", type=float, default=[0.0, -1.0, -2.0],
                    help="log-score added to every continuation node")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    lagd = load_draft(args.lag_draft)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    D = bs - 1
    cutsets = [[int(c) for c in cs.split(",")] for cs in args.cuts]
    all_cuts = sorted({c for cs in cutsets for c in cs})
    assert max(all_cuts) <= D

    acc = defaultdict(list)
    reach = defaultdict(list)
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            for s in range(n0, len(x) - 2 * bs - 1, args.stride):
                truth = x[s + 1 : s + 1 + D + D].tolist()
                # block k: standard DFlash block with fresh features
                lg0 = draft_logits_lagged(draft, target, feats, x, s, 0, bs).float()
                lq0 = torch.log_softmax(lg0, -1).cpu()
                spine = lg0.argmax(-1).tolist()
                base = {B: tree_paths(build_ddtree(lg0, B), lq0) for B in set(args.budgets + [2 * b for b in args.budgets])}
                for B in args.budgets:
                    acc[("ddtree", B, d)].append(accepted(set(base[B]), truth))
                    acc[("ddtree2x", B, d)].append(accepted(set(base[2 * B]), truth))
                # continuations off the spine, drafted without features for the lagged span
                xs = x.clone()
                xs[s + 1 : s + 1 + D] = torch.tensor(spine, device=x.device)
                spine_score = torch.cumsum(lq0[torch.arange(D), torch.tensor(spine)], 0).tolist()
                conts = {}
                for c in all_cuts:
                    lgc = draft_logits_lagged(lagd, target, feats, xs, s + c, c, bs).float()
                    lqc = torch.log_softmax(lgc, -1).cpu()
                    conts[c] = {B: tree_paths(build_ddtree(lgc, B), lqc) for B in args.budgets}
                    ok = spine[:c] == truth[:c]
                    reach[(c, d)].append(float(ok))
                for B in args.budgets:
                    for ci, cs in enumerate(cutsets):
                        for pen in args.penalties:
                            cands = dict(base[B])
                            for c in cs:
                                pre = tuple(spine[:c])
                                for path, sc in conts[c][B].items():
                                    full = pre + path
                                    v = spine_score[c - 1] + sc + pen
                                    if v > cands.get(full, float("-inf")):
                                        cands[full] = v
                            for j in range(1, D + 1):          # spine prefixes must exist to hang continuations
                                cands.setdefault(tuple(spine[:j]), spine_score[j - 1])
                            acc[(f"block[{args.cuts[ci]}]p{pen:g}", B, d)].append(accepted(select(cands, B), truth))
        print(f"[{d}] done", flush=True)

    rows = []
    names = sorted({k[0] for k in acc}, key=lambda n: (n != "ddtree", n != "ddtree2x", n))
    for B in args.budgets:
        ref = mean(sum((acc[("ddtree", B, d)] for d in args.datasets), []))
        for nm in names:
            allv, row = [], {"method": nm, "budget": B}
            for d in args.datasets:
                v = acc[(nm, B, d)]
                row[d] = mean(v)
                allv += v
            row["ALL"] = mean(allv)
            row["vs_ddtree"] = row["ALL"] / ref
            rows.append(row)
    print_table(rows, ["method", "budget", "ALL", "vs_ddtree"] + args.datasets,
                "accepted tokens per target pass at equal node budget (incl. bonus)")
    rrows = [{"cut": c, **{d: mean(reach[(c, d)]) for d in args.datasets}} for c in all_cuts]
    print_table(rrows, ["cut"] + args.datasets, "P(spine correct up to the cut depth)")
    save_json("exp15_blocktree", {"args": vars(args), "rows": rows, "reach": rrows})


if __name__ == "__main__":
    main()
