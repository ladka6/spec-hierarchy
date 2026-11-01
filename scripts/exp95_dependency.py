"""Experiment 95: how sparse are within-block dependencies in the target's own text? (frozen Qwen3-8B, T=0)

Diagnostic for "train the target so a mistake only invalidates its descendants": if real text has dense within-block
dependencies, a sparse dependency graph (or a short latent scaffold) has little room.

Per block of B tokens y_1..y_B of the target's greedy text after anchor s: for each i in 1..B-1, replace y_i by the
target's second choice at that position (a plausible alternative), keep every other token, and run the target. For each
k > i, record whether the target's argmax at k (given the perturbed prefix) differs from y_k. That gives an influence
matrix I[i, k]: does k depend on i, with everything else in the block fixed.

Reports
  1. P(argmax at i+d changes | y_i changed) by distance d
  2. per perturbation: how many later positions change (out-degree) and the share of the suffix that stays valid
  3. dependency graph per block: mean number of in-block parents per position, longest dependency chain (critical
     path) vs B; also the share of changes at positions where the target itself is near a tie (margin < 0.1)

  python scripts/exp95_dependency.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def longest_chain(edges, B):
    """Longest path (in nodes) of the DAG on 1..B with edges i -> k (i < k)."""
    best = [1] * (B + 1)
    for k in range(1, B + 1):
        for i in edges.get(k, ()):
            best[k] = max(best[k], best[i] + 1)
    return max(best[1:])


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16", help="only to generate the text quickly")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--tie", type=float, default=0.1)
    ap.add_argument("--out", default="exp95_dependency")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    B = args.block
    dev = next(target.parameters()).device

    by_d = defaultdict(lambda: [0, 0])          # d -> [changed, total]
    outdeg, retained = [], []
    indeg, chains, tie_changes, n_changes = [], [], 0, 0
    outbucket = Counter()
    n_blocks = 0
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            full = torch.softmax(target(x[None], logits_to_keep=0).logits[0].float(), -1)       # [T, V]
            top2 = full.topk(2, -1)
            for s in range(n0, len(x) - B - 1, args.stride):
                n_blocks += 1
                y = x[s + 1 : s + 1 + B]                                   # y_1..y_B (absolute s+1..s+B)
                alt = top2.indices[s : s + B, 1]                           # target's 2nd choice for y_1..y_B
                margin = (top2.values[s : s + B, 0] - top2.values[s : s + B, 1]).tolist()
                seqs = []
                for i in range(1, B):                                      # perturb y_i (1-based), i < B
                    z = x[: s + 1 + B].clone()
                    z[s + i] = alt[i - 1]
                    seqs.append(z)
                batch = torch.stack(seqs)                                   # [B-1, s+1+B]
                lg = target(batch.to(dev), logits_to_keep=B + 1).logits[:, :-1].float()   # predictions for y_1..y_B
                pred = lg.argmax(-1)                                       # [B-1, B]
                edges = defaultdict(list)
                for i in range(1, B):
                    changed = 0
                    for k in range(i + 1, B + 1):
                        c = bool(pred[i - 1, k - 1] != y[k - 1])
                        d = k - i
                        by_d[d][0] += c
                        by_d[d][1] += 1
                        if c:
                            changed += 1
                            edges[k].append(i)
                            n_changes += 1
                            tie_changes += margin[k - 1] < args.tie
                    outdeg.append(changed)
                    retained.append(1 - changed / (B - i))
                    outbucket["0" if changed == 0 else "1" if changed == 1 else "2-3" if changed <= 3 else
                              "4-7" if changed <= 7 else "8+"] += 1
                indeg.append(sum(len(v) for v in edges.values()) / (B - 1))
                chains.append(longest_chain(edges, B))
        print(f"[{dname}] {n_blocks} blocks", flush=True)

    rows = [{"distance d": d, "P(changes)": c / t} for d, (c, t) in sorted(by_d.items())]
    print_table(rows, list(rows[0]), f"1. a changed token flips the target's choice d positions later ({n_blocks} blocks)")
    n = len(outdeg)
    rows2 = [{"later positions changed": b, "share of perturbations": outbucket[b] / n}
             for b in ("0", "1", "2-3", "4-7", "8+")]
    print_table(rows2, list(rows2[0]), "2a. out-degree of one changed token")
    summ = {"mean later positions changed": sum(outdeg) / n,
            "mean share of the suffix still valid": sum(retained) / n,
            "mean in-block parents per position": sum(indeg) / len(indeg),
            "mean longest dependency chain (of B)": sum(chains) / len(chains),
            "B": B,
            "share of flips at target near-ties": tie_changes / max(n_changes, 1)}
    print_table([summ], list(summ), "2b/3. summary")
    save_json(args.out, {"args": vars(args), "by_distance": rows, "outdegree": rows2, "summary": summ})


if __name__ == "__main__":
    main()
