"""Experiment 19 (idea B): trees from a dependency head vs DDTree, offline.

For each anchor: one drafter pass (backbone rows + per-position marginals). DDTree builds its
tree from the marginals; the dependency head (hspec/dephead.py) rescores the top-M candidates
per depth given the parent token (M*(D-1)+1 rows in one batch) and a best-first tree is built
from those first-order conditionals. Heads compared: none (the same candidate sets scored by
the backbone alone: a sanity check, ~DDTree), teacher-forced (tf) and on-policy (tree).

Offline on target-greedy trajectories, anchors every `stride` positions; accepted tokens per
target pass (incl. bonus) at equal node budget.

  python scripts/exp19_dephead.py --heads tf=.../dephead_tf.pt tree=.../dephead_tree.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import _output_head, extract_context_feature  # noqa: E402
from exp15_blocktree import accepted, tree_paths  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.dephead import backbone_rows, dep_tree, load_dephead  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.tree import build_ddtree  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--heads", nargs="+", required=True, help="name=path")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--budgets", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--ms", nargs="+", type=int, default=[4, 8])
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    dtype = next(target.parameters()).dtype
    dev = next(target.parameters()).device
    heads = {"none": None, **{k: load_dephead(v, dev, dtype) for k, v in (h.split("=", 1) for h in args.heads)}}
    bs = draft.block_size
    D = bs - 1

    acc = defaultdict(list)
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            anchors = list(range(n0, len(x) - bs - 1, args.stride))
            for c in range(0, len(anchors), 16):
                chunk = anchors[c : c + 16]
                H = backbone_rows(draft, target, feats, x, chunk, bs)
                for a, s in enumerate(chunk):
                    truth = x[s + 1 : s + 1 + D].tolist()
                    lg = draft.compute_logits(H[a], _output_head(target)).float()
                    lq = torch.log_softmax(lg, -1)
                    for B in args.budgets:
                        acc[("ddtree", B, d)].append(accepted(set(tree_paths(build_ddtree(lg, B), lq.cpu())), truth))
                        for hn, hd in heads.items():
                            for M in args.ms:
                                t = dep_tree(draft, target, hd, H[a], lq, int(x[s]), B, M)
                                acc[(f"dep-{hn} M{M}", B, d)].append(accepted(set(t), truth))
        print(f"[{d}] done", flush=True)

    rows = []
    names = sorted({k[0] for k in acc}, key=lambda n: (n != "ddtree", n))
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
    save_json("exp19_dephead", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
