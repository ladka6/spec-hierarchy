"""Experiment 97: are the errors of different drafts correlated? (offline, T=0)

exp96: given a correct prefix, DFlash is right ~90% at every position, so acceptance is compounding of a constant
per-token hazard. If K drafts verified together (free width at batch 1) failed at INDEPENDENT positions, the hazard
of the best draft would drop from h to ~h^K and acceptance would jump. What matters is how correlated the drafts'
failures are.

Per block (target greedy text t = truth), a PRIMARY draft (plain DFlash argmax) and alternative drafts from:
  beam      the k-best full paths through DFlash's per-position marginals (beam search, product of probabilities)
  sample    K independent samples from the marginals (temperature --temp)
  model     other drafters on the same input (fine-tuned DFlash checkpoints: different weights, same conditioning)
For each source and K: mean accepted of the primary, of the best of the K (= what verifying all K as a tree gives),
the tree size, and the RESCUE rate: when the primary fails at position e (prefix right), how often does an alternative
draft share the correct prefix AND get position e right? Under independence the rescue rate would be about the
alternative's own accuracy at a correct prefix (~0.9); a low rescue rate means correlated failures.

  python scripts/exp97_decorrelation.py --models $HOME/hspec_ckpt/drafter_infill $HOME/hspec_ckpt/drafter_depth10
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
from exp27_refresh import blocks_logits, matched, tree_nodes  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def kbest_paths(logp, K, width=8):
    """K highest-probability full paths through independent marginals logp [D, V] (beam search)."""
    D = logp.shape[0]
    v, i = logp.topk(width, -1)
    v, i = v.tolist(), i.tolist()
    beams = [(0.0, [])]
    for d in range(D):
        cand = [(s + v[d][c], p + [i[d][c]]) for s, p in beams for c in range(width)]
        cand.sort(key=lambda z: -z[0])
        beams = cand[:K]
    return [p for _, p in beams]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--models", nargs="*", default=[], help="other drafter checkpoints (same architecture)")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="exp97_decorrelation")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    others = {}
    for m in args.models:
        if Path(m).expanduser().exists():
            others[Path(m).name] = load_draft(str(Path(m).expanduser()))
        else:
            print(f"[models] skipping missing {m}", flush=True)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)
    gen = torch.Generator(device="cuda").manual_seed(0)

    rec = defaultdict(list)          # source -> [(primary acc, best-of-K acc, nodes)]
    resc = defaultdict(lambda: [0, 0])          # source -> [rescued, failures of the primary]
    alive = defaultdict(lambda: [0, 0])         # source -> [alive alternatives right at e, alive alternatives]
    n_blocks = 0
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            xl = x.tolist()
            for s in range(ids.shape[1], len(x) - bs - 1, args.stride):
                n_blocks += 1
                t = xl[s + 1 : s + 1 + D]
                logp = torch.log_softmax(blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].float(), -1)
                y = logp.argmax(-1).tolist()
                a0 = matched(y, t)
                alts = {}
                for K in args.ks:
                    alts[f"beam K={K}"] = kbest_paths(logp, K)[1:]
                    probs = torch.softmax(logp / args.temp, -1)
                    alts[f"sample K={K}"] = [torch.multinomial(probs, 1, generator=gen)[:, 0].tolist()
                                             for _ in range(K - 1)]
                if others:
                    oth = [blocks_logits(m, target, feats, x, [(s, 0)], bs)[0].argmax(-1).tolist()
                           for m in others.values()]
                    for name, o in zip(others, oth):
                        alts[f"model {name}"] = [o]
                    if len(oth) > 1:
                        alts["models all"] = oth
                for src, al in alts.items():
                    chains = [y] + al
                    best = max(matched(c, t) for c in chains)
                    rec[src].append((a0 + 1, best + 1, tree_nodes(chains)))
                    if a0 < D:
                        resc[src][1] += 1
                        resc[src][0] += any(matched(c, t) > a0 for c in al)
                        for c in al:                                   # alternatives still on the true path at e
                            if matched(c[:a0], t[:a0]) == a0:
                                alive[src][1] += 1
                                alive[src][0] += c[a0] == t[a0]
        print(f"[{dname}] {n_blocks} blocks", flush=True)

    rows = []
    for src, v in rec.items():
        rows.append({"alternative drafts": src, "primary accepted": sum(a for a, _, _ in v) / len(v),
                     "best of all accepted": sum(b for _, b, _ in v) / len(v),
                     "tree nodes": sum(n for _, _, n in v) / len(v),
                     "rescue rate at primary's failure": resc[src][0] / max(resc[src][1], 1),
                     "alive alternatives right at e": alive[src][0] / max(alive[src][1], 1),
                     "alive per failure": alive[src][1] / max(resc[src][1], 1)})
    print_table(rows, list(rows[0]), f"1. do alternative drafts survive where the primary fails? ({n_blocks} blocks)")
    print("[reference] 'alive alternatives right at e': among alternative drafts still on the true path at the primary's "
          "first error, the share that is right there. Independent failures -> about their accuracy at a correct "
          "prefix (~0.9, exp96); much lower -> they fail where the primary fails (correlated)", flush=True)
    save_json(args.out, {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
