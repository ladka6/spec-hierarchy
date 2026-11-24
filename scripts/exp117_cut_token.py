"""Experiment 117: the token at the cut. Where a chain continues is easy to find; WHAT goes there is the hard part.

A chain of blocks continues after the drafter's first error e only if the redraft produces the target's token there
(in plain DFlash that token is the bonus of the verify pass). Known (exp112): with the real features of the kept
tokens the redraft's first token is right in 0.54 of the cases, without features 0.20. Three questions, no training,
frozen original drafter, the cut placed exactly at the first error (oracle location):

1. Does the redraft just REPEAT the token that was wrong the first time? At the cut we know the drafted token is
   wrong. "masked" = the redraft's first position with that token excluded.
2. Candidates: is the target's token among the first 2 / 3 / 5 choices (of the redraft, with or without the wrong
   token; of the first draft)? This is what branching at the cut would get (width in the verify pass is free).
3. Two looks: the first draft and the redraft are two predictions of the same position from different context.
   "two looks" = sum of their log-probabilities, wrong token excluded.
Split by how sure the TARGET is at that position (its top-1 probability): where the target itself is unsure, no
cheap method can know its choice; where it is sure, the error is the drafter's and should be fixable.

Feature conditions for the redraft (the cut's anchor is the last kept token, s + e):
  real   real target features for the kept tokens (what an exact feature source would give)
  none   no features for the kept tokens, their tokens only (the lag input)
Columns, share of the anchors with an error (also for e > 0 only: at e = 0 nothing is kept, the redraft is the same
call again and only the masked / candidate rows can differ):
  fix          redraft's first token = target's token
  repeat       redraft's first token = the same wrong token;   repeat|fail = among the failures
  masked       first token right when the wrong token is excluded
  top-k        target's token among the redraft's first k;   m top-k = with the wrong token excluded
  two looks    first draft + redraft, wrong token excluded
  draft 2nd / draft top-3 / top-5   from the FIRST draft alone (no redraft at all), wrong token excluded for 2nd
Chain table (accepted drafted tokens, both blocks): one block; plain redraft; masked; two looks; "any of 3" = the
target's token is among the masked top-3 (what a 3-way branch at the cut gets, with the same redraft tail);
ideal = an exact verify at the cut (the bonus token given, redraft with real features).

  python scripts/exp117_cut_token.py --data /scratch-shared/$USER/hspec_lagft/data
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp107_lift_diag import load_seqs  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

KEYS = ("fix", "repeat", "masked", "top-2", "top-3", "top-5", "m top-2", "m top-3", "two looks",
        "draft 2nd", "draft top-3", "draft top-5")
BINS = (("sure >= 0.9", 0.9, 1.01), ("0.5 - 0.9", 0.5, 0.9), ("unsure < 0.5", 0.0, 0.5))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--anchors", type=int, default=2000)
    ap.add_argument("--fit-tokens", type=int, default=600_000, help="as exp112: fixes the held-out split")
    ap.add_argument("--val-tokens", type=int, default=15_000)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--save", default="exp117_cut_token")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    dev, t0 = "cuda", time.time()

    seqs = load_seqs(args)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    head = target.lm_head
    lids, bs = list(draft.target_layer_ids), draft.block_size

    # ---- the held-out sequences of exp112 / exp113 (the first 600 anchors are the same) ----
    def take(i, n_tokens, cap):
        n = 0
        while n < n_tokens and i < cap:
            n += len(seqs[i][0]) - seqs[i][1]
            i += 1
        return i

    max_fit = int(0.6 * len(seqs))
    fit_end = take(take(0, args.fit_tokens, max_fit), args.val_tokens, max_fit + (len(seqs) - max_fit) // 4)
    rest = seqs[fit_end:]
    cand_a = [(si, s) for si, (ids, lo) in enumerate(rest) for s in range(max(lo, 18), len(ids) - bs)]
    random.Random(1).shuffle(cand_a)
    by_seq = {}
    for si, s in sorted(cand_a[: args.anchors]):
        by_seq.setdefault(si, []).append(s)
    print(f"[setup] {len(rest)} held-out sequences, {sum(len(v) for v in by_seq.values())} anchors, block {bs}, "
          f"{time.time() - t0:.0f}s", flush=True)

    def blk(feats, x, s, lag=0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return blocks_logits(draft, target, feats, x, [(s, lag)], bs)[0].float()

    def rank_in(lp, tok, k):
        return tok in lp.topk(k).indices.tolist()

    conds = ("real", "none")
    cnt = {(c, sub): {k: 0 for k in KEYS} for c in conds for sub in ("all errors", "e > 0")}
    n_sub = {"all errors": 0, "e > 0": 0}
    binc = {b[0]: {k: 0 for k in ("n", "fix", "masked", "m top-3", "two looks", "draft 2nd")} for b in BINS}
    ch = {k: [] for k in ("one block", "plain real", "masked real", "two looks real", "any of 3 real", "ideal",
                          "plain none", "masked none", "two looks none", "any of 3 none")}
    n_clean = n_used = 0
    for n_done, (si, anchors) in enumerate(by_seq.items()):
        ids, _lo = rest[si]
        x = ids.to(dev)
        hs = target(x[None], output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        feats = extract_context_feature(hs, lids)
        truth_all = x.tolist()
        for s in anchors:
            if s + 2 * bs >= len(ids):
                continue
            n_used += 1
            truth = truth_all[s + 1 :]
            L1 = blk(feats[:, :s], x, s)
            chain = L1.argmax(-1).tolist()
            e = matched(chain, truth)
            if e >= len(chain):                              # no error in the block: nothing to resolve
                n_clean += 1
                for k in ch:
                    ch[k].append(e)
                continue
            t, w = truth[e], chain[e]
            a2 = s + e                                       # anchor of the redraft = the last kept token
            tr = truth_all[a2 + 1 :]                         # tr[0] = t
            lp1 = F.log_softmax(L1[e], -1)
            lp1m = lp1.clone()
            lp1m[w] = float("-inf")
            p_t = torch.softmax(head(hs[-1][0, s + e]).float(), -1).max().item()   # the target's own confidence
            subs = ["all errors"] + (["e > 0"] if e > 0 else [])
            for sub in subs:
                n_sub[sub] += 1
            ch["one block"].append(e)
            ideal = e + 1 + matched(blk(feats[:, : a2 + 1], x, a2 + 1).argmax(-1).tolist(), truth_all[a2 + 2 :])
            ch["ideal"].append(ideal)
            for c in conds:
                R = blk(feats[:, :a2], x, a2) if c == "real" else blk(feats[:, :s], x, a2, e)
                first = R.argmax(-1).tolist()
                tail = matched(first[1:], tr[1:])            # what follows the cut token in the same redraft
                lp2 = F.log_softmax(R[0], -1)
                lp2m = lp2.clone()
                lp2m[w] = float("-inf")
                two = lp1m + lp2m
                r = {"fix": first[0] == t, "repeat": first[0] == w, "masked": lp2m.argmax().item() == t,
                     "top-2": rank_in(lp2, t, 2), "top-3": rank_in(lp2, t, 3), "top-5": rank_in(lp2, t, 5),
                     "m top-2": rank_in(lp2m, t, 2), "m top-3": rank_in(lp2m, t, 3),
                     "two looks": two.argmax().item() == t, "draft 2nd": lp1m.argmax().item() == t,
                     "draft top-3": rank_in(lp1, t, 3), "draft top-5": rank_in(lp1, t, 5)}
                for sub in subs:
                    for k in KEYS:
                        cnt[(c, sub)][k] += r[k]
                ch[f"plain {c}"].append(e + matched(first, tr))
                ch[f"masked {c}"].append(e + 1 + tail if r["masked"] else e)
                ch[f"two looks {c}"].append(e + 1 + tail if r["two looks"] else e)
                ch[f"any of 3 {c}"].append(e + 1 + tail if r["m top-3"] else e)
                if c == "real" and e > 0:
                    for name, lo_p, hi_p in BINS:
                        if lo_p <= p_t < hi_p:
                            b = binc[name]
                            b["n"] += 1
                            for k in ("fix", "masked", "m top-3", "two looks", "draft 2nd"):
                                b[k] += r[k]
        del hs, feats
        if n_done % 200 == 0:
            print(f"[run] {n_done}/{len(by_seq)} sequences, {n_used} anchors, {time.time() - t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    rows = []
    for sub in ("all errors", "e > 0"):
        for c in conds:
            n = max(n_sub[sub], 1)
            r = {"anchors": sub, "features": c, "n": n_sub[sub]}
            r.update({k: cnt[(c, sub)][k] / n for k in KEYS})
            r["repeat|fail"] = cnt[(c, sub)]["repeat"] / max(n - cnt[(c, sub)]["fix"], 1)
            rows.append(r)
    cols = ["anchors", "features", "n", "fix", "repeat", "repeat|fail", "masked", "two looks", "top-2", "top-3", "top-5",
            "m top-2", "m top-3", "draft 2nd", "draft top-3", "draft top-5"]
    print(f"\n{n_used} anchors, {n_clean} without an error in the block, {n_sub['all errors']} with one "
          f"({n_sub['all errors'] - n_sub['e > 0']} of them at the first drafted token)")
    print_table(rows, cols, "The token at the cut (cut exactly at the drafter's first error)")
    brow = []
    for name, _, _ in BINS:
        b = binc[name]
        n = max(b["n"], 1)
        brow.append({"target's top-1 prob": name, "n": b["n"], "share": b["n"] / max(n_sub["e > 0"], 1),
                     **{k: b[k] / n for k in ("fix", "masked", "two looks", "m top-3", "draft 2nd")}})
    print_table(brow, list(brow[0]), "By how sure the target is at the cut (real features, e > 0)")
    crow = [{k: mean(v) for k, v in ch.items()}]
    print_table(crow, list(crow[0]), "Chain of two blocks, accepted drafted tokens (all anchors)")
    save_json(args.save, {"args": vars(args), "rows": rows, "bins": brow, "chain": crow[0],
                          "n": {"used": n_used, "clean": n_clean, **n_sub}})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
