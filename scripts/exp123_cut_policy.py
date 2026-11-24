"""Experiment 123: the cut as a POLICY. Not "find the exact error and fix it" but "where to stop trusting the draft,
and what to do with the token there", scored by the accepted length of the whole chain.

exp117 placed the cut exactly at the drafter's first error (oracle location) and found that excluding the token
known to be wrong raises the redraft's fix rate from 0.54 to 0.69, and that drafting the continuation from the
chosen token as anchor gives more than the tail of the redraft. Neither is usable as is: a real rule does not know
the error position, and excluding a token that was right loses the chain. Here every cut position c = 0 .. 15 is
evaluated on every anchor, so any rule can be scored, including blocks without an error (c = 15: plain extension).

Chain for a cut c: keep the first c drafted tokens; redraft from the last kept token; token at position c:
  plain   the redraft's first choice
  soft    the redraft's log-probabilities, with the token the first draft had there scaled by the first draft's own
          confidence in it (log p2[w] + log conf_c): a doubted token is suppressed, a trusted one is not
  mask    that token excluded (right only if the cut is at the error)
then the continuation:
  tail      the rest of the redraft block                               (2 drafter calls per chain)
  anchored  a new block with the chosen token as anchor                 (3 drafter calls)
Accepted drafted tokens A = leading tokens of [kept, token, continuation] equal to the target's. A late cut keeps a
wrong token (A = e, the one-block value); an early cut with a wrong token at c LOSES tokens (A = c < e).

Rules: fixed c; conf < thr (first position whose max-probability is below thr, else 15); oracle (c = e, the first
error; 15 on a clean block); best = the best c in hindsight per chain variant (upper bound for any cut rule).
Feature conditions for the kept tokens: real (target features), none (tokens only).

Tables: A per rule and variant; cut statistics of the conf rules; estimated speed against one block per verify,
((A + 1) / (F + calls * D + E)) / ((A1 + 1) / (F + D)), for the timings --D / --F and feature-pass costs --E.

  python scripts/exp123_cut_policy.py --data /scratch-shared/$USER/hspec_lagft/data
"""

from __future__ import annotations

import argparse
import math
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

SUPS = ("plain", "soft", "mask")
CONTS = ("tail", "anchored")
CONDS = ("real", "none")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--anchors", type=int, default=1000)
    ap.add_argument("--fixed", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--thrs", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--D", type=float, default=3.3, help="ms per drafter call (cost model)")
    ap.add_argument("--F", type=float, default=19.0, help="ms per full verify (cost model)")
    ap.add_argument("--E", nargs="+", type=float, default=[0.0, 2.6, 5.3], help="ms of the feature pass per chain")
    ap.add_argument("--fit-tokens", type=int, default=600_000, help="as exp112: fixes the held-out split")
    ap.add_argument("--val-tokens", type=int, default=15_000)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--save", default="exp123_cut_policy")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    dev, t0 = "cuda", time.time()

    seqs = load_seqs(args)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    nb = bs - 1                                              # drafted tokens per block

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

    rules = [f"fixed {k}" for k in args.fixed] + [f"conf<{t:g}" for t in args.thrs] + ["oracle", "best"]
    variants = [(c, sp, ct) for c in CONDS for sp in SUPS for ct in CONTS]
    acc = {(r, v): [] for r in rules for v in variants}
    cut = {f"conf<{t:g}": {"late": 0, "exact": 0, "early": 0, "lost": 0, "none": 0} for t in args.thrs}
    one, n_used, n_clean = [], 0, 0
    for n_done, (si, anchors) in enumerate(by_seq.items()):
        ids, _lo = rest[si]
        x = ids.to(dev)
        hs = target(x[None], output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        feats = extract_context_feature(hs, lids)
        truth_all = x.tolist()
        for s in anchors:
            if s + 3 * bs >= len(ids):
                continue
            n_used += 1
            truth = truth_all[s + 1 :]
            L1 = blk(feats[:, :s], x, s)
            conf = torch.softmax(L1, -1).max(-1).values.tolist()
            chain = L1.argmax(-1).tolist()
            e = matched(chain, truth)
            one.append(e)
            n_clean += e >= nb
            A = {}                                           # (cond, c, sup, cont) -> accepted drafted tokens
            for cond in CONDS:
                for c in range(nb + 1):
                    a2 = s + c                               # the last kept token = anchor of the redraft
                    xk = x.clone()
                    if c:
                        xk[s + 1 : s + 1 + c] = torch.tensor(chain[:c], device=dev)
                    R = blk(feats[:, :a2], xk, a2) if cond == "real" else blk(feats[:, :s], xk, a2, c)
                    rtok = R.argmax(-1).tolist()
                    lp = F.log_softmax(R[0], -1)
                    picks = {"plain": rtok[0]}
                    if c < nb:
                        w = chain[c]
                        soft = lp.clone()
                        soft[w] += math.log(max(conf[c], 1e-6))
                        picks["soft"] = soft.argmax().item()
                        hard = lp.clone()
                        hard[w] = float("-inf")
                        picks["mask"] = hard.argmax().item()
                    else:                                    # nothing was drafted at position c: plain extension
                        picks["soft"] = picks["mask"] = rtok[0]
                    anchored = {}
                    for sp, v in picks.items():
                        A[(cond, c, sp, "tail")] = matched(chain[:c] + [v] + rtok[1:], truth)
                        if v not in anchored:
                            xv = xk.clone()
                            xv[a2 + 1] = v
                            C = (blk(feats[:, : a2 + 1], xv, a2 + 1) if cond == "real"
                                 else blk(feats[:, :s], xv, a2 + 1, c + 1))
                            anchored[v] = C.argmax(-1).tolist()
                        A[(cond, c, sp, "anchored")] = matched(chain[:c] + [v] + anchored[v], truth)
            cs = {f"fixed {k}": min(k, nb) for k in args.fixed}
            for t in args.thrs:
                low = [i for i, p in enumerate(conf) if p < t]
                cs[f"conf<{t:g}"] = low[0] if low else nb
                st = cut[f"conf<{t:g}"]
                c = cs[f"conf<{t:g}"]
                st["none"] += not low
                st["late"] += c > e
                st["exact"] += c == e
                st["early"] += c < e
                st["lost"] += max(e - c, 0)
            cs["oracle"] = e
            for v in variants:
                cond, sp, ct = v
                for r, c in cs.items():
                    acc[(r, v)].append(A[(cond, c, sp, ct)])
                acc[("best", v)].append(max(A[(cond, c, sp, ct)] for c in range(nb + 1)))
        del hs, feats
        if n_done % 100 == 0:
            print(f"[run] {n_done}/{len(by_seq)} sequences, {n_used} anchors, {time.time() - t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    a1 = mean(one)
    print(f"\n{n_used} anchors, one block: {a1:.3f} accepted drafted tokens, {n_clean} blocks without an error")
    out = {"args": vars(args), "one_block": a1, "n": n_used, "tables": {}}
    for cond in CONDS:
        rows = []
        for r in rules:
            row = {"rule": r}
            for sp in SUPS:
                for ct in CONTS:
                    row[f"{sp} / {ct}"] = mean(acc[(r, (cond, sp, ct))])
            rows.append(row)
        print_table(rows, list(rows[0]), f"Accepted drafted tokens of the chain, features for the kept tokens: {cond} "
                                         f"(one block {a1:.3f})")
        out["tables"][cond] = rows
    crow = [{"rule": r, "no cut (extend)": st["none"] / n_used, "late": st["late"] / n_used,
             "exact": st["exact"] / n_used, "early": st["early"] / n_used,
             "tokens redrafted per early cut": st["lost"] / max(st["early"], 1)} for r, st in cut.items()]
    print_table(crow, list(crow[0]), "Where the confidence rules cut, relative to the first error")
    srows = []
    for cond in CONDS:
        for r in rules:
            for sp in SUPS:
                for ct in CONTS:
                    a = mean(acc[(r, (cond, sp, ct))])
                    calls = 2 if ct == "tail" else 3
                    row = {"features": cond, "rule": r, "token": sp, "continuation": ct, "A": a}
                    for e_ms in args.E:
                        row[f"speed E={e_ms:g}"] = ((a + 1) / (args.F + calls * args.D + e_ms)) / ((a1 + 1) / (args.F + args.D))
                    srows.append(row)
    for cond in CONDS:
        top = sorted([r for r in srows if r["features"] == cond and r["rule"] != "best"],
                     key=lambda r: -r[f"speed E={args.E[0]:g}"])[:10]
        print_table(top, list(top[0]), f"Estimated speed against one block per verify, ten best rule / variant pairs, "
                                       f"{cond} features (D = {args.D} ms, F = {args.F} ms; E = feature pass, ms)")
    out["speed"], out["cuts"] = srows, crow
    save_json(args.save, out)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
