"""Experiment 124: how deep can the chain go. exp123 scored two blocks; this runs B = 1 .. --depth stages.

A stage: one drafter call from the last kept token (15 drafted tokens), a cut c in that block, keep the first c
tokens, the next stage starts from the token before the cut. The token the previous stage had at the cut is handled
at the first position of the next block (plain / soft / mask, as exp123). The chain after B stages is everything kept
plus the whole last block; A(B) = its leading tokens equal to the target's. One verify at the end, B drafter calls.

Variants (cut rule / token at the cut):
  oracle / plain        cut at the first error of the block; the redraft's first choice is committed
  oracle / mask         the token known to be wrong excluded; the choice is committed (wrong -> the chain is dead)
  oracle / mask retry   as above, but a wrong choice is cut again at the same position, with every token already
                        tried there excluded (a stage per retry: what a perfect detector could do)
  conf<t / plain, soft  cut at the first position whose max-probability is below t; from the second stage on the
                        first token of a block is committed (it was just chosen), so every stage advances
Feature conditions for the hypothesised (in-flight) tokens: real (target features), none (features only up to the
anchor of stage 1; the lag grows with every stage and is reported), and L<k> for k in --shallow: the REAL state of
target layer k (the first k layers run over the in-flight tokens, the early-exit shape) through a projection fitted
in closed form to the drafter's fused feature (exp118: with the interface refitted, shallow states are usable).
These sit between the two ends at a known price, k layers per stage, which the speed table charges.

Tables per condition: A(B); tokens added by stage B and the ratio to the stage before (the f of the chain formula);
share of chains still correct and the mean lag at the start of each stage; speed against one block per verify,
((A(B) + 1) / (F(n) + B D + (B - 1) E)) / ((A(1) + 1) / (F(16) + D)), F(n) = --F + --F-slope (n - 16) for the n
tokens verified, with the measured timings of bench_band as defaults.

  python scripts/exp124_chain_depth.py --data /scratch-shared/$USER/hspec_lagft/data
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp107_lift_diag import load_seqs  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

CONDS = ("real", "none")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--anchors", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--thrs", nargs="+", type=float, default=[0.5])
    ap.add_argument("--D", type=float, default=3.55, help="ms per drafter call")
    ap.add_argument("--F", type=float, default=20.4, help="ms of a full verify of 16 tokens")
    ap.add_argument("--F-slope", type=float, default=0.04, help="ms per further verified token")
    ap.add_argument("--E", nargs="+", type=float, default=[0.0, 1.46, 3.61], help="ms of a feature / cut pass per stage")
    ap.add_argument("--shallow", nargs="*", type=int, default=[2, 4, 8, 16, 33], help="layers for the L<k> conditions")
    ap.add_argument("--layer-ms", type=float, default=0.54, help="ms per target layer over the in-flight tokens")
    ap.add_argument("--fit-seqs", type=int, default=150)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--fit-tokens", type=int, default=600_000, help="as exp112: fixes the held-out split")
    ap.add_argument("--val-tokens", type=int, default=15_000)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--save", default="exp124_chain_depth")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    dev, t0 = "cuda", time.time()

    seqs = load_seqs(args)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    nb, B = bs - 1, args.depth
    H = target.config.hidden_size
    fc0 = draft.fc                                           # the drafter's fusion: taps -> drafter width
    draft.fc = nn.Identity()                                 # from here on the drafter is fed FUSED features
    conds = list(CONDS) + [f"L{k}" for k in args.shallow]
    need = B * nb + bs + 2                                   # room for B full blocks after the anchor

    def take(i, n_tokens, cap):
        n = 0
        while n < n_tokens and i < cap:
            n += len(seqs[i][0]) - seqs[i][1]
            i += 1
        return i

    max_fit = int(0.6 * len(seqs))
    fit_end = take(take(0, args.fit_tokens, max_fit), args.val_tokens, max_fit + (len(seqs) - max_fit) // 4)
    rest = seqs[fit_end:]
    cand_a = [(si, s) for si, (ids, lo) in enumerate(rest) for s in range(max(lo, 18), len(ids) - need)]
    random.Random(1).shuffle(cand_a)
    by_seq = {}
    for si, s in sorted(cand_a[: args.anchors]):
        by_seq.setdefault(si, []).append(s)
    n_anchor = sum(len(v) for v in by_seq.values())
    print(f"[setup] {len(rest)} held-out sequences, {n_anchor} anchors with {need} tokens after them, block {bs}, "
          f"depth {B}, {time.time() - t0:.0f}s", flush=True)

    def fused(hs):
        return fc0(extract_context_feature(hs, lids))

    # closed-form projections: state of layer k -> the fused feature (as exp118)
    proj = {}
    if args.shallow:
        Hd = fc0.out_features
        xtx = {k: torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev) for k in args.shallow}
        xty = {k: torch.zeros(H + 1, Hd, dtype=torch.float64, device=dev) for k in args.shallow}
        for ids, lo in seqs[: args.fit_seqs]:
            hs = target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
            y = fused(hs)[0, lo:].double()
            for k in args.shallow:
                h = hs[k][0, lo:].double()
                w = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
                ha = torch.cat([h, torch.ones_like(h[:, :1])], -1) * w
                xtx[k] += ha.T @ ha
                xty[k] += ha.T @ (y * w)
            del hs
        for k in args.shallow:
            A = xtx[k] + args.ridge * xtx[k].diagonal().mean() * torch.eye(H + 1, dtype=torch.float64, device=dev)
            sol = torch.linalg.solve(A, xty[k]).float()
            proj[k] = nn.Linear(H, Hd, bias=True).to(dev)
            proj[k].weight.copy_(sol[:H].T)
            proj[k].bias.copy_(sol[H])
        del xtx, xty
        print(f"[fit] projections for layers {args.shallow} on {args.fit_seqs} sequences, {time.time() - t0:.0f}s",
              flush=True)

    def blk(feats, x, s, lag=0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return blocks_logits(draft, target, feats, x, [(s, lag)], bs)[0].float()

    variants = [("oracle", "plain"), ("oracle", "mask"), ("oracle", "mask retry")]
    for t in args.thrs:
        variants += [(f"conf<{t:g}", "plain"), (f"conf<{t:g}", "soft")]
    keys = [(c, v) for c in conds for v in variants]
    acc = {k: [[] for _ in range(B)] for k in keys}          # A after stage b
    length = {k: [[] for _ in range(B)] for k in keys}       # tokens handed to the verify after stage b
    alive = {k: [0] * B for k in keys}                       # chains correct up to the anchor of stage b
    lags = {k: [[] for _ in range(B)] for k in keys}         # in-flight tokens before the anchor of stage b

    def ctx(cond, hs, feats, s, a):
        """fused features of positions < a for a stage anchored at a (stage 1 was anchored at s)"""
        if cond == "real":
            return feats[:, :a]
        k = int(cond[1:])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            new = proj[k](hs[k][:, s:a])
        return torch.cat([feats[:, :s], new.to(feats.dtype)], 1)

    def run(cond, rule, sup, x, hs, feats, s, truth, first):
        """one chain; first = (logits of stage 1); returns per stage (A, verified length, alive, lag)"""
        thr = float(rule.split("<")[1]) if rule.startswith("conf") else None
        kept, a, out = [], s, []                             # kept drafted tokens, anchor of the current stage
        banned, pending, dead_at = {}, None, None            # position -> tokens tried; (token, conf) of the cut
        xk = x.clone()
        for b in range(B):
            if dead_at is not None:                          # the kept part is wrong: nothing later can add tokens
                out.append((dead_at, len(kept) + nb, 0, a - s))
                continue
            if b == 0:
                R = first
            else:
                R = blk(feats[:, :s], xk, a, a - s) if cond == "none" else blk(ctx(cond, hs, feats, s, a), xk, a)
            p = torch.softmax(R, -1)
            tok, conf = R.argmax(-1).tolist(), p.max(-1).values.tolist()
            if pending is not None:                          # the token at the cut of the stage before
                lp = F.log_softmax(R[0], -1)
                if sup == "soft":
                    lp[pending[0]] += math.log(max(pending[1], 1e-6))
                elif sup.startswith("mask"):
                    lp[banned[a + 1]] = float("-inf")
                tok[0] = lp.argmax().item()
                conf[0] = p[0, tok[0]].item()
            tr = truth[len(kept):]
            e = matched(tok, tr)
            out.append((len(kept) + e, len(kept) + nb, 1, a - s))
            if thr is None:
                c = e
            else:
                low = [i for i in range(1 if b else 0, nb) if conf[i] < thr]
                c = low[0] if low else nb
            if sup != "mask retry" and b and c == 0:         # the committed token is wrong: the chain ends here
                dead_at = len(kept)
                kept = kept + tok[:1]
                continue
            if c > e:                                        # a late cut keeps a wrong token
                dead_at = len(kept) + e
            if c < nb:
                banned.setdefault(a + c + 1, []).append(tok[c])
                pending = (tok[c], conf[c])
            else:
                pending = None
            if c:
                xk[a + 1 : a + 1 + c] = torch.tensor(tok[:c], device=dev)
            kept = kept + tok[:c]
            a += c
        return out

    n_used = 0
    for n_done, (si, anchors) in enumerate(by_seq.items()):
        ids, _lo = rest[si]
        x = ids.to(dev)
        hs = target(x[None], output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        feats = fused(hs)
        truth_all = x.tolist()
        for s in anchors:
            n_used += 1
            truth = truth_all[s + 1 :]
            first = blk(feats[:, :s], x, s)
            for cond, (rule, sup) in keys:
                for b, (a_b, n_b, al, lag) in enumerate(run(cond, rule, sup, x, hs, feats, s, truth, first)):
                    k = (cond, (rule, sup))
                    acc[k][b].append(a_b)
                    length[k][b].append(n_b)
                    alive[k][b] += al
                    if al:
                        lags[k][b].append(lag)
        del hs, feats
        if n_done % 100 == 0:
            print(f"[run] {n_done}/{len(by_seq)} sequences, {n_used} anchors, {time.time() - t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    out = {"args": vars(args), "n": n_used, "tables": {}}
    cols = [f"B={b + 1}" for b in range(B)]
    print(f"\n{n_used} anchors")
    for cond in conds:
        own = [args.layer_ms * int(cond[1:])] if cond.startswith("L") else args.E
        rows_a, rows_d, rows_f, rows_al, rows_lag, rows_s = [], [], [], [], [], []
        for v in variants:
            k, name = (cond, v), f"{v[0]} / {v[1]}"
            A = [mean(acc[k][b]) for b in range(B)]
            add = [A[0]] + [A[b] - A[b - 1] for b in range(1, B)]
            rows_a.append({"variant": name, **dict(zip(cols, A))})
            rows_d.append({"variant": name, **dict(zip(cols, add))})
            rows_f.append({"variant": name, **{cols[b]: add[b] / add[b - 1] if add[b - 1] > 1e-6 else float("nan")
                                               for b in range(1, B)}})
            rows_al.append({"variant": name, **{cols[b]: alive[k][b] / max(n_used, 1) for b in range(B)}})
            rows_lag.append({"variant": name, **{cols[b]: mean(lags[k][b]) for b in range(B)}})
            base = (A[0] + 1) / (args.F + args.D)
            for e_ms in own:
                row = {"variant": name, "E ms": e_ms}
                for b in range(B):
                    t = args.F + args.F_slope * max(mean(length[k][b]) - 16, 0) + (b + 1) * args.D + b * e_ms
                    row[cols[b]] = ((A[b] + 1) / t) / base
                row["best B"] = max(range(B), key=lambda b: row[cols[b]]) + 1
                rows_s.append(row)
        print_table(rows_a, ["variant"] + cols, f"Accepted drafted tokens after B stages, in-flight features: {cond}")
        print_table(rows_d, ["variant"] + cols, f"Tokens added by stage B, {cond}")
        print_table(rows_f, ["variant"] + cols[1:], f"Ratio of the tokens added by stage B to stage B - 1, {cond}")
        print_table(rows_al, ["variant"] + cols, f"Share of chains still correct at the start of stage B, {cond}")
        print_table(rows_lag, ["variant"] + cols, f"Mean in-flight tokens before the anchor of stage B, {cond}")
        print_table(rows_s, ["variant", "E ms"] + cols + ["best B"],
                    f"Speed against one block per verify, {cond} (D = {args.D}, F = {args.F} + {args.F_slope}/token; "
                    f"E = feature or cut pass per stage, ms)")
        out["tables"][cond] = {"A": rows_a, "added": rows_d, "ratio": rows_f, "alive": rows_al, "lag": rows_lag,
                               "speed": rows_s}
    summ = []
    for cond in conds:
        t = out["tables"][cond]
        row = {"in-flight features": cond,
               "ms per stage": args.layer_ms * int(cond[1:]) if cond.startswith("L") else 0.0}
        for name in ("oracle / mask", "oracle / mask retry", f"conf<{args.thrs[0]:g} / plain"):
            ra = next(r for r in t["A"] if r["variant"] == name)
            rs = next(r for r in t["speed"] if r["variant"] == name)        # the first cost row of this variant
            al = next(r for r in t["alive"] if r["variant"] == name)
            if name == "oracle / mask":
                row["survive cut 1"] = al["B=3"] if B >= 3 else float("nan")
                row["A(2)"] = ra["B=2"] if B >= 2 else float("nan")
                row[f"A({B})"] = ra[f"B={B}"]
            row[f"speed {name}"] = max(rs[c] for c in cols)
            row[f"B {name}"] = rs["best B"]
        summ.append(row)
    print_table(summ, list(summ[0]), "Summary: in-flight features by source, each charged its own cost per stage "
                                     "(real and none: free)")
    out["summary"] = summ
    save_json(args.save, out)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
