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
anchor of stage 1; the lag grows with every stage and is reported).

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

    def blk(feats, x, s, lag=0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return blocks_logits(draft, target, feats, x, [(s, lag)], bs)[0].float()

    variants = [("oracle", "plain"), ("oracle", "mask"), ("oracle", "mask retry")]
    for t in args.thrs:
        variants += [(f"conf<{t:g}", "plain"), (f"conf<{t:g}", "soft")]
    keys = [(c, v) for c in CONDS for v in variants]
    acc = {k: [[] for _ in range(B)] for k in keys}          # A after stage b
    length = {k: [[] for _ in range(B)] for k in keys}       # tokens handed to the verify after stage b
    alive = {k: [0] * B for k in keys}                       # chains correct up to the anchor of stage b
    lags = {k: [[] for _ in range(B)] for k in keys}         # in-flight tokens before the anchor of stage b

    def run(cond, rule, sup, x, feats, s, truth, first):
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
                R = blk(feats[:, :a], xk, a) if cond == "real" else blk(feats[:, :s], xk, a, a - s)
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
        feats = extract_context_feature(hs, lids)
        truth_all = x.tolist()
        for s in anchors:
            n_used += 1
            truth = truth_all[s + 1 :]
            first = blk(feats[:, :s], x, s)
            for cond, (rule, sup) in keys:
                for b, (a_b, n_b, al, lag) in enumerate(run(cond, rule, sup, x, feats, s, truth, first)):
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
    for cond in CONDS:
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
            for e_ms in args.E:
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
    save_json(args.save, out)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
