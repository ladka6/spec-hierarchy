"""Experiment 46 (CPU): verification policies for block drafters on an MoE target, per block and per batch size.

Uses exp45 --save: per DFlash block, the accepted length, the drafter's confidence per position, and the
experts every verified token routes to in every layer of Qwen3-30B-A3B (anchor + 15 drafted tokens).
Cost model: weights read per forward (memory-bound decoding) = non-expert weights + expert size x sum over
layers of the number of distinct experts needed by ALL tokens in the forward (all requests of the batch).
  plain decoding, batch B   B tokens (each request's anchor), cost of their expert union
  speculative, batch B      drafter (fixed cost) + verify of each request's anchor + its first L_b drafts
Policies per request: fixed L, confidence c (verify while drafter prob >= c), oracle (L = accepted).
Batches are random groups of B blocks (requests at different points of different texts). Reported per B:
speedup over plain decoding at the same B, eta = speedup / tau, the share of verification weights that
serve only rejected tokens, and the same numbers for an equally accepted dense target (verify ~ free).

  python scripts/exp46_moe_policy.py --routing /scratch-shared/$USER/hspec_moe/routing.pt
"""

from __future__ import annotations

import argparse
import random

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--groups", type=int, default=400, help="random batches per batch size")
    ap.add_argument("--t-draft-frac", type=float, default=0.2, help="drafter cost / batch-1 plain step cost")
    args = ap.parse_args()

    d = torch.load(args.routing)
    A, dprob = d["A"].long(), d["dprob"].float()
    rout = d["draft_rout"].long()                                   # [n, L, 16, k]; index 0 = anchor
    n, L, T, k = rout.shape
    E = int(rout.max()) + 1
    ne, ex = float(d["non_expert"]), float(d["expert"])
    mask = torch.zeros(n, L, T, E, dtype=torch.bool)
    mask.scatter_(3, rout, True)
    cum = mask.cummax(dim=2).values                                 # cum[:, :, i] = experts of tokens 0..i
    ar1 = ne + ex * float(mask[:, :, 0].sum((1, 2)).float().mean())  # batch-1 plain step (anchor only)
    t_draft = args.t_draft_frac * ar1
    print(f"{n} blocks, {L} layers, {E} experts, top-{k}; mean accepted {A.float().mean():.2f}; "
          f"plain step = {ar1 / 1e9:.2f}B weights, drafter = {t_draft / 1e9:.2f}B-equivalent")

    pols = {f"fixed L={l}": torch.full((n,), l) for l in (2, 4, 6, 8, 10, 12, 15)}
    for c in (0.1, 0.3, 0.5, 0.7):
        pols[f"conf>={c}"] = (dprob >= c).long().cumprod(1).sum(1)
    pols["oracle (L=accepted)"] = A.clone()
    rng = random.Random(0)
    rows = []
    for B in args.batches:
        groups = [torch.tensor(rng.sample(range(n), B)) for _ in range(args.groups)]
        ar_cost = sum(ne + ex * float(mask[g][:, :, 0].any(0).sum()) for g in groups) / len(groups)
        base = None
        for name, Lp in pols.items():
            tok = cost = waste_all = waste_need = 0.0
            for g in groups:
                Lg = Lp[g]
                sel = cum[g, :, :, :].gather(2, Lg.view(-1, 1, 1, 1).expand(-1, L, 1, E)).squeeze(2)   # [B, L, E]
                u = float(sel.any(0).sum())
                need = cum[g].gather(2, torch.minimum(A[g], Lg).view(-1, 1, 1, 1).expand(-1, L, 1, E)).squeeze(2)
                tok += float((torch.minimum(A[g], Lg) + 1).sum())
                cost += t_draft + ne + ex * u
                waste_all += ex * u
                waste_need += ex * float(need.any(0).sum())
            tok /= len(groups)
            cost /= len(groups)
            speed = (tok / cost) / (B / ar_cost)
            tau = tok / B
            dense = tau * 1.0 / (1.0 + args.t_draft_frac)            # dense: verify ~ plain step, eta ~ 1/(1+d)
            row = {"B": B, "policy": name, "tau": tau, "speedup_moe": speed, "eta_moe": speed / tau,
                   "wasted_expert_loads": 1 - waste_need / waste_all, "speedup_dense_equiv": dense}
            if name == "fixed L=15":
                base = speed
            rows.append(row)
        for r in rows:
            if r["B"] == B:
                r["vs_dflash"] = r["speedup_moe"] / base
        print(f"\n== batch {B}: plain step = {ar_cost / 1e9:.2f}B weights ==")
        print(f"{'policy':22s} {'tau':>6s} {'speedup':>8s} {'eta':>6s} {'waste':>6s} {'vs_DFlash':>9s} {'dense_equiv':>11s}")
        for r in rows:
            if r["B"] == B:
                print(f"{r['policy']:22s} {r['tau']:6.2f} {r['speedup_moe']:8.2f} {r['eta_moe']:6.3f} "
                      f"{r['wasted_expert_loads']:6.1%} {r['vs_dflash']:9.3f} {r['speedup_dense_equiv']:11.2f}")


if __name__ == "__main__":
    main()
