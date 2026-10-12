"""Experiment 48 (CPU): what is a small drafter-side calibrator worth on an MoE target?

exp46: a confidence threshold recovers ~70% of the oracle's gain at batch 1. The rest needs a better
estimate of (a) how far a block will be accepted and (b) what verifying each extra token will cost.
Calibrator here = a tiny logistic model on the drafter's own confidences:
    P(A >= j) = sigmoid(w1 * sum_{i<=j} log p_i + w2 * min_{i<=j} p_i + w3 * j + w4)
fit by cross-fitting (two halves; each half scored by the model fit on the other). Given P(A >= j),
the expected tokens of verifying L drafted tokens is 1 + sum_{j<=L} P(A >= j); the policy picks the L
that maximises expected tokens / (drafter + cost(L)) (exact for one block; Dinkelbach for a stream).
cost(L) either from the average expert-union curve (realistic: no expert predictor) or from the block's
own routing (upper bound for a perfect expert-footprint predictor).

Compared at batch 1 and 4 (weights-read cost model, exp45 routing):
  DFlash (L=15), best confidence threshold, calibrated + average cost, calibrated + exact cost, oracle.

  python scripts/exp48_calibrator.py --routing /scratch-shared/$USER/hspec_moe/routing.pt
"""

from __future__ import annotations

import argparse
import random

import torch


def features(dprob):
    lp = dprob.clamp_min(1e-6).log().cumsum(1)                     # [n, 15]
    mn = dprob.cummin(1).values
    j = torch.arange(1, dprob.shape[1] + 1, dtype=torch.float).expand_as(dprob)
    return torch.stack([lp, mn, j / 15.0, torch.ones_like(lp)], -1)    # [n, 15, 4]


def fit_logistic(X, y, steps=500, lr=0.1):
    w = torch.zeros(X.shape[-1], requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--groups", type=int, default=300)
    ap.add_argument("--t-draft-frac", type=float, default=0.2)
    args = ap.parse_args()

    d = torch.load(args.routing)
    A, dprob = d["A"].long(), d["dprob"].float()
    rout = d["draft_rout"].long()
    n, L, T, k = rout.shape
    E = int(rout.max()) + 1
    ne, ex = float(d["non_expert"]), float(d["expert"])
    mask = torch.zeros(n, L, T, E, dtype=torch.bool)
    mask.scatter_(3, rout, True)
    for t in range(1, T):
        mask[:, :, t] |= mask[:, :, t - 1]
    U = mask.sum(-1).sum(1).float()                                 # [n, 16] sum over layers of union for 0..t
    cost_exact = ne + ex * U                                        # weights read verifying anchor + t drafts
    cost_avg = cost_exact.mean(0, keepdim=True).expand_as(cost_exact)
    ar1 = float(cost_exact[:, 0].mean())
    t_draft = args.t_draft_frac * ar1

    # calibrator: P(A >= j), cross-fitted
    X = features(dprob)                                             # [n, 15, 4]
    Y = (A.view(-1, 1) >= torch.arange(1, T).view(1, -1)).float()   # [n, 15]
    half = torch.randperm(n, generator=torch.Generator().manual_seed(0)) < n // 2
    P = torch.zeros(n, T - 1)
    for tr in (half, ~half):
        w = fit_logistic(X[tr].reshape(-1, 4), Y[tr].reshape(-1))
        P[~tr] = torch.sigmoid(X[~tr] @ w)
    brier = float(((P - Y) ** 2).mean())
    conf_brier = float(((dprob.cumprod(1) - Y) ** 2).mean())
    print(f"calibrator Brier {brier:.4f} vs raw cumulative-product confidence {conf_brier:.4f}")

    exp_tok = torch.cat([torch.ones(n, 1), 1 + P.cumsum(1)], 1)      # [n, 16]: expected tokens verifying L drafts

    def choose(cost):
        return (exp_tok / (t_draft + cost)).argmax(1)               # L in 0..15

    pols = {"DFlash (L=15)": torch.full((n,), T - 1)}
    for c in (0.3, 0.5, 0.7):
        pols[f"conf>={c}"] = (dprob >= c).long().cumprod(1).sum(1)
    pols["calibrated + avg cost"] = choose(cost_avg)
    pols["calibrated + exact cost"] = choose(cost_exact)
    pols["oracle"] = A.clone()

    rng = random.Random(0)
    for B in args.batches:
        groups = [torch.tensor(rng.sample(range(n), B)) for _ in range(args.groups)]
        ar = sum(ne + ex * float(mask[g][:, :, 0].any(0).sum()) for g in groups) / len(groups)
        res = {}
        for name, Lp in pols.items():
            tok = cost = 0.0
            for g in groups:
                Lg = Lp[g]
                sel = mask[g].gather(2, Lg.view(-1, 1, 1, 1).expand(-1, L, 1, E)).squeeze(2)
                tok += float((torch.minimum(A[g], Lg) + 1).sum())
                cost += t_draft + ne + ex * float(sel.any(0).sum())
            res[name] = ((tok / cost) / (B / ar), tok / len(groups) / B, float(Lp.float().mean()))
        base = res["DFlash (L=15)"][0]
        orc = res["oracle"][0]
        print(f"\n== batch {B} ==")
        print(f"{'policy':26s} {'speedup':>8s} {'vs DFlash':>9s} {'share of oracle gain':>21s} {'tau':>6s} {'mean L':>7s}")
        for name, (sp, tau, ml) in res.items():
            share = (sp - base) / (orc - base) if orc > base else float("nan")
            print(f"{name:26s} {sp:8.2f} {sp / base:9.3f} {share:21.1%} {tau:6.2f} {ml:7.2f}")


if __name__ == "__main__":
    main()
