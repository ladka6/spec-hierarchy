"""Experiment 52 (CPU): speculative verification on an MoE as a submodular problem.

Setting (memory-bound decoding). A verify pass over a set V of tokens reads
    C(V) = W_ne + w_e * sum_l | U_{t in V} E_t^l |          (E_t^l: experts of token t at layer l)
The union is a coverage function, so C is monotone submodular in V; for a dense model it does not depend on V.
A round commits min(A, l) + 1 tokens when the anchor and the first l drafts are verified (A = accepted drafts), so
with q_j = P(A >= j) the expected gain is G(l) = 1 + sum_{j<=l} q_j (modular along the prefix chain). Throughput
is G / (d + C). Consequences tested here, with exp45's routing (Qwen3-30B-A3B experts of 8B-DFlash blocks):

  1. A token whose experts are already covered costs nothing: verify it whatever its q_j.
  2. The router of layer l is computed BEFORE that layer's experts are read, so a pass can decide per layer which
     prefix to keep (dropping later tokens is exact: attention is causal), using the layer's true footprint.
     Policy "layer-wise": at each layer pick l_l <= l_{l-1} maximising G(l) - lam * (cost of layer l for prefix l
     + (L-l-1) * the same as an estimate for the layers left). lam is cross-fitted (chosen on one half of the
     blocks, scored on the other). Committed tokens = min(A, l_final) + 1.
  3. Batches: the cost is the union over all requests, so a request's tokens are cheap where other requests already
     loaded the experts. Layer-wise greedy for the batch: per layer, coordinate ascent over requests with the
     others' footprint fixed.
Compared: DFlash (l=15), confidence truncation (cumulative confidence >= c, c cross-fitted), the best pre-pass
choice of l with the whole footprint known (exp48's upper bound), layer-wise greedy, oracle (l = A).
q_j = cumulative product of the drafter's top-1 probabilities (as calibrated as exp48's calibrator).
Also reported: the locality coefficient kappa(n) = real union of n consecutive tokens / union if each token picked
k experts uniformly at random, which sets how fast C grows along the chain.

  python scripts/exp52_submodular.py --routing /scratch-shared/$USER/hspec_moe/routing.pt
"""

from __future__ import annotations

import argparse
import random

import torch


def prefix_masks(rout, E):
    """rout [n, L, T, k] -> cum [n, L, T, E] bool: experts of tokens 0..t."""
    n, L, T, k = rout.shape
    m = torch.zeros(n, L, T, E, dtype=torch.bool)
    m.scatter_(3, rout, True)
    for t in range(1, T):
        m[:, :, t] |= m[:, :, t - 1]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--groups", type=int, default=150)
    ap.add_argument("--t-draft-frac", type=float, default=0.2)
    args = ap.parse_args()

    d = torch.load(args.routing)
    A, dprob = d["A"].long(), d["dprob"].float()
    rout = d["draft_rout"].long()
    n, L, T, k = rout.shape
    E = int(rout.max()) + 1
    ne, ex = float(d["non_expert"]), float(d["expert"])
    cum = prefix_masks(rout, E)                                       # [n, L, T, E]
    U = cum.sum(-1).float()                                           # [n, L, T] per-layer union of tokens 0..t
    q = dprob.cumprod(1)                                              # [n, 15] P(A >= j), j = 1..15
    G = torch.cat([torch.ones(n, 1), 1 + q.cumsum(1)], 1)            # [n, 16] expected gain verifying l drafts
    ar1 = ne + ex * float(U[:, :, 0].sum(1).mean())
    t_draft = args.t_draft_frac * ar1
    half = torch.zeros(n, dtype=torch.bool)
    half[torch.randperm(n, generator=torch.Generator().manual_seed(0))[: n // 2]] = True

    # locality coefficient
    rnd = torch.tensor([E * (1 - (1 - k / E) ** t) for t in range(1, T + 1)])
    real = U.mean((0, 1))
    print(f"{n} blocks, {L} layers, {E} experts, top-{k}; mean accepted {A.float().mean():.2f}")
    print("locality: tokens  " + " ".join(f"{t:5d}" for t in (1, 2, 4, 8, 12, 16)))
    print("  union, real     " + " ".join(f"{float(real[t - 1]):5.1f}" for t in (1, 2, 4, 8, 12, 16)))
    print("  union, random   " + " ".join(f"{float(rnd[t - 1]):5.1f}" for t in (1, 2, 4, 8, 12, 16)))
    print("  kappa(n)        " + " ".join(f"{float(real[t - 1] / rnd[t - 1]):5.2f}" for t in (1, 2, 4, 8, 12, 16)))

    # ---------- batch 1 policies ----------
    def score(l_final, cost, idx):
        com = (torch.minimum(A[idx], l_final[idx]) + 1).float()
        return float(com.sum()) / float(cost[idx].sum()) * ar1, float(com.mean())

    full_cost = t_draft + ne + ex * U.sum(1)                          # [n, 16] cost if prefix l kept in all layers

    def layerwise(lam):
        ell = torch.full((n,), T - 1)
        cost = torch.full((n,), t_draft + ne)
        ar = torch.arange(T)
        for l in range(L):
            val = G - lam * ex * (L - l) * U[:, l]                    # this layer + same-size estimate for the rest
            val = val.masked_fill(ar[None] > ell[:, None], -1e30)
            ell = val.argmax(1)
            cost += ex * U[:, l].gather(1, ell[:, None]).squeeze(1)
        return ell, cost

    def crossfit(make, grid):
        """choose the parameter on one half, score on the other; returns pooled (speedup, committed)."""
        tot_c = tot_w = 0.0
        com_all = []
        for tr in (half, ~half):
            best = max(grid, key=lambda g: score(*make(g), tr)[0])
            l_f, cost = make(best)
            te = ~tr
            c = (torch.minimum(A[te], l_f[te]) + 1).float()
            tot_c += float(c.sum())
            tot_w += float(cost[te].sum())
            com_all.append(c)
        return tot_c / tot_w * ar1, float(torch.cat(com_all).mean())

    rows = {}
    allidx = torch.ones(n, dtype=torch.bool)
    l15 = torch.full((n,), T - 1)
    rows["DFlash (l=15)"] = score(l15, full_cost[:, -1], allidx)
    rows["confidence truncation"] = crossfit(
        lambda c: ((q >= c).long().sum(1), full_cost.gather(1, (q >= c).long().sum(1)[:, None]).squeeze(1)),
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    lp = (G / full_cost).argmax(1)
    rows["pre-pass, footprint known"] = score(lp, full_cost.gather(1, lp[:, None]).squeeze(1), allidx)
    lams = [x / ar1 for x in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0)]
    rows["layer-wise greedy"] = crossfit(layerwise, lams)
    rows["oracle (l=A)"] = score(A, full_cost.gather(1, A[:, None]).squeeze(1), allidx)
    print("\n== batch 1 (weights-read model, drafter = 0.2 plain step) ==")
    print(f"{'policy':28s} {'speedup':>8s} {'committed':>9s}")
    for name, (sp, com) in rows.items():
        print(f"{name:28s} {sp:8.2f} {com:9.2f}")

    # ---------- batches ----------
    rng = random.Random(0)
    for B in [b for b in args.batches if b > 1]:
        groups = [torch.tensor(rng.sample(range(n), B)) for _ in range(args.groups)]
        plain = sum(ne + ex * float(cum[g][:, :, 0].any(0).sum()) for g in groups) / len(groups)

        def run(policy, param):
            tok = cost = 0.0
            for g in groups:
                cg, Gg, Ag, qg = cum[g], G[g], A[g], q[g]                 # cg [B, L, T, E]
                if policy == "fixed":
                    ell = param[g] if torch.is_tensor(param) else torch.full((B,), param)
                    sel = cg.gather(2, ell.view(-1, 1, 1, 1).expand(-1, L, 1, E)).squeeze(2)
                    c = t_draft + ne + ex * float(sel.any(0).sum())
                else:                                                     # layer-wise greedy for the batch
                    lam = param
                    ell = torch.full((B,), T - 1)
                    c = t_draft + ne
                    for l in range(L):
                        for _ in range(2):
                            for b in range(B):
                                others = cg[torch.arange(B) != b, l].gather(
                                    1, ell[torch.arange(B) != b].view(-1, 1, 1).expand(-1, 1, E)).squeeze(1).any(0)
                                u = (cg[b, l, : ell[b] + 1] | others).sum(-1).float()    # [ell_b + 1]
                                val = Gg[b, : ell[b] + 1] - lam * ex * (L - l) * u
                                ell[b] = int(val.argmax())
                        sel = cg[:, l].gather(1, ell.view(-1, 1, 1).expand(-1, 1, E)).squeeze(1)
                        c += ex * float(sel.any(0).sum())
                tok += float((torch.minimum(Ag, ell) + 1).sum())
                cost += c
            return tok / cost / (B / plain), tok / len(groups) / B

        conf = {c: (q >= c).long().sum(1) for c in (0.1, 0.3, 0.5)}
        res = {"DFlash (l=15)": run("fixed", T - 1)}
        for c, lc in conf.items():
            res[f"confidence >= {c}"] = run("fixed", lc)
        for x in (0.05, 0.1, 0.2, 0.3, 0.5):
            res[f"layer-wise greedy lam={x}"] = run("layer", x / ar1)
        res["oracle (l=A)"] = run("fixed", A)
        print(f"\n== batch {B} ({args.groups} random batches) ==")
        print(f"{'policy':28s} {'speedup':>8s} {'committed':>9s}")
        for name, (sp, com) in res.items():
            print(f"{name:28s} {sp:8.2f} {com:9.2f}", flush=True)


if __name__ == "__main__":
    main()
