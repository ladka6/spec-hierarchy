"""Experiment 50 (CPU): can the drafter's block tell the MoE which experts the block will need?

Block-shared routing (exp47/49) picks a block's S experts from the PREVIOUS block, which is stale. A drafter has
a guess of the CURRENT block (its draft). exp45 --save holds, per DFlash block, the Qwen3-30B-A3B routing of the
drafted tokens and of the true continuation (index 0 = anchor). Here: build a set of S experts per layer from
different sources and measure how well it covers the experts the TRUE tokens route to.

Comparison on the second half of each block (true tokens 8..15), where all predictors are available:
  stale     top-S of the true tokens 0..7 (the previous 8 tokens: today's design with n=8)
  draft     top-S of the drafted tokens 8..15 routed by the target (upper bound for a perfect draft->routing head)
  oracle    top-S of the true tokens 8..15 themselves (best possible set of size S)
and on the full block (true 0..15): draft 0..15 vs oracle.
Coverage = share of the true tokens' (token, expert) picks that fall inside the set; also the share of tokens
whose whole top-k is inside. Split by accepted length (draft quality).

Part 2 (what matters for committed tokens): the committed outputs of a round are computed on inputs that are the
anchor and the ACCEPTED drafts (positions 0..A), whose routing the draft gives exactly; rejected inputs only matter
through which experts they pull into the set. So: coverage of positions 0..A only, for sets built from all drafts,
from drafts weighted by the drafter's confidence (cumulative product, ~ P(still accepted)), and the best set of size
S for those positions. Then the LOSSLESS variant: verify with set S (built from the confident draft) plus "escaped"
tokens that use their own top-k (anchor always; drafts with cumulative confidence >= c). A position computed with all
its picks inside the set (or escaped) in every layer is computed exactly as by the original model, and the router can
check this at runtime, so the round commits tokens up to the first accepted position that is not exact. Cost per
layer = |set U experts of escaped tokens|. Speedup at batch 1 in the weights-read model of exp46 (drafter 0.2 step);
S=0 is pure confidence truncation, "escape all" is DFlash.

  python scripts/exp50_draftsets.py --routing /scratch-shared/$USER/hspec_moe/routing.pt
"""

from __future__ import annotations

import argparse

import torch


def topS(counts, S):
    g = torch.Generator().manual_seed(0)
    noise = torch.rand(counts.shape, generator=g) * 1e-3                 # break count ties deterministically
    keep = torch.zeros(counts.shape, dtype=torch.bool)
    keep.scatter_(-1, (counts + noise).topk(S, -1).indices, True)
    return keep                                                          # [n, L, E]


def counts_of(rout, E):
    """rout [n, L, t, k] -> [n, L, E] number of picks per expert."""
    n, L = rout.shape[:2]
    c = torch.zeros(n, L, E)
    c.scatter_add_(2, rout.reshape(n, L, -1), torch.ones(n, L, rout.shape[2] * rout.shape[3]))
    return c


def coverage(keep, target):
    """keep [n, L, E] bool, target [n, L, t, k] -> (pick coverage [n], whole-token coverage [n])."""
    inside = keep.gather(2, target.reshape(*target.shape[:2], -1)).reshape(target.shape)   # [n, L, t, k]
    return inside.float().mean((1, 2, 3)), inside.all(-1).float().mean((1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True)
    ap.add_argument("--sets", nargs="+", type=int, default=[8, 12, 16, 24, 32])
    args = ap.parse_args()

    d = torch.load(args.routing)
    A = d["A"].long()
    dr, tr = d["draft_rout"].long(), d["true_rout"].long()               # [n, L, 16, k]
    n, L, T, k = dr.shape
    E = int(max(dr.max(), tr.max())) + 1
    print(f"{n} blocks, {L} layers, {E} experts, top-{k}; mean accepted {A.float().mean():.2f}")

    h = T // 2
    tgt2, tgt = tr[:, :, h:], tr
    src = {"stale (true 0..7)": counts_of(tr[:, :, :h], E),
           "draft (drafted 8..15)": counts_of(dr[:, :, h:], E),
           "stale + draft": counts_of(tr[:, :, :h], E) + counts_of(dr[:, :, h:], E),
           "oracle (true 8..15)": counts_of(tgt2, E)}
    buckets = {"all": torch.ones(n, dtype=torch.bool), "A<=3": A <= 3, "4<=A<=9": (A >= 4) & (A <= 9), "A>=10": A >= 10}

    print("\n== second half of the block (true tokens 8..15): share of expert picks inside the set "
          "(share of tokens fully inside) ==")
    print(f"{'S':>3s} {'source':24s} " + " ".join(f"{b:>15s}" for b in buckets))
    for S in args.sets:
        for name, c in src.items():
            pc, tc = coverage(topS(c, S), tgt2)
            print(f"{S:3d} {name:24s} " + " ".join(f"{float(pc[m].mean()):6.3f} ({float(tc[m].mean()):5.3f})"
                                                     for m in buckets.values()))
        print()

    print("== full block (anchor + 15, true tokens 0..15) ==")
    full = {"draft (anchor + drafted)": counts_of(dr, E), "oracle (true 0..15)": counts_of(tgt, E)}
    print(f"{'S':>3s} {'source':24s} " + " ".join(f"{b:>15s}" for b in buckets))
    for S in args.sets:
        for name, c in full.items():
            pc, tc = coverage(topS(c, S), tgt)
            print(f"{S:3d} {name:24s} " + " ".join(f"{float(pc[m].mean()):6.3f} ({float(tc[m].mean()):5.3f})"
                                                     for m in buckets.values()))
        print()
    print("blocks per bucket: " + ", ".join(f"{b} {int(m.sum())}" for b, m in buckets.items()))
    part2(d, args, buckets)


def part2(d, args, buckets):
    A, dprob = d["A"].long(), d["dprob"].float()
    dr, tr = d["draft_rout"].long(), d["true_rout"].long()
    n, L, T, k = dr.shape
    E = int(max(dr.max(), tr.max())) + 1
    pos = torch.arange(T)
    need = pos[None] <= A[:, None]                                       # [n, 16] inputs of committed outputs
    same = (dr == tr).all(-1).all(1)[need].float().mean()
    print(f"\n== part 2: committed tokens only (positions 0..A) ==\nsanity: draft routing == true routing on accepted "
          f"positions for {float(same):.1%} of them (differences = numerics)")
    tok = torch.zeros(n, L, T, E, dtype=torch.bool)
    tok.scatter_(3, dr, True)                                            # what the verify pass actually computes
    w = torch.cat([torch.ones(n, 1), dprob.cumprod(1)], 1)               # [n, 16] ~ P(position still accepted)
    srcs = {"draft, all 16": torch.ones(n, T), "draft, confidence-weighted": w, "best set for 0..A": need.float()}
    print(f"{'S':>3s} {'set built from':28s} " + " ".join(f"{b:>22s}" for b in buckets))
    print(f"{'':32s}" + " ".join(f"{'picks / tok-layer / tok-all':>22s}" for _ in buckets))
    for S in args.sets:
        for name, wt in srcs.items():
            keep = topS(torch.einsum("nt,nlte->nle", wt, tok.float()), S)
            inside = keep[:, :, None, :] | ~tok                          # [n, L, T, E]: pick inside or not a pick
            pick = (tok & keep[:, :, None, :]).sum(-1).float() / k       # [n, L, T]
            full = inside.all(-1)                                        # [n, L, T]
            cells = []
            for m in buckets.values():
                nm = need & m[:, None]
                cells.append(f"{float(pick.mean(1)[nm].mean()):.3f}/{float(full.float().mean(1)[nm].mean()):.3f}/"
                              f"{float(full.all(1)[nm].float().mean()):.3f}")
            print(f"{S:3d} {name:28s} " + " ".join(f"{c:>22s}" for c in cells))
        print()

    ne, ex = float(d["non_expert"]), float(d["expert"])
    ar1 = ne + ex * float(tok[:, :, 0].sum((1, 2)).float().mean())
    t_draft = 0.2 * ar1
    print("== lossless verification with a draft-built expert set + escaped tokens (batch 1, weights-read model) ==")
    print(f"{'S':>3s} {'escape':>10s} {'committed':>9s} {'vs A+1':>7s} {'experts/layer':>13s} {'speedup':>8s}")
    rows = [(0, "all")] + [(S, c) for S in [0] + args.sets for c in (0.9, 0.5, 0.1, "anchor")]
    for S, c in rows:
        keep = topS(torch.einsum("nt,nlte->nle", w, tok.float()), S) if S else torch.zeros(n, L, E, dtype=torch.bool)
        esc = torch.ones(n, T, dtype=torch.bool) if c == "all" else (w >= (2.0 if c == "anchor" else c))
        esc[:, 0] = True
        exact = esc | (keep[:, :, None, :] | ~tok).all(-1).all(1)        # [n, 16] position computed exactly
        bad = ~exact & need
        committed = torch.where(bad.any(1), bad.float().argmax(1), A + 1)
        u = (keep | (tok & esc[:, None, :, None]).any(2)).sum(-1).float()  # [n, L]
        cost = t_draft + ne + ex * u.sum(1)
        sp = float(committed.sum()) / float(cost.sum()) * ar1
        lab = "DFlash" if c == "all" else str(c)
        print(f"{S:3d} {lab:>10s} {float(committed.float().mean()):9.2f} {float((committed.float() / (A + 1)).mean()):7.3f} "
              f"{float(u.mean()):13.1f} {sp:8.2f}")
    need_u = (tok & need[:, None, :, None]).any(2).sum(-1).float()
    sp = float((A + 1).sum()) / float((t_draft + ne + ex * need_u.sum(1)).sum()) * ar1
    print(f"oracle truncation (load exactly the experts of 0..A): {float(need_u.mean()):.1f} experts/layer, speedup {sp:.2f}")


if __name__ == "__main__":
    main()
