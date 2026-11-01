"""Experiment 84: how much of L25 / L33 does DFlash actually use? (no training)

Chained round as in exp80-83 (oracle cut n, block 2 at anchor s+n, frozen original DFlash, all rounds, tokens per
verification n + m2 + 1). The in-flight positions get the REAL deep features, but only their component inside a
rank-r subspace (the rest set to the dataset mean); shallow taps at their mean:

    F_r = mu + U_r U_r^T (F - mu)        per layer (L25, L33), U_r orthonormal [H, r]

Subspaces, fitted on the held-in part of --cache (never the eval sequences):
  pca      top-r principal directions of the real states (data variance)
  fisher   top-r eigenvectors of the drafter's empirical Fisher: E[g g^T], g = gradient of block 2's cross-entropy on
           the true continuation with respect to the in-flight features (directions that change DFlash's output)
  random   a random r-dim subspace (control)
If a small r keeps most of the ceiling, an estimator only has to predict an r-dim code per position.

  python scripts/exp84_subspace.py --cache seqs.pt --fit 300 --stops 600
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

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--fit", type=int, default=300, help="rounds (held-in sequences) for the PCA / Fisher fit")
    ap.add_argument("--stops", type=int, default=600)
    ap.add_argument("--ranks", nargs="+", type=int, default=[16, 64, 256, 1024])
    ap.add_argument("--save", default="exp84_subspace")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    nv = max(int(len(seqs) * 0.2), 1)
    val, held = seqs[:nv], seqs[nv:]
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H, K = target.config.hidden_size, len(lids)
    deep = {"25": lids.index(25), "33": lids.index(33)}

    @torch.no_grad()
    def feats_of(x):
        out = target(x[None], output_hidden_states=True, use_cache=False, logits_to_keep=1)
        return extract_context_feature(out.hidden_states, lids)[0]

    def rounds(pool, k, seed):
        r = random.Random(seed)
        cand = [(i, s) for i, (ids, lo) in enumerate(pool) for s in range(lo + 8, len(ids) - 3 * bs)]
        r.shuffle(cand)
        by = {}
        for i, s in cand[:k]:
            by.setdefault(i, []).append(s)
        return by

    # ------------------------------------------------------------------ fit: mean, covariance, Fisher
    S1 = torch.zeros(K, H, dtype=torch.float64, device=dev)
    S2 = {t: torch.zeros(H, H, dtype=torch.float64, device=dev) for t in deep}
    G = {t: torch.zeros(H, H, dtype=torch.float64, device=dev) for t in deep}
    n_pos = n_grad = 0
    fit = rounds(held, args.fit, 11)
    for i, anchors in fit.items():
        ids, lo = held[i]
        x = ids.to(dev)
        f = feats_of(x)
        fr = f[lo:].double().view(-1, K, H)
        S1 += fr.sum(0)
        for t, ti in deep.items():
            S2[t] += fr[:, ti].T @ fr[:, ti]
        n_pos += fr.shape[0]
        for s in anchors:
            with torch.no_grad():
                y1 = blocks_logits(draft, target, f[None, :s], x, [(s, 0)], bs)[0].argmax(-1).tolist()
            n = matched(y1, ids[s + 1 :].tolist())
            if n == 0:
                continue
            a = s + n
            inflight = f[s:a].float().clone().requires_grad_(True)
            mix = torch.cat([f[:s], inflight.to(f.dtype)], 0)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = blocks_logits(draft, target, mix, x, [(a, 0)], bs)[0].float()
            loss = F.cross_entropy(lg, x[a + 1 : a + bs])
            loss.backward()
            g = inflight.grad.double().view(n, K, H)
            for t, ti in deep.items():
                G[t] += g[:, ti].T @ g[:, ti]
            n_grad += n
    mu = (S1 / n_pos).float()
    basis = {}
    for t, ti in deep.items():
        m = mu[ti].double()
        cov = S2[t] / n_pos - torch.outer(m, m)
        ev_c, U_c = torch.linalg.eigh(cov)
        ev_f, U_f = torch.linalg.eigh(G[t] / max(n_grad, 1))
        U_r, _ = torch.linalg.qr(torch.randn(H, H, dtype=torch.float64, device=dev, generator=torch.Generator(
            device=dev).manual_seed(0)))
        basis[t] = {"pca": (U_c.flip(1).float(), ev_c.flip(0)), "fisher": (U_f.flip(1).float(), ev_f.flip(0)),
                    "random": (U_r.float(), None)}
    print(f"[fit] {n_pos} positions for PCA, {n_grad} in-flight positions with gradients, {time.time() - t0:.0f}s",
          flush=True)
    share = {}
    for t in deep:
        for name in ("pca", "fisher"):
            ev = basis[t][name][1].clamp_min(0)
            share[f"{name} L{t}"] = {r: float(ev[:r].sum() / ev.sum()) for r in args.ranks}
    for k_, v in share.items():
        print(f"[fit] {k_}: share of total by rank " + "  ".join(f"r={r}: {s:.3f}" for r, s in v.items()), flush=True)

    # ------------------------------------------------------------------ eval
    conds = ["real", "mean"] + [f"{b}{r}" for b in ("pca", "fisher", "random") for r in args.ranks]

    def project(e, cond):
        """e [n, K*H] real in-flight features -> shallow taps at mean, deep taps projected"""
        n = e.shape[0]
        ev = e.float().view(n, K, H)
        out = mu[None].expand(n, K, H).clone()
        if cond == "real":
            return ev.view(n, K * H)
        if cond == "mean":
            return out.view(n, K * H)
        name = cond.rstrip("0123456789")
        r = int(cond[len(name):])
        for t, ti in deep.items():
            U = basis[t][name][0][:, :r]
            d = ev[:, ti] - mu[ti]
            out[:, ti] = mu[ti] + (d @ U) @ U.T
        return out.view(n, K * H)

    recs = []
    with torch.inference_mode():
        for done, (i, anchors) in enumerate(rounds(val, args.stops, 7).items()):
            ids, lo = val[i]
            x = ids.to(dev)
            f = feats_of(x)
            for s in anchors:
                y1 = blocks_logits(draft, target, f[None, :s], x, [(s, 0)], bs)[0].argmax(-1).tolist()
                n = matched(y1, ids[s + 1 :].tolist())
                r = {"n": n}
                if n == 0:
                    r.update({c: 0 for c in conds})
                    recs.append(r)
                    continue
                a = s + n
                truth = ids[a + 1 :].tolist()
                for c in conds:
                    fz = torch.cat([f[:s], project(f[s:a], c).to(f.dtype)], 0)[None]
                    lg = blocks_logits(draft, target, fz, x, [(a, 0)], bs)[0]
                    r[c] = matched(lg.argmax(-1).tolist(), truth)
                recs.append(r)
            if done % 10 == 0:
                print(f"[eval] {done} sequences, {len(recs)} rounds, {time.time() - t0:.0f}s", flush=True)

    k = len(recs)
    normal = sum(r["n"] + 1 for r in recs) / k
    real = sum(r["n"] + r["real"] + 1 for r in recs) / k
    rows = [{"in-flight features": "none (normal DFlash)", "tokens / verification": normal, "share of ceiling gain": 0.0}]
    for c in conds:
        t = sum(r["n"] + r[c] + 1 for r in recs) / k
        rows.append({"in-flight features": c, "tokens / verification": t,
                     "share of ceiling gain": (t - normal) / max(real - normal, 1e-9)})
    print_table(rows, list(rows[0]), f"real deep features restricted to a rank-r subspace (rest = mean), {k} rounds, "
                                     f"oracle cut, frozen DFlash")
    save_json(args.save, {"args": vars(args), "rows": rows, "share": share})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
