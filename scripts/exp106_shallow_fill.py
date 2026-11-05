"""Experiment 106: can a shallow exit state stand in for the deep taps DFlash reads? (no training)

For in-flight positions the target only runs its first k layers (early exit). DFlash reads the outputs of layers
lid + 1 for lid in draft.target_layer_ids (Llama DFlash: 1, 8, 15, 22, 29 -> hidden_states 2, 9, 16, 23, 30). Taps with
lid + 1 <= k are computed exactly by the exit; the deeper taps are missing. This script measures how close the exit
state h_k = hidden_states[k] is to those missing taps, and what the existing (frozen) drafter does if the missing taps
of the last m context positions are filled from h_k.

Part 1, closeness (response positions of held-out sequences), per exit k and missing tap:
  raw cosine      cos(h_k, tap)                (the metric of exp71's corruption test, which used raw tap vectors)
  centred cosine  cos(h_k - mu_k, tap - mu_tap)   (raw cosines are inflated by a few huge constant dimensions)
  norm ratio      |h_k| / |tap|
  ridge           the same cosines for a closed-form linear map h_k -> tap (ridge, fitted on separate sequences;
                  nothing is trained by gradient descent)
Part 2, the frozen drafter (exp71 protocol: anchor s, the last m context positions get substituted features, the
block from s is scored against the target's greedy continuation; tau = matched drafted tokens + 1):
  real            all taps real
  removed         the last m positions have no features at all (context ends at s - m)
  mean_k          missing taps = dataset mean of that tap (exact shallow taps real)
  ident_k         missing taps = h_k
  scaled_k        missing taps = h_k * (mean |tap| / mean |h_k|)
  ridge_k         missing taps = ridge(h_k)

Data: --data (directory of shard*.pt from gen_traindata.py, records {'ids', 'n_prompt'}) or --cache (a list of
(ids, n_prompt) tuples, e.g. from exp69). Both are the target's own greedy text.

  python scripts/exp106_shallow_fill.py --target facebook/layerskip-llama3-8B \
      --draft z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat --data ~/hspec_llama/data_layerskip-llama3-8B --exits 4 8 16
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


def load_seqs(args):
    seqs = []
    if args.cache:
        for ids, lo in torch.load(args.cache):
            seqs.append((ids.long(), int(lo)))
    else:
        for f in sorted(Path(args.data).expanduser().glob("shard*.pt")):
            for d in torch.load(f):
                seqs.append((d["ids"].long(), int(d["n_prompt"])))
    random.Random(0).shuffle(seqs)
    return [(ids[: args.max_len], lo) for ids, lo in seqs if len(ids[: args.max_len]) - lo > 32]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="facebook/layerskip-llama3-8B")
    ap.add_argument("--draft", default="z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat")
    ap.add_argument("--data", default="", help="directory with shard*.pt")
    ap.add_argument("--cache", default="", help="list of (ids, n_prompt), e.g. from exp69")
    ap.add_argument("--exits", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--m", type=int, default=8, help="in-flight positions whose features are substituted")
    ap.add_argument("--fit-tokens", type=int, default=100_000, help="tokens for the ridge fit")
    ap.add_argument("--val-tokens", type=int, default=10_000, help="tokens to pick the ridge strength")
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--anchors", type=int, default=1000)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--save", default="exp106_shallow_fill")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    dev, t0 = "cuda", time.time()

    seqs = load_seqs(args)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    hidx = [l + 1 for l in lids]                      # hidden_states index of each tap
    H, nT = target.config.hidden_size, len(lids)
    missing = {k: [j for j, h in enumerate(hidx) if h > k] for k in args.exits}
    print(f"[setup] {len(seqs)} sequences; taps hs{hidx}; missing per exit: "
          + ", ".join(f"k={k}: hs{[hidx[j] for j in v]}" for k, v in missing.items()), flush=True)

    def hidden(ids):
        return target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states

    # ---- ridge fit (closed form) on the first sequences: centred normal equations per exit, per missing tap ----
    pairs = [(k, j) for k in args.exits for j in missing[k]]
    XtX = {k: torch.zeros(H, H, device=dev) for k in args.exits}
    XtY = {p: torch.zeros(H, H, device=dev) for p in pairs}
    sx = {k: torch.zeros(H, device=dev) for k in args.exits}
    sy = {j: torch.zeros(H, device=dev) for j in range(nT)}
    nx = {k: 0.0 for k in args.exits}                 # norms for the scaled fill
    ny = {j: 0.0 for j in range(nT)}
    val = []
    n, i = 0, 0
    max_fit_seqs = int(0.6 * len(seqs))               # always keep >= 40% of the sequences held out
    while n < args.fit_tokens + args.val_tokens and i < max_fit_seqs:
        ids, lo = seqs[i]
        i += 1
        hs = hidden(ids)
        X = {k: hs[k][0, lo:].float() for k in args.exits}
        Y = {j: hs[h][0, lo:].float() for j, h in enumerate(hidx)}
        if n >= args.fit_tokens:
            val.append((X, Y))
        else:
            for k in args.exits:
                XtX[k] += X[k].T @ X[k]
                sx[k] += X[k].sum(0)
                nx[k] += X[k].norm(dim=-1).sum().item()
            for j in range(nT):
                sy[j] += Y[j].sum(0)
                ny[j] += Y[j].norm(dim=-1).sum().item()
            for k, j in pairs:
                XtY[(k, j)] += X[k].T @ Y[j]
        n += len(X[args.exits[0]])
    n_fit = min(n, args.fit_tokens)
    assert val, "no validation tokens left for the ridge strength: lower --fit-tokens or add data"
    fit_end = i
    mu_x = {k: sx[k] / n_fit for k in args.exits}
    mu_y = {j: sy[j] / n_fit for j in range(nT)}
    scale = {(k, j): (ny[j] / n_fit) / (nx[k] / n_fit) for k, j in pairs}

    def ccos(a, b, ma, mb):
        return F.cosine_similarity(a - ma, b - mb, dim=-1)

    W, best_lam = {}, {}
    for k in args.exits:
        C = XtX[k] / n_fit - torch.outer(mu_x[k], mu_x[k])
        tr = C.diagonal().mean()
        for j in missing[k]:
            Cxy = XtY[(k, j)] / n_fit - torch.outer(mu_x[k], mu_y[j])
            best = None
            for lam in args.lams:
                Wl = torch.linalg.solve(C + lam * tr * torch.eye(H, device=dev), Cxy)
                sc = torch.cat([ccos((X[k] - mu_x[k]) @ Wl + mu_y[j], Y[j], mu_y[j], mu_y[j]) for X, Y in val]).mean()
                if best is None or sc > best[0]:
                    best = (sc.item(), lam, Wl)
            best_lam[(k, j)], W[(k, j)] = best[1], best[2]
    del XtX, XtY, val
    print(f"[fit] {n_fit} tokens from {fit_end} sequences, {time.time() - t0:.0f}s", flush=True)

    def ridge(k, j, x):
        return ((x.float() - mu_x[k]) @ W[(k, j)] + mu_y[j]).to(x.dtype)

    # ---- held-out sequences: closeness stats and drafter tau ----
    rest = seqs[fit_end:]
    cand = [(si, s) for si, (ids, lo) in enumerate(rest) for s in range(lo + args.m, len(ids) - bs)]
    random.Random(1).shuffle(cand)
    cand = sorted(cand[: args.anchors])
    by_seq = {}
    for si, s in cand:
        by_seq.setdefault(si, []).append(s)

    conds = ["real", "removed"] + [f"{kind}_{k}" for k in args.exits for kind in ("mean", "ident", "scaled", "ridge")]
    tau = {c: [] for c in conds}
    stats = {(k, j): {"raw": [], "cen": [], "norm": [], "ridge_raw": [], "ridge_cen": []} for k, j in pairs}

    for n_done, (si, anchors) in enumerate(by_seq.items()):
        ids, lo = rest[si]
        x = ids.to(dev)
        hs = hidden(ids)
        feats = extract_context_feature(hs, lids)                     # [1, T, nT*H]
        for k, j in pairs:
            a, b = hs[k][0, lo:].float(), hs[hidx[j]][0, lo:].float()
            st = stats[(k, j)]
            st["raw"].append(F.cosine_similarity(a, b, dim=-1).cpu())
            st["cen"].append(ccos(a, b, mu_x[k], mu_y[j]).cpu())
            st["norm"].append((a.norm(dim=-1) / b.norm(dim=-1)).cpu())
            r = ridge(k, j, a)
            st["ridge_raw"].append(F.cosine_similarity(r, b, dim=-1).cpu())
            st["ridge_cen"].append(ccos(r, b, mu_y[j], mu_y[j]).cpu())
        truth_all = x.tolist()
        for s in anchors:
            truth = truth_all[s + 1 :]
            m = args.m
            L = blocks_logits(draft, target, feats[:, :s], x, [(s, 0)], bs)[0]
            tau["real"].append(matched(L.argmax(-1).tolist(), truth) + 1)
            L = blocks_logits(draft, target, feats[:, : s - m], x, [(s, m)], bs)[0]
            tau["removed"].append(matched(L.argmax(-1).tolist(), truth) + 1)
            for k in args.exits:
                hk = hs[k][0, s - m : s]
                for kind in ("mean", "ident", "scaled", "ridge"):
                    fz = feats[:, :s].clone()
                    blk = fz[0, s - m : s].view(m, nT, H)
                    for j in missing[k]:
                        if kind == "mean":
                            blk[:, j] = mu_y[j].to(blk.dtype)
                        elif kind == "ident":
                            blk[:, j] = hk
                        elif kind == "scaled":
                            blk[:, j] = (hk.float() * scale[(k, j)]).to(blk.dtype)
                        else:
                            blk[:, j] = ridge(k, j, hk)
                    fz[0, s - m : s] = blk.view(m, nT * H)
                    L = blocks_logits(draft, target, fz, x, [(s, 0)], bs)[0]
                    tau[f"{kind}_{k}"].append(matched(L.argmax(-1).tolist(), truth) + 1)
        if n_done % 20 == 0:
            print(f"[run] {n_done}/{len(by_seq)} sequences, {time.time() - t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    rows1 = []
    for k, j in pairs:
        st = {key: torch.cat(v) for key, v in stats[(k, j)].items()}
        rows1.append({"exit k": k, "missing tap": f"L{lids[j]} (hs{hidx[j]})",
                      "raw cos": st["raw"].mean().item(), "centred cos": st["cen"].mean().item(),
                      "norm ratio": st["norm"].mean().item(),
                      "ridge raw cos": st["ridge_raw"].mean().item(), "ridge centred cos": st["ridge_cen"].mean().item(),
                      "ridge lam": best_lam[(k, j)]})
    print_table(rows1, list(rows1[0]), "1. exit state h_k vs the missing deep taps (held-out response positions)")

    real = mean(tau["real"])
    rem = mean(tau["removed"])
    rows2 = [{"features for the last m positions": c, "tau": mean(tau[c]),
              "share of real": (mean(tau[c]) - rem) / max(real - rem, 1e-9)} for c in conds]
    print_table(rows2, list(rows2[0]), f"2. frozen drafter, m = {args.m} in-flight positions, {len(cand)} anchors "
                                       f"(share of real = (tau - removed) / (real - removed))")
    save_json(args.save, {"args": vars(args), "closeness": rows1, "tau": rows2})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
