"""Experiment 89: hedged multi-round spines for a block-diffusion drafter with a target copy (offline, T=0).

Question: in a hierarchy (block drafter -> cheap copy of the target -> target), does it pay to build a long spine over
several drafter+copy rounds before one target pass, and to hedge the spine at the positions where it is most likely
wrong? Which signal should pick the hedge points: the copy's margin (calibrated about the target's near-ties) or the
drafter's own confidence (SpecBranch-style)?

Per anchor s (target greedy text as ground truth):
  round r = 1..R   the drafter drafts 15 tokens at the end of the spine, conditioned on real target features up to s and
                   the copy's features after s; the copy runs over them, the spine keeps the drafted tokens up to the
                   copy's first disagreement plus the copy's own token there (16 tokens if no disagreement).
  hedges           K spine positions with the lowest signal get one alternative token (copy top-2, or the drafter's
                   best other token) and a 15-token drafter continuation from it (copy features up to the branch point).
                   All continuations of a round can share one batched drafter pass (cost: one drafter pass).
  target           one pass over the tree (spine + branches); accepted = longest match with the target's text + 1.
Cost model (ms, batch 1, measured vLLM numbers): verify t_V(nodes) interpolated from 1:12.7, 16:13.3, 32:14.1, 64:14.2,
128:15.8, 256:19.0; drafter --t-draft; copy --t-copy. Plain DFlash: one drafter pass + t_V(16).
Also: calibration of both signals about the target (AUROC for "target disagrees at this spine position", on spine
positions whose prefix still matches the target).

  python scripts/exp89_hedge.py --copy rtn4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched, tree_nodes  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

TV = [(1, 12.7), (16, 13.3), (32, 14.1), (64, 14.2), (128, 15.8), (256, 19.0)]


def t_verify(n):
    for (a, ta), (b, tb) in zip(TV[:-1], TV[1:]):
        if n <= b:
            return ta + (tb - ta) * max(n - a, 0) / (b - a)
    return TV[-1][1] * n / TV[-1][0]


def auroc(scores, labels):
    """AUROC of low score predicting label 1."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank, r_pos, i = 0.0, 0.0, 0
    while i < len(allv):                                  # average ranks over ties
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        r_pos += avg * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    u = r_pos - len(pos) * (len(pos) + 1) / 2             # ranks of positives (ascending score)
    return 1.0 - u / (len(pos) * len(neg))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="rtn4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--t-draft", type=float, default=2.6)
    ap.add_argument("--t-copy", type=float, default=4.5)
    ap.add_argument("--out", default="exp89_hedge")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    copy = load_mid(args.copy)
    stops = stop_ids(target, tok)
    bs, D, R = draft.block_size, draft.block_size - 1, args.rounds
    lids = list(draft.target_layer_ids)
    dev = next(target.parameters()).device

    rec = defaultdict(list)            # (r, policy, K) -> [(accepted, nodes)]
    base = []
    cal = {"copy": ([], []), "drafter": ([], []), "top2_cover": []}
    n_anchor = 0
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            horizon = R * bs + bs
            for s in range(n0, len(x) - horizon - 1, args.stride):
                n_anchor += 1
                truth = x[s + 1 : s + 1 + horizon].tolist()
                P, info = [], []            # spine tokens after s; per token: copy margin, copy alt, drafter p, alt
                cf = None                   # copy features [1, T, W] of the latest copy run
                spine_at = []               # spine length after each round
                for r in range(R):
                    a = s + len(P)
                    z = torch.cat([x[: s + 1], torch.tensor(P, dtype=x.dtype, device=dev)])
                    mix = feats[:, :s] if r == 0 else torch.cat([feats[:, :s], cf[:, s:a]], 1)
                    lg = blocks_logits(draft, target, mix, z, [(a, 0)], bs)[0].float()     # [15, V]
                    pr = torch.softmax(lg, -1)
                    top2d = pr.topk(2, -1)
                    y = top2d.indices[:, 0].tolist()
                    if r == 0:
                        base.append(matched(y, truth) + 1)
                    seq = torch.cat([z, torch.tensor(y, dtype=x.dtype, device=dev)])
                    co = copy(seq[None], output_hidden_states=True, logits_to_keep=bs)
                    cf = extract_context_feature(co.hidden_states, lids).to(feats.dtype)
                    cp = torch.softmax(co.logits[0].float(), -1)                          # [16, V]: predicts y_0..y_14, next
                    top2c = cp.topk(2, -1)
                    c1, c2 = top2c.indices[:, 0].tolist(), top2c.indices[:, 1].tolist()
                    marg = (top2c.values[:, 0] - top2c.values[:, 1]).tolist()
                    j = next((k for k in range(D) if c1[k] != y[k]), D)
                    new = y[:j] + [c1[j]]
                    for k, t in enumerate(new):
                        if k < D:
                            pd = float(pr[k, t])
                            d_alt = y[k] if t != y[k] else int(top2d.indices[k, 1])
                        else:
                            pd, d_alt = 1.0, None                 # copy's bonus token: no drafter information
                        info.append({"cm": marg[k], "c_alt": c2[k], "dp": pd, "d_alt": d_alt})
                    P += new
                    spine_at.append(len(P))
                # calibration on spine positions whose prefix still matches the target's text
                m = matched(P, truth)
                for k in range(min(m + 1, len(P))):
                    wrong = P[k] != truth[k]
                    cal["copy"][0].append(info[k]["cm"])
                    cal["copy"][1].append(wrong)
                    cal["drafter"][0].append(info[k]["dp"])
                    cal["drafter"][1].append(wrong)
                    if wrong:
                        cal["top2_cover"].append(info[k]["c_alt"] == truth[k])
                # hedge continuations (cached per spine index and alternative)
                cache = {}

                def branch(i, alt):
                    if (i, alt) not in cache:
                        zb = torch.cat([x[: s + 1], torch.tensor(P[:i] + [alt], dtype=x.dtype, device=dev)])
                        mixb = torch.cat([feats[:, :s], cf[:, s : s + i + 1]], 1)
                        cont = blocks_logits(draft, target, mixb, zb, [(s + i + 1, 0)], bs)[0].argmax(-1).tolist()
                        cache[(i, alt)] = P[:i] + [alt] + cont
                    return cache[(i, alt)]

                for r in range(R):
                    L = spine_at[r]
                    spine = P[:L]
                    for pol, key, altk in (("copy", "cm", "c_alt"), ("drafter", "dp", "d_alt")):
                        cand = sorted((k for k in range(L) if info[k][altk] is not None), key=lambda k: info[k][key])
                        for K in args.ks:
                            if K == 0 and pol == "drafter":
                                continue
                            chains = [spine] + [branch(k, info[k][altk]) for k in cand[:K]]
                            acc = max(matched(c, truth) for c in chains) + 1
                            rec[(r + 1, pol if K else "none", K)].append((acc, tree_nodes(chains)))
            print(f"[{dname}] {n_anchor} anchors so far", flush=True)

    t_base = t_verify(bs) + args.t_draft
    tau0 = sum(base) / len(base)
    rate0 = tau0 / t_base
    rows = [{"rounds": 0, "hedge signal": "plain DFlash", "K": 0, "accepted": tau0, "nodes": bs,
             "time ms": t_base, "tokens/ms": rate0, "vs DFlash": 1.0}]
    for (r, pol, K), v in sorted(rec.items()):
        acc = sum(a for a, _ in v) / len(v)
        nodes = sum(n for _, n in v) / len(v)
        t = t_verify(nodes) + r * (args.t_draft + args.t_copy) + (args.t_draft if K else 0.0)
        rows.append({"rounds": r, "hedge signal": pol, "K": K, "accepted": acc, "nodes": nodes, "time ms": t,
                     "tokens/ms": acc / t, "vs DFlash": acc / t / rate0})
    print_table(rows, list(rows[0]), f"1. hedged spines ({n_anchor} anchors, copy {args.copy}, t_copy {args.t_copy} ms)")
    best = max(rows[1:], key=lambda r: r["tokens/ms"])
    print(f"[best] {best}", flush=True)

    calib = {"AUROC copy margin": auroc(*cal["copy"]), "AUROC drafter prob": auroc(*cal["drafter"]),
             "spine positions": len(cal["copy"][0]),
             "disagreement rate": sum(cal["copy"][1]) / max(len(cal["copy"][1]), 1),
             "target token = copy top-2 when wrong": sum(cal["top2_cover"]) / max(len(cal["top2_cover"]), 1)}
    print_table([calib], list(calib), "2. which signal knows where the spine leaves the target's path")
    save_json(args.out, {"args": vars(args), "rows": rows, "calibration": calib})


if __name__ == "__main__":
    main()
