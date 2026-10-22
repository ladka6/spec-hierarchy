"""Experiment 71: how exact must target-like features be? Fidelity -> accepted length, per tap (offline, Colab-ready)

A refresh gives the drafter features for positions it drafted itself; they come from an emulator (copy, ridge map,
trained model) and are only approximately the target's. Here the target's REAL features of the last m context
positions before each anchor (the positions a restart would refresh) are replaced by versions with an exact cosine c
to the real ones (same norm, random direction in the orthogonal complement), one tap at a time or all taps together,
and the DFlash block from the anchor is scored against the target's greedy continuation.

  tau(c, tap, m)  mean accepted length (matched drafted tokens + 1) on the same anchors
Reference rows: real features (c = 1), recent features removed entirely (context ends m positions earlier, the
"none" source), and copy features for those positions (if --copy is given).

  python scripts/exp71_fidelity.py --cache /content/seqs.pt
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

from hspec.models import load_draft, load_mid, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

COS = (0.999, 0.99, 0.97, 0.95, 0.9, 0.8, 0.6, 0.3)


def at_cosine(f, c, gen):
    """Vectors with cosine exactly c to f (row-wise) and the same norm."""
    u = F.normalize(f.float(), dim=-1)
    r = torch.randn(f.shape, generator=gen, device=f.device)
    r = F.normalize(r - (r * u).sum(-1, keepdim=True) * u, dim=-1)
    out = (c * u + (1 - c * c) ** 0.5 * r) * f.float().norm(dim=-1, keepdim=True)
    return out.to(f.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B", help="'' to skip the copy reference")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences, e.g. from exp69")
    ap.add_argument("--anchors", type=int, default=1500)
    ap.add_argument("--m", nargs="+", type=int, default=[8])
    ap.add_argument("--m-all", nargs="+", type=int, default=[4, 15], help="extra m values, all taps only")
    ap.add_argument("--save", default="exp71_fidelity")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(0)

    seqs = torch.load(args.cache)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    copy = load_mid(args.copy, dev) if args.copy else None
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H = target.config.hidden_size
    nT = len(lids)
    M = max(args.m + args.m_all)

    # sample anchors (sequence, position) with at least M context positions in the response and a full block after
    cand = [(i, s) for i, (ids, lo) in enumerate(seqs) for s in range(lo + M, len(ids) - bs)]
    random.Random(0).shuffle(cand)
    cand = sorted(cand[: args.anchors])
    by_seq = {}
    for i, s in cand:
        by_seq.setdefault(i, []).append(s)

    conds = [("real", None, 1.0, m) for m in args.m + args.m_all]
    conds += [("removed", None, 0.0, m) for m in args.m + args.m_all]
    if copy is not None:
        conds += [("copy", None, 0.0, m) for m in args.m + args.m_all]
    for m in args.m:
        for tap in list(range(nT)) + ["all"]:
            for c in COS:
                conds.append(("cos", tap, c, m))
    for m in args.m_all:
        for c in COS:
            conds.append(("cos", "all", c, m))
    res = {cd: [] for cd in conds}

    with torch.inference_mode():
        for n_done, (i, anchors) in enumerate(by_seq.items()):
            ids, lo = seqs[i]
            x = ids.to(dev)
            out = target(x[None], output_hidden_states=True, use_cache=False)
            feats = extract_context_feature(out.hidden_states, lids)              # [1, T, 5H]
            del out
            cf = None
            if copy is not None:
                co = copy(x[None], output_hidden_states=True, use_cache=False)
                cf = extract_context_feature(co.hidden_states, lids).to(feats.dtype)
                del co
            for s in anchors:
                truth = x[s + 1 :].tolist()
                for cd in conds:
                    kind, tap, c, m = cd
                    if kind == "removed":
                        L = blocks_logits(draft, target, feats[:, : s - m], x, [(s, m)], bs)[0]
                    else:
                        fz = feats[:, :s].clone()
                        if kind == "copy":
                            fz[:, s - m : s] = cf[:, s - m : s]
                        elif kind == "cos":
                            blk = fz[0, s - m : s].view(m, nT, H)
                            taps = range(nT) if tap == "all" else [tap]
                            for t in taps:
                                blk[:, t] = at_cosine(blk[:, t], c, gen)
                            fz[0, s - m : s] = blk.view(m, nT * H)
                        L = blocks_logits(draft, target, fz, x, [(s, 0)], bs)[0]
                    res[cd].append(matched(L.argmax(-1).tolist(), truth) + 1)
            if n_done % 20 == 0:
                print(f"[run] {n_done}/{len(by_seq)} sequences, {sum(len(v) for v in by_seq.values())} anchors total, "
                      f"{time.time() - t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    for m in sorted(set(args.m + args.m_all)):
        ref = [{"source": k, "tau": mean(res[(k, None, c, m)])} for k, c in
               (("real", 1.0), ("removed", 0.0)) + ((("copy", 0.0),) if copy is not None else ())]
        print_table(ref, ["source", "tau"], f"m = {m} refreshed positions: reference sources")
        taps = list(range(nT)) + ["all"] if m in args.m else ["all"]
        rows = []
        for tap in taps:
            rows.append({"tap": f"L{lids[tap]}" if tap != "all" else "all",
                         **{f"cos{c}": mean(res[("cos", tap, c, m)]) for c in COS}})
        print_table(rows, list(rows[0]), f"m = {m}: tau when the last {m} context positions get features at cosine c "
                                         f"to the real ones (one tap, or all taps)")
    save_json(args.save, {"args": vars(args), "n_anchors": len(cand),
                          "results": {f"{k}|{t}|{c}|{m}": mean(v) for (k, t, c, m), v in res.items()}})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
