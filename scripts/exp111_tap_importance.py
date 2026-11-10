"""Experiment 111: which target layers does the (adapted) DFlash drafter depend on, and where?

The Llama DFlash drafter reads 5 taps: hidden_states[id + 1] for target layers 1 / 8 / 15 / 22 / 29. On held-out
target-greedy text (the train_lag.py validation split, never trained on), the drafter drafts the block at an anchor s
from the real features of [0, s); a tap is then ablated and the drop in accepted length is the tap's importance.

Ablation = replace by the tap's dataset mean (mean ablation; zeros are off-distribution). Scopes:
  all      the tap is ablated at every context position: how much the drafter uses that layer at all
  m8, m32  only at the last m positions before the anchor: how much it needs the layer for the most recent tokens
           (the in-flight situation between verifications; 8 ~ one block, 32 ~ three)
Per tap: drop (only this tap ablated) and keep (only this tap real, all others ablated).
Groups: early1 = only tap L1 real (what a partial pass of k <= 8 layers gives exactly), early2 = taps L1 + L8 real
(k >= 9), deep = only the three deep taps ablated (= early2). Reference rows: real, and removed (the last m positions
have tokens but no features at all, the "features off" lower bound).

Also: each tap's share of the drafter's fused input, ||W_fc[:, tap] x_tap|| / sum over taps, averaged over positions.

tau = matched drafted tokens + 1 (the bonus token), as DFlash reports it.

  python scripts/exp111_tap_importance.py --draft .../drafter_ls_ft/epoch3 --data .../data_ft_clean
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def val_split(data_dir, n_val, max_len, seed):
    """Exactly train_lag.py's validation sequences (same load order, filter and shuffle)."""
    data = []
    for f in sorted(Path(data_dir).glob("shard*.pt")):
        data += torch.load(f)
    data = [d for d in data if len(d["ids"]) <= max_len]
    random.Random(seed).shuffle(data)
    return data[:n_val]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="facebook/layerskip-llama3-8B")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--data", required=True, help="the data dir the drafter was trained on (its val split is used)")
    ap.add_argument("--val-seqs", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchors", type=int, default=1500)
    ap.add_argument("--ms", nargs="+", type=int, default=[8, 32])
    ap.add_argument("--mean-seqs", type=int, default=40, help="sequences used for the per-tap dataset mean")
    ap.add_argument("--save", default="exp111_tap_importance")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    seqs = [(d["ids"].long(), d["n_prompt"]) for d in val_split(args.data, args.val_seqs, args.max_len, args.seed)]
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H, nT = target.config.hidden_size, len(lids)
    M = max(args.ms)
    names = [f"L{l}" for l in lids]

    def feats_of(x):
        out = target(x[None], output_hidden_states=True, use_cache=False)
        return extract_context_feature(out.hidden_states, lids)            # [1, T, nT*H]

    # per-tap dataset mean and fc contribution shares
    fc = next(m for m in draft.modules() if isinstance(m, torch.nn.Linear) and m.in_features == nT * H)
    acc, cnt, contrib = torch.zeros(nT * H, device=dev, dtype=torch.float32), 0, torch.zeros(nT, device=dev)
    for ids, lo in seqs[: args.mean_seqs]:
        f = feats_of(ids.to(dev))[0, lo:].float()
        acc += f.sum(0)
        cnt += f.shape[0]
        for t in range(nT):
            sl = slice(t * H, (t + 1) * H)
            contrib[t] += (f[:, sl].to(fc.weight.dtype) @ fc.weight[:, sl].T).float().norm(dim=-1).sum()
    mean_vec = (acc / cnt).to(torch.bfloat16)
    share = (contrib / contrib.sum()).tolist()

    cand = [(i, s) for i, (ids, lo) in enumerate(seqs) for s in range(lo + M, len(ids) - bs)]
    random.Random(1).shuffle(cand)
    cand = sorted(cand[: args.anchors])
    by_seq = {}
    for i, s in cand:
        by_seq.setdefault(i, []).append(s)

    scopes = ["all"] + [f"m{m}" for m in args.ms]
    conds = [("real", None, "all")] + [("removed", None, f"m{m}") for m in args.ms]
    for sc in scopes:
        conds += [("drop", t, sc) for t in range(nT)] + [("keep", t, sc) for t in range(nT)]
        conds += [("early1", None, sc), ("early2", None, sc)]
    res = {c: [] for c in conds}

    def ablated(feats, s, sc, real_taps):
        fz = feats[:, :s].clone()
        lo = 0 if sc == "all" else s - int(sc[1:])
        blk = fz[0, lo:s].view(-1, nT, H)
        mv = mean_vec.view(nT, H)
        for t in range(nT):
            if t not in real_taps:
                blk[:, t] = mv[t]
        fz[0, lo:s] = blk.view(-1, nT * H)
        return fz

    for n_done, (i, anchors) in enumerate(by_seq.items()):
        ids, lo = seqs[i]
        x = ids.to(dev)
        feats = feats_of(x)
        for s in anchors:
            truth = x[s + 1 :].tolist()
            for cd in conds:
                kind, t, sc = cd
                if kind == "real":
                    L = blocks_logits(draft, target, feats[:, :s], x, [(s, 0)], bs)[0]
                elif kind == "removed":
                    m = int(sc[1:])
                    L = blocks_logits(draft, target, feats[:, : s - m], x, [(s, m)], bs)[0]
                else:
                    real_taps = {"drop": set(range(nT)) - {t}, "keep": {t}, "early1": {0}, "early2": {0, 1}}[kind]
                    L = blocks_logits(draft, target, ablated(feats, s, sc, real_taps), x, [(s, 0)], bs)[0]
                res[cd].append(matched(L.argmax(-1).tolist(), truth) + 1)
        if n_done % 20 == 0:
            print(f"[run] {n_done}/{len(by_seq)} sequences, {len(cand)} anchors, {time.time() - t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    real = mean(res[("real", None, "all")])
    print(f"\nreal features: tau {real:.3f} ({len(cand)} anchors on {len(by_seq)} held-out sequences)")
    print_table([{"tap": names[t], "fc_share": share[t]} for t in range(nT)], ["tap", "fc_share"],
                "share of the drafter's fused input per tap (||W_fc[:, tap] x_tap||, normalised)")
    rows = []
    for t in range(nT):
        r = {"tap": names[t]}
        for sc in scopes:
            r[f"drop_{sc}"] = mean(res[("drop", t, sc)])
            r[f"keep_{sc}"] = mean(res[("keep", t, sc)])
        rows.append(r)
    for kind, label in (("early1", "early: L1 only"), ("early2", "early: L1+L8")):
        r = {"tap": label}
        for sc in scopes:
            r[f"keep_{sc}"] = mean(res[(kind, None, sc)])
        rows.append(r)
    r = {"tap": "removed (no features)"}
    for m in args.ms:
        r[f"keep_m{m}"] = mean(res[("removed", None, f"m{m}")])
    rows.append(r)
    cols = ["tap"] + [f"{k}_{sc}" for sc in scopes for k in ("drop", "keep")]
    print_table(rows, cols, f"tau with taps mean-ablated (real = {real:.3f}); drop = only this tap ablated, "
                            f"keep = only this tap (or group) real; scope all / last m context positions")
    save_json(args.save, {"args": vars(args), "real": real, "fc_share": dict(zip(names, share)),
                          "tau": {f"{k}|{'' if t is None else names[t]}|{sc}": mean(v)
                                  for (k, t, sc), v in res.items()}})


if __name__ == "__main__":
    main()
