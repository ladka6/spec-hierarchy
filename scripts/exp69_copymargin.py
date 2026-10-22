"""Experiment 69: where does the cheap copy disagree with the target, and does it know? (offline, Colab-ready)

exp68: the copy's error that flips its greedy token is position-specific (near-ties), not a smooth bias. If so, the
copy's own top-1 vs top-2 margin should flag its disagreements, and the target's token should usually be the copy's
second choice there. Then a copy chain that branches ONLY at low-margin positions (copy top-2) keeps the true path
much more often, at few extra nodes.

On greedy target responses with teacher forcing (exact for greedy: while the true path is in the tree, the copy's
prefix is the target's prefix):
  1. per copy-margin bin: share of positions, disagreement rate, P(target token in copy top-2 / top-4 | disagree),
     and the target's own margin there (are these near-ties for the target too?)
  2. AUROC of the copy margin for predicting a disagreement; share of disagreements below margin 0.1 / 0.3
  3. branch-at-low-margin policy, theta = margin threshold, k = candidates at a branch point:
       q_eff     share of positions where the target's token is in the candidate set (top-1, or top-k if margin < theta)
       branches  branch points per 100 tokens
       P(run>=L) share of start positions whose true path survives L tokens (upper bound: every branch continued)

  python scripts/exp69_copymargin.py --n 150 --cache /content/seqs.pt
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

from exp67_probe import generate  # noqa: E402

from hspec.data import load_prompts  # noqa: E402
from hspec.models import load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

BINS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001]
LENS = (5, 10, 20, 33, 50, 100)


def runs(ok):
    r = torch.zeros(len(ok) + 1, dtype=torch.long)
    for i in range(len(ok) - 1, -1, -1):
        r[i] = r[i + 1] + 1 if ok[i] else 0
    return r[:-1]


def auroc(score, label):
    """P(score of a positive < score of a negative): positives = disagreements, low margin should flag them."""
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    # AUROC of (-score) for label: low margin = positive
    r_pos = ranks[label].sum()
    return float(1 - (r_pos - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--cache", default="", help="save / reuse the generated sequences here (.pt)")
    ap.add_argument("--save", default="exp69_copymargin")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    tok = load_tokenizer(args.target)
    target = load_target(args.target, dev)
    if args.cache and Path(args.cache).exists():
        seqs = torch.load(args.cache)
        print(f"[data] {len(seqs)} sequences from {args.cache}", flush=True)
    else:
        prompts = []
        for ds in args.datasets:
            try:
                prompts += load_prompts(ds, args.n)
            except Exception as e:  # noqa: BLE001
                print(f"[data] {ds} skipped: {type(e).__name__}: {str(e)[:120]}", flush=True)
        seqs = generate(target, tok, prompts, args.max_new, args.gen_batch, dev)
        random.Random(0).shuffle(seqs)
        if args.cache:
            torch.save(seqs, args.cache)
    copy = load_mid(args.copy, dev)

    @torch.inference_mode()
    def top(model, x, lo):
        lg = model(x[None], use_cache=False, logits_to_keep=len(x) - lo + 1).logits[0, :-1].float()   # predicts lo..T-1
        p = torch.softmax(lg, -1)
        v, i = p.topk(4, -1)
        return v, i

    cm, tm, dis, in2, in4, top1 = [], [], [], [], [], []
    seq_bounds = []
    with torch.inference_mode():
        for si, (ids, lo) in enumerate(seqs):
            x = ids.to(dev)
            if len(x) - lo < 8:
                continue
            tv, ti = top(target, x, lo)
            cv, ci = top(copy, x, lo)
            t1 = ti[:, 0]
            cm.append((cv[:, 0] - cv[:, 1]).cpu())
            tm.append((tv[:, 0] - tv[:, 1]).cpu())
            dis.append((ci[:, 0] != t1).cpu())
            in2.append((ci[:, :2] == t1[:, None]).any(-1).cpu())
            in4.append((ci == t1[:, None]).any(-1).cpu())
            seq_bounds.append(len(t1))
            if si % 50 == 0:
                print(f"[run] {si}/{len(seqs)} sequences, {time.time() - t0:.0f}s", flush=True)
    CM, TM, DIS, IN2, IN4 = map(torch.cat, (cm, tm, dis, in2, in4))
    n = len(CM)
    rows = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = (CM >= a) & (CM < b)
        k = int(m.sum())
        if k == 0:
            continue
        d = m & DIS
        rows.append({"copy_margin": f"{a:.2f}-{min(b, 1):.2f}", "share": k / n, "disagree": float(DIS[m].float().mean()),
                     "share_of_all_disagreements": int(d.sum()) / max(int(DIS.sum()), 1),
                     "tgt_in_copy_top2|dis": float(IN2[d].float().mean()) if d.any() else float("nan"),
                     "tgt_in_copy_top4|dis": float(IN4[d].float().mean()) if d.any() else float("nan"),
                     "target_margin": float(TM[m].mean())})
    print_table(rows, list(rows[0]), f"copy ({args.copy}) vs target greedy token, by the copy's own top1-top2 margin "
                                     f"({n} positions)")
    summ = {"q": 1 - float(DIS.float().mean()), "tgt_in_copy_top2": float(IN2.float().mean()),
            "tgt_in_copy_top4": float(IN4.float().mean()), "auroc_margin_flags_disagreement": auroc(CM, DIS),
            "disagreements_with_margin<0.1": float((CM[DIS] < 0.1).float().mean()),
            "disagreements_with_margin<0.3": float((CM[DIS] < 0.3).float().mean()),
            "target_margin_at_disagreements": float(TM[DIS].mean()),
            "target_margin_overall": float(TM.mean())}
    print_table([summ], list(summ), "summary")

    pol = []
    for theta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        for kk, IN in ((2, IN2), (4, IN4)):
            if theta == 0.0 and kk == 4:
                continue
            branch = CM < theta
            ok = torch.where(branch, IN, ~DIS)
            reach, starts = {L: 0 for L in LENS}, {L: 0 for L in LENS}
            off = 0
            for L_seq in seq_bounds:
                o = ok[off : off + L_seq]
                r = runs(o.tolist())
                left = torch.arange(L_seq, 0, -1)
                for L in LENS:
                    sel = left >= L
                    starts[L] += int(sel.sum())
                    reach[L] += int((r[sel] >= L).sum())
                off += L_seq
            pol.append({"theta": theta, "k": kk if theta > 0 else 1, "q_eff": float(ok.float().mean()),
                        "branches_per_100": 100 * float(branch.float().mean()),
                        **{f"P(run>={L})": reach[L] / max(starts[L], 1) for L in LENS}})
    print_table(pol, list(pol[0]), "branch only where the copy's margin < theta, with its top-k; q_eff = target token "
                                   "in the candidate set; runs are an upper bound (every branch continued)")
    save_json(args.save, {"args": vars(args), "bins": rows, "summary": summ, "policy": pol})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
