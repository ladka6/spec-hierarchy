"""Experiment 66: how closely does a cheap copy follow the target? Per-token greedy agreement q and run lengths.

A chain built by drafter + copy rounds is accepted by the target up to the first token where the copy's greedy choice
differs from the target's. So the accepted length of such a chain is the copy's run length: the number of consecutive
positions, from a start position, where argmax(copy) == argmax(target) on the same (true) prefix. With teacher forcing
on one sequence this is exact for greedy decoding: while the copy agrees, its prefix IS the target's prefix.

For each copy: one forward per held-out sequence, compared with the target's argmax at every response position.
Reported:
  q            share of response positions where the copy's argmax equals the target's
  P(run>=L)    share of start positions whose run reaches L (only starts with >= L positions left: no censoring)
  cond_k       P(agree at offset k | agreed at 1..k-1), k = 1..--max-k (flat = geometric runs, 1/(1-q) mean)
  1/(1-q)      the mean run implied by q if runs were geometric
Copies: any load_mid spec (bnb4, rtnN, hqqN, tao4, lora:<dir> for the trained copy, ...).

  python scripts/exp66_copyagree.py --teacher .../teacher.pt --copies bnb4:Qwen/Qwen3-8B rtn3:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import free, load_mid, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

LENS = (2, 5, 10, 20, 33, 50, 100, 200)


@torch.inference_mode()
def argmaxes(model, seqs, dev, chunk=2048):
    out = []
    for d in seqs:
        x = d["ids"].long().to(dev)
        lo = max(d["n_prompt"] - 1, 0)
        logits = model(x[None, : len(x) - 1], use_cache=False, logits_to_keep=len(x) - 1 - lo).logits[0]
        out.append(logits.argmax(-1).cpu())          # predictions for positions lo+1 .. len-1
        del logits
    return out


def runs(agree):
    """Run length from every start: r[i] = consecutive True from i (censored at the end)."""
    r = torch.zeros(len(agree) + 1, dtype=torch.long)
    for i in range(len(agree) - 1, -1, -1):
        r[i] = r[i + 1] + 1 if agree[i] else 0
    return r[:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--copies", nargs="+", default=["bnb4:Qwen/Qwen3-8B", "rtn3:Qwen/Qwen3-8B"])
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--max-k", type=int, default=64)
    ap.add_argument("--save", default="exp66_copyagree")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    seqs = torch.load(args.teacher)["val"]
    if args.n_val:
        seqs = seqs[: args.n_val]
    target = load_target(args.target, dev)
    tgt = argmaxes(target, seqs, dev)
    # how often the stored sequence is the target's own greedy continuation (exactness of the run lengths)
    data_greedy = sum(int((t == d["ids"][max(d["n_prompt"] - 1, 0) + 1 :].long()).sum()) for t, d in zip(tgt, seqs))
    n_pos = sum(len(t) for t in tgt)
    del target
    free()
    print(f"[exp66] {len(seqs)} sequences, {n_pos} response positions; stored text = target greedy at "
          f"{data_greedy / n_pos:.3f} of them", flush=True)

    summary, conds = [], {}
    for spec in args.copies:
        try:
            copy = load_mid(spec, dev)
        except Exception as e:  # noqa: BLE001
            print(f"[{spec}] failed to load: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        cp = argmaxes(copy, seqs, dev)
        del copy
        free()
        agree_all, reach, starts = [], {L: 0 for L in LENS}, {L: 0 for L in LENS}
        ck_num, ck_den = torch.zeros(args.max_k), torch.zeros(args.max_k)
        for a, b in zip(cp, tgt):
            ag = a == b
            agree_all.append(ag)
            r = runs(ag.tolist())
            left = torch.arange(len(ag), 0, -1)                     # positions left from each start
            for L in LENS:
                ok = left >= L
                starts[L] += int(ok.sum())
                reach[L] += int((r[ok] >= L).sum())
            for k in range(1, args.max_k + 1):                      # cond_k over starts with >= k positions left
                ok = left >= k
                ck_den[k - 1] += int((r[ok] >= k - 1).sum())
                ck_num[k - 1] += int((r[ok] >= k).sum())
        q = float(torch.cat(agree_all).float().mean())
        row = {"copy": spec, "q": q, "mean_run_if_geometric": 1 / max(1 - q, 1e-9)}
        row.update({f"P(run>={L})": (reach[L] / starts[L]) if starts[L] else float("nan") for L in LENS})
        summary.append(row)
        conds[spec] = (ck_num / ck_den.clamp(min=1)).tolist()
        print(row, flush=True)

    if not summary:
        return
    print_table(summary, list(summary[0]), "copy vs target greedy agreement on held-out response positions; "
                                           "P(run>=L) = share of start positions whose run of agreeing tokens "
                                           "reaches L (only starts with at least L positions left)")
    ks = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    crow = [{"copy": s, **{f"k{k}": c[k - 1] for k in ks if k <= args.max_k}} for s, c in conds.items()]
    print_table(crow, list(crow[0]), "cond_k = P(copy agrees at offset k | agreed at offsets 1..k-1)")
    save_json(args.save, {"args": vars(args), "data_greedy_share": data_greedy / n_pos, "summary": summary,
                          "cond": conds})


if __name__ == "__main__":
    main()
