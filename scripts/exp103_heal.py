"""Experiment 103: does a DFlash block heal after a short repair at its fork? (offline, T=0, no training)

Question: when DFlash's block breaks at its first error e, is the rest of the block still usable after replacing only
a few tokens around e? If yes, a method can repair a window and keep the drafted tail instead of discarding it.
exp96 measured fixed-width oracle windows (11.5 accepted for one window of 4). exp92 measured single-token shifts
(sub fix 9.56, shift fix 9.05; 83% of first errors are neither a clean substitution nor a single shift). This
experiment asks the next questions:
  1. windows of width w = 1..6 whose true replacement may be LONGER or SHORTER than w (length change s = -2..+2), so
     the tail after the window is re-attached shifted by s
  2. how often the WHOLE rest of the block is right after one repair
  3. where the tail is not right: is the true token at least in DFlash's top-k at that tail position (or at the
     neighbouring position, i.e. a misalignment)?

Per block (target greedy text t = truth, plain DFlash draft y, DFlash top-16 per position, first error e):
  repaired block  r = y[:e] + t[e : e + w + s] + y[e + w :], cut to the block length D (oracle window content)
  Table 1  per (w, s): mean accepted (matched prefix + 1, max D + 1), share of blocks fully accepted, mean run of the
           re-attached tail beyond the window
  Table 2  per w: best s per block (what a verify tree with shifted tails could reach), share of blocks where s != 0
           is strictly better than s = 0
  Table 3  tail with top-k: after the repair (s = 0 and best s), leading run where every tail token of the truth is
           in DFlash's top-k at the tail token's ORIGINAL position (k = 1, 2, 4, 8, 16): what selecting among DFlash's
           candidates after the window could add
  Table 4  wrong tail tokens (w = 4, s = 0, positions after the window): rank of the true token in DFlash's
           distribution at that position, and the best rank over the neighbouring positions (j-1, j, j+1)
Only blocks with an error inside the block are used for tables 1-4; blocks without error are counted.

  python scripts/exp103_heal.py
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
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

KS = (1, 2, 4, 8, 16)


def repaired(y, t, e, w, s, D):
    """Window y[e:e+w] replaced by the true segment t[e:e+w+s]; tail y[e+w:] re-attached; cut to D."""
    seg = t[e : e + w + s] if w + s > 0 else []
    return (y[:e] + seg + y[e + w :])[:D]


def topk_run(t, top, e, w, s, D, k):
    """Accepted length when prefix and window are right and each re-attached tail token may be any of DFlash's top-k
    at its ORIGINAL position (tail token from original index i sits at new index i + s)."""
    n = min(e + w + s, D)                      # prefix + window are right by construction
    if w + s < 0:
        n = e
    j = n
    while j < D:
        i = j - s                              # original position of the tail token now at j
        if i < e + w or i >= D or t[j] not in top[i][:k]:
            break
        j += 1
    return j + 1


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--shifts", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
    ap.add_argument("--out", default="exp103_heal")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)

    blocks = []                                # (y, t, top16 per position, ranks-of-truth helper logits)
    n_all = n_clean = 0
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            xl = x.tolist()
            for s0 in range(ids.shape[1], len(x) - bs - 1, args.stride):
                lg = blocks_logits(draft, target, feats, x, [(s0, 0)], bs)[0].float()
                t = xl[s0 + 1 : s0 + 1 + D]
                y = lg.argmax(-1).tolist()
                n_all += 1
                e = matched(y, t)
                if e >= D:
                    n_clean += 1
                    continue
                top = lg.topk(16, -1).indices.tolist()
                # rank of the true token at every position (1 = argmax), capped at 17 = "beyond top-16"
                tr = torch.tensor(t, device=lg.device)
                tl = lg.gather(1, tr[:, None])
                rank = (lg > tl).sum(-1).add(1).clamp(max=17).tolist()
                blocks.append((y, t, top, rank, e))
        print(f"[{dname}] {n_all} blocks, {len(blocks)} with an error", flush=True)

    plain = mean(b[4] + 1 for b in blocks)
    print(f"plain DFlash on blocks with an error: {plain:.3f}  ({n_clean} of {n_all} blocks had no error)", flush=True)

    # Table 1 and 2
    acc = defaultdict(list)
    full = defaultdict(list)
    tail = defaultdict(list)
    best = defaultdict(list)
    best_shift_helps = defaultdict(list)
    best_s_of = {}
    for bi, (y, t, top, rank, e) in enumerate(blocks):
        for w in args.widths:
            per_s = {}
            for s in args.shifts:
                if w + s < 0:
                    continue
                r = repaired(y, t, e, w, s, D)
                a = matched(r, t) + 1
                per_s[s] = a
                acc[(w, s)].append(a)
                full[(w, s)].append(a == D + 1)
                tail[(w, s)].append(max(0, a - 1 - min(e + w + s, D)))
            bs_ = max(per_s, key=lambda s: (per_s[s], -abs(s)))
            best[w].append(per_s[bs_])
            best_shift_helps[w].append(per_s[bs_] > per_s.get(0, 0))
            best_s_of[(bi, w)] = bs_
    rows1 = []
    for w in args.widths:
        for s in args.shifts:
            if (w, s) in acc:
                rows1.append({"window w": w, "length change s": s, "accepted": mean(acc[(w, s)]),
                              "whole block right": mean(full[(w, s)]), "tail run after window": mean(tail[(w, s)])})
    rows2 = [{"window w": w, "accepted, s = 0": mean(acc[(w, 0)]), "accepted, best s": mean(best[w]),
              "share where s != 0 is better": mean(best_shift_helps[w])} for w in args.widths]

    # Table 3: tail tokens allowed anywhere in DFlash's top-k
    rows3 = []
    for w in args.widths:
        row = {"window w": w}
        for k in KS:
            row[f"top{k}, s=0"] = mean(topk_run(t, top, e, w, 0, D, k) for (y, t, top, rank, e) in blocks)
        row["top4, best s"] = mean(topk_run(t, top, e, w, best_s_of[(bi, w)], D, 4)
                                   for bi, (y, t, top, rank, e) in enumerate(blocks))
        rows3.append(row)

    # Table 4: where are the wrong tail tokens (w = 4, s = 0)?
    hist = defaultdict(int)
    hist_nb = defaultdict(int)
    n_wrong = 0
    for y, t, top, rank, e in blocks:
        r = repaired(y, t, e, 4, 0, D)
        for j in range(min(e + 4, D), D):
            if r[j] == t[j]:
                continue
            n_wrong += 1
            rk = rank[j]
            hist[rk] += 1
            # neighbour positions: is t[j] ranked better as a candidate of position j-1 or j+1? (misalignment)
            nb = rk
            for jj in (j - 1, j + 1):
                if e + 4 <= jj < D and t[j] in top[jj]:
                    nb = min(nb, top[jj].index(t[j]) + 1)
            hist_nb[nb] += 1

    def bucket(h):
        out = {}
        for name, lo, hi in (("rank 1", 1, 1), ("rank 2", 2, 2), ("rank 3-4", 3, 4), ("rank 5-8", 5, 8),
                             ("rank 9-16", 9, 16), ("beyond 16", 17, 17)):
            out[name] = sum(v for k_, v in h.items() if lo <= k_ <= hi) / max(n_wrong, 1)
        return out

    rows4 = [{"where": "same position", **bucket(hist)},
             {"where": "best of positions j-1, j, j+1", **bucket(hist_nb)}]

    cols1 = ["window w", "length change s", "accepted", "whole block right", "tail run after window"]
    print_table(rows1, cols1, "1. oracle window repair at the first error, drafted tail re-attached (shift s)")
    print_table(rows2, list(rows2[0]), "2. best length change per block (what shifted tails in a verify tree reach)")
    print_table(rows3, list(rows3[0]), "3. after the window: tail tokens may be any of DFlash's top-k at their position")
    print_table(rows4, list(rows4[0]), f"4. wrong tail tokens after a window of 4 ({n_wrong}): rank of the true token")
    save_json(args.out, {"args": vars(args), "blocks": n_all, "blocks_without_error": n_clean,
                         "blocks_with_error": len(blocks), "plain_error_blocks": plain,
                         "repair": rows1, "best_shift": rows2, "tail_topk": rows3, "wrong_tail_rank": rows4})


if __name__ == "__main__":
    main()
