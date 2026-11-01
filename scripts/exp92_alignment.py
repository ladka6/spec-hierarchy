"""Experiment 92: is DFlash's decay along the block about CONTENT or about ALIGNMENT? (offline, T=0)

Hypothesis: the drafter often knows which tokens come but not exactly where (an extra / missing token, a different
split of a word or number shifts everything after it). Verification needs an exact prefix, so one shift kills the rest
of the block. DFlash2's selector walks position-locked candidates (top-16 per position) and cannot repair a shift.

Per anchor (target greedy text t = truth, plain DFlash block y, top-k candidates per position):
  Table 1, per block position k:
    pos      y_k = t_k                                   (positional accuracy)
    win1/2   t_k in y_{k-w..k+w}                          (right content, off by <= w positions)
    bag      t_k anywhere in y                            (content somewhere in the block)
    lock16   t_k in top-16 at position k                  (what a position-locked selector can reach, DFlash2)
    win1top3 t_k in the top-3 of positions k-1..k+1       (<= 9 candidates, shift-tolerant)
  Table 2, accepted length per verification (matched prefix + 1) under oracle repairs of the FIRST error e:
    plain          DFlash as is
    lock16 path    longest prefix with every t_k in top-16 at k (DFlash2's oracle ceiling)
    shift fix      one deletion (drop y_e) or one insertion (put t_e before y_e, only if t_e is among the top-16 at e
                   or e+1), then keep matching: repairs alignment only
    sub fix        replace y_e by t_e (one free correction, as a copy would give)
    any fix        best of shift / sub
  Table 3: type of the first error: substitution (rest stays aligned), draft skipped a token, draft has an extra token,
           other.

  python scripts/exp92_alignment.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def first_error_type(y, t, e, look=3):
    """Classify the first mismatch at e by what re-aligns the next `look` tokens."""
    def same(a, b):
        return len(a) >= look and len(b) >= look and a[:look] == b[:look]
    if same(y[e + 1 :], t[e + 1 :]):
        return "substitution"
    if same(y[e:], t[e + 1 :]):
        return "draft skipped a token"
    if same(y[e + 1 :], t[e:]):
        return "draft has an extra token"
    return "other"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="exp92_alignment")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)
    keys = ["pos", "win1", "win2", "bag", "lock16", "win1top3"]
    hits = {k: [0] * D for k in keys}
    acc = {k: [] for k in ["plain", "lock16 path", "shift fix", "sub fix", "any fix"]}
    etype = Counter()
    n_anc = 0
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            xl = x.tolist()
            for s in range(n0, len(x) - bs - 5, args.stride):
                n_anc += 1
                t = xl[s + 1 : s + 1 + D + 4]                     # a few extra tokens for shifted matching
                lg = blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].float()
                top = lg.topk(16, -1).indices.tolist()             # [D][16]
                y = [c[0] for c in top]
                for k in range(D):
                    tk = t[k]
                    hits["pos"][k] += y[k] == tk
                    hits["win1"][k] += tk in y[max(0, k - 1) : k + 2]
                    hits["win2"][k] += tk in y[max(0, k - 2) : k + 3]
                    hits["bag"][k] += tk in y
                    hits["lock16"][k] += tk in top[k]
                    hits["win1top3"][k] += any(tk in top[j][:3] for j in range(max(0, k - 1), min(D, k + 2)))
                m = matched(y, t[:D])
                acc["plain"].append(m + 1)
                n_lock = 0
                while n_lock < D and t[n_lock] in top[n_lock]:
                    n_lock += 1
                acc["lock16 path"].append(n_lock + 1)
                if m >= D:
                    for k in ("shift fix", "sub fix", "any fix"):
                        acc[k].append(m + 1)
                    continue
                e = m
                etype[first_error_type(y, t, e)] += 1
                dele = e + matched(y[e + 1 :], t[e:])                                   # drop y_e
                ins = e + 1 + matched(y[e:], t[e + 1 :]) if (t[e] in top[e] or (e + 1 < D and t[e] in top[e + 1])) \
                    else e                                                              # put t_e before y_e
                sub = e + 1 + matched(y[e + 1 :], t[e + 1 :])                           # replace y_e by t_e
                shift = min(max(dele, ins), D)
                sub = min(sub, D)
                acc["shift fix"].append(max(shift, m) + 1)
                acc["sub fix"].append(sub + 1)
                acc["any fix"].append(max(shift, sub, m) + 1)
            print(f"[{dname}] {n_anc} anchors", flush=True)

    rows = [{"position": k + 1, **{key: hits[key][k] / n_anc for key in keys}} for k in range(D)]
    print_table(rows, ["position"] + keys, f"1. per-position recall ({n_anc} blocks)")
    rows2 = [{"accepted per verification": k, "mean": sum(v) / len(v)} for k, v in acc.items()]
    print_table(rows2, list(rows2[0]), "2. oracle repairs of the first error")
    tot = sum(etype.values())
    rows3 = [{"first error": k, "share": v / tot} for k, v in etype.most_common()]
    print_table(rows3, list(rows3[0]), f"3. type of the first error ({tot} blocks with an error)")
    save_json(args.out, {"args": vars(args), "per_position": rows, "repairs": rows2, "error_types": rows3})


if __name__ == "__main__":
    main()
