"""Experiment 31: what computation is the block drafter missing? Retrieval vs depth (offline, T=0).

exp26: real target features over the drafted prefix fix ~80% of DFlash's stops, tokens alone ~49%,
and 38% of stops are at positions where the target is nearly certain. Hypothesis: many of those
"easy for the target, impossible for the drafter" tokens are in-context retrieval (names, numbers,
identifiers, phrases that already occurred), which needs long-range attention through depth.

For every stop of the DFlash chain (first wrong drafted token, at position p, true token x[p]):

  fresh / tokens   does a redraft from the true prefix fix it (real features up to p-1 / tokens only,
                   the latter with the lag-trained drafter if given), and how many tokens it then accepts
  copyable@n       oracle: some earlier occurrence of the last n tokens x[p-n:p] in the context is
                   followed by x[p] (the token is retrievable by a suffix match of length >= n)
  pointer          the prompt-lookup rule (most recent occurrence of the longest suffix match, n >= 2,
                   up to 8) predicts x[p]; copy_run = how many tokens it then gets right (cap 16)
  in_context       x[p] occurs anywhere before p (weak upper bound)

Key number: among stops that need target computation (fresh fixes, tokens does not), the share a
suffix-match pointer recovers. High -> the missing computation is mostly retrieval and a retrieval
channel can replace target features there; low -> it is genuine depth.

  python scripts/exp31_retrieval.py --lag-draft /scratch-shared/$USER/hspec_lagft/drafter_lag16
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
from exp26_lossdiag import cond_logits, ebin  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

NMAX = 8


class Lookup:
    """Earlier occurrences of n-grams (n = 1 .. NMAX) in a token list: (n-gram) -> end indices e
    (the n-gram is xs[e-n:e], followed by xs[e])."""

    def __init__(self, xs):
        self.xs = xs
        self.idx = [None] + [defaultdict(list) for _ in range(NMAX)]
        for n in range(1, NMAX + 1):
            for e in range(n, len(xs)):
                self.idx[n][tuple(xs[e - n : e])].append(e)

    def ends(self, p, n):
        """End indices e < p of earlier occurrences of xs[p-n:p] (excluding the suffix itself)."""
        if p - n < 0:
            return []
        return [e for e in self.idx[n].get(tuple(self.xs[p - n : p]), []) if e < p]

    def copyable(self, p, n):
        t = self.xs[p]
        return any(self.xs[e] == t for m in range(n, NMAX + 1) for e in self.ends(p, m))

    def pointer(self, p, nmin=2):
        """Prompt-lookup prediction at p: most recent occurrence of the longest suffix (>= nmin).
        Returns (predicted token, source end index) or (None, None)."""
        for n in range(NMAX, nmin - 1, -1):
            es = self.ends(p, n)
            if es:
                return self.xs[es[-1]], es[-1]
        return None, None

    def run(self, p, e, cap=16):
        k = 0
        while k < cap and p + k < len(self.xs) and self.xs[e + k] == self.xs[p + k]:
            k += 1
        return k


def lead(hits):
    """Number of leading True values."""
    n = 0
    while n < len(hits) and hits[n]:
        n += 1
    return n


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--lag-draft", default=None)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="exp31_retrieval")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    lagd = load_draft(args.lag_draft) if args.lag_draft else None
    stops = stop_ids(target, tok)
    bs = draft.block_size
    D = bs - 1
    recs = []
    for d in args.datasets:
        for pr in load_prompts(d, args.n):
            ids = encode(tok, pr)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            out = target(x[None], output_hidden_states=True)
            feats = extract_context_feature(out.hidden_states, draft.target_layer_ids)
            lpt = torch.log_softmax(out.logits[0].float(), -1)
            ent = (-(lpt.exp() * lpt).nan_to_num(0.0).sum(-1)).cpu().tolist()   # ent[p-1]: entropy of x[p]
            del out, lpt
            xs = x.tolist()
            lk = Lookup(xs)
            for s in range(n0, len(x) - bs - 1, args.stride):
                truth = x[s + 1 : s + bs]
                lg = draft_logits_lagged(draft, target, feats, x, s, 0, bs)
                k = lead((lg.argmax(-1) == truth).tolist()) + 1          # 1-based offset of the first miss
                if k > D:
                    continue
                p = s + k
                rec = {"ds": d, "k": k, "ent": ent[p - 1], "copy": {n: lk.copyable(p, n) for n in (1, 2, 3, 4)},
                       "in_context": xs[p] in set(xs[:p])}
                pt, e = lk.pointer(p, 2)
                rec["pointer"] = pt == xs[p]
                rec["copy_run"] = lk.run(p, e) if rec["pointer"] else 0
                if k >= 2:                       # redraft re-anchored at p-1 with the true prefix
                    for m, dm in (("fresh", draft), ("tokens", lagd)):
                        if dm is None:
                            continue
                        L = cond_logits(dm, target, feats, x, s, bs, m)[k - 2]
                        t = x[p : s + bs]
                        r = (L[: len(t)].argmax(-1) == t).tolist()
                        rec[m] = bool(r[0])
                        rec[m + "_run"] = lead(r)
                recs.append(rec)
            print(f"[{d}] {len(recs)} stops so far", flush=True)

    def summarize(rs, label):
        two = [r for r in rs if "fresh" in r]
        return {"group": label, "stops": len(rs), "share": len(rs) / max(len(recs), 1),
                "fresh_fix": mean(r["fresh"] for r in two), "tokens_fix": mean(r["tokens"] for r in two if "tokens" in r),
                "copy@1": mean(r["copy"][1] for r in rs), "copy@2": mean(r["copy"][2] for r in rs),
                "copy@3": mean(r["copy"][3] for r in rs), "pointer": mean(r["pointer"] for r in rs),
                "copy_run": mean(r["copy_run"] for r in rs if r["pointer"]),
                "fresh_run": mean(r["fresh_run"] for r in two), "in_ctx": mean(r["in_context"] for r in rs)}

    cols = ["group", "stops", "share", "fresh_fix", "tokens_fix", "copy@1", "copy@2", "copy@3", "pointer",
            "copy_run", "fresh_run", "in_ctx"]
    rows = [summarize(recs, "all")] + [summarize([r for r in recs if r["ds"] == d], d) for d in args.datasets]
    print_table(rows, cols, "DFlash chain stops (T=0): fixes by redraft vs retrievability from context")

    have_tok = any("tokens" in r for r in recs)
    cat = []
    two = [r for r in recs if "fresh" in r]
    if have_tok:
        groups = {"needs computation (fresh fixes, tokens not)": [r for r in two if r["fresh"] and not r["tokens"]],
                  "tokens enough": [r for r in two if r["tokens"]],
                  "nothing fixes": [r for r in two if not r["fresh"] and not r["tokens"]]}
    else:
        groups = {"fresh fixes": [r for r in two if r["fresh"]], "fresh does not": [r for r in two if not r["fresh"]]}
    for g, rs in groups.items():
        cat.append(summarize(rs, g))
    print_table(cat, cols, "stops by what fixes them: how many are retrievable")

    erows = []
    for eb in sorted({ebin(r["ent"]) for r in recs}):
        erows.append(summarize([r for r in recs if ebin(r["ent"]) == eb], f"ent {eb}"))
    print_table(erows, cols, "stops by the target's entropy at the stop")
    save_json(args.out, {"args": vars(args), "overall": rows, "by_fix": cat, "by_entropy": erows,
                         "n_stops": len(recs)})


if __name__ == "__main__":
    main()
