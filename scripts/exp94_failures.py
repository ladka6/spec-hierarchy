"""Experiment 94: why does DFlash fail where it fails? Failure dump + causal decomposition (offline, T=0).

For every block (target greedy text = truth, plain DFlash with real target features), the first wrong position e
(0-based; the draft matched x[s+1 .. s+e] and missed x[s+e+1]) is tagged with what would have fixed it:
  intrinsic     the target itself is near a tie there (target top-1 minus top-2 probability < --tie)
  computation   a fresh DFlash block at the last correct token (real target features of the drafted, correct prefix)
                predicts it: the drafter lacked the target's computation over its own tokens (n/a when e = 0)
  selection     the target's token is in the drafter's top-16 at e (rank reported)
  memory        the target's (previous token, token) bigram already occurs earlier in the text (copyable)
  confident     the drafter was confident (>= 0.5) in its wrong token
plus a token category of the target's token (number, punctuation/space, newline, code-ish, word, other).

Outputs
  1. share of failures per tag (multi-label) and an exclusive breakdown in priority order
     (first position / intrinsic / computation / memory / selection / rest)
  2. per token category: share of failures and the tag rates
  3. a readable dump of --dump random failures (markdown): context tail, draft with the error marked, truth, the
     drafter's top-5 at e, the target's margin, the tags. READ THESE.

  python scripts/exp94_failures.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
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

CODEISH = set("()[]{}=_:;.<>+-*/%#'\"`\\|&^~!")


def category(s: str) -> str:
    if "\n" in s:
        return "newline"
    st = s.strip()
    if not st:
        return "space"
    if any(c.isdigit() for c in st):
        return "number"
    if all(c in CODEISH or c in ",?" for c in st):
        return "punctuation/symbol"
    if any(c in CODEISH for c in st) or ("_" in st) or (st != st.lower() and st[1:] != st[1:].lower()):
        return "code-ish"
    if st.isalpha():
        return "word"
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
    ap.add_argument("--tie", type=float, default=0.1)
    ap.add_argument("--dump", type=int, default=150)
    ap.add_argument("--out", default="exp94_failures")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)

    def dec(ids):
        return tok.decode(ids)

    recs, n_blocks = [], 0
    for dname in args.datasets:
        for pi, p in enumerate(load_prompts(dname, args.n)):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            out = target(x[None], output_hidden_states=True, logits_to_keep=0)
            feats = extract_context_feature(out.hidden_states, lids)
            tp = torch.softmax(out.logits[0].float(), -1)
            tmargin = (lambda v: (v[:, 0] - v[:, 1]))(tp.topk(2, -1).values).tolist()      # margin at each position
            del out, tp
            xl = x.tolist()
            anchors = list(range(n0, len(x) - bs - 1, args.stride))
            fails = []
            for s in anchors:
                n_blocks += 1
                truth = xl[s + 1 : s + 1 + D]
                pr = torch.softmax(blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].float(), -1)
                top = pr.topk(16, -1)
                y = top.indices[:, 0].tolist()
                e = matched(y, truth)
                if e >= D:
                    continue
                fails.append((s, e, truth, y, top.indices[e].tolist(), top.values[e].tolist()))
            # computation test: fresh blocks at the last correct token, batched per sequence
            fresh = {}
            todo = [(s, e) for s, e, *_ in fails if e > 0]
            for i in range(0, len(todo), 32):
                chunk = todo[i : i + 32]
                L = blocks_logits(draft, target, feats, x, [(s + e, 0) for s, e in chunk], bs)
                for (s, e), row in zip(chunk, L):
                    fresh[(s, e)] = int(row[0].argmax())
            for s, e, truth, y, cand, cprob in fails:
                pos = s + 1 + e                                   # absolute position of the missed token
                want = truth[e]
                prev = xl[pos - 1]
                ctx = xl[:pos]
                bigram = any(ctx[j] == prev and ctx[j + 1] == want for j in range(len(ctx) - 1))
                tags = {
                    "first position": e == 0,
                    "intrinsic": tmargin[pos - 1] < args.tie,
                    "computation": e > 0 and fresh.get((s, e)) == want,
                    "selection": want in cand,
                    "memory": bigram,
                    "confident": cprob[0] >= 0.5,
                }
                recs.append({"dataset": dname, "prompt": pi, "s": s, "e": e, "want": want, "got": y[e],
                             "cat": category(dec([want])), "rank": cand.index(want) + 1 if want in cand else None,
                             "tmargin": tmargin[pos - 1], "dconf": cprob[0], "tags": tags,
                             "ctx": xl[max(n0, pos - 40) : pos], "draft": y, "truth": truth,
                             "top5": list(zip(cand[:5], cprob[:5]))})
        print(f"[{dname}] {n_blocks} blocks, {len(recs)} failures", flush=True)

    nf = len(recs)
    keys = ["first position", "intrinsic", "computation", "selection", "memory", "confident"]
    rows = [{"tag": k, "share of failures": sum(r["tags"][k] for r in recs) / nf} for k in keys]
    print_table(rows, list(rows[0]), f"1a. failures by tag, multi-label ({nf} failures in {n_blocks} blocks)")
    order = ["first position", "intrinsic", "computation", "memory", "selection"]
    excl = Counter()
    for r in recs:
        excl[next((k for k in order if r["tags"][k]), "rest")] += 1
    rows_x = [{"cause (priority order)": k, "share": excl[k] / nf} for k in order + ["rest"]]
    print_table(rows_x, list(rows_x[0]), "1b. exclusive breakdown (first matching cause wins)")
    # e > 0 only: the failures a refresh could address
    later = [r for r in recs if r["e"] > 0]
    rows_l = [{"tag": k, "share (e > 0)": sum(r["tags"][k] for r in later) / max(len(later), 1)} for k in keys[1:]]
    print_table(rows_l, list(rows_l[0]), f"1c. failures after the first position ({len(later)})")
    bycat = defaultdict(list)
    for r in recs:
        bycat[r["cat"]].append(r)
    rows2 = []
    for c, v in sorted(bycat.items(), key=lambda kv: -len(kv[1])):
        row = {"token category": c, "share of failures": len(v) / nf}
        for k in ("intrinsic", "computation", "selection", "memory", "confident"):
            row[k] = sum(r["tags"][k] for r in v) / len(v)
        rows2.append(row)
    print_table(rows2, list(rows2[0]), "2. failures by the category of the target's token")

    res = Path(os.environ.get("HSPEC_RESULTS", "results"))
    res.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    sample = rng.sample(recs, min(args.dump, nf))
    lines = ["# DFlash failures (random sample)\n"]
    for i, r in enumerate(sample):
        e = r["e"]
        draft_txt = dec(r["draft"][:e]) + " ⟦" + dec([r["draft"][e]]) + "⟧ " + dec(r["draft"][e + 1 :])
        truth_txt = dec(r["truth"][:e]) + " ⟦" + dec([r["truth"][e]]) + "⟧ " + dec(r["truth"][e + 1 :])
        tags = ", ".join(k for k, v in r["tags"].items() if v) or "none"
        top5 = ", ".join(f"{dec([t])!r} {p:.2f}" for t, p in r["top5"])
        lines += [f"## {i + 1}. {r['dataset']} #{r['prompt']}  position {e + 1}  [{r['cat']}]  tags: {tags}",
                  f"- context tail: `{dec(r['ctx'])!r}`",
                  f"- draft: `{draft_txt!r}`",
                  f"- truth: `{truth_txt!r}`",
                  f"- drafter top-5 at the error: {top5}   (target's token rank: {r['rank']})",
                  f"- target margin there: {r['tmargin']:.2f}\n"]
    (res / f"{args.out}_dump.md").write_text("\n".join(lines))
    print(f"[dump] {len(sample)} failures -> {res / (args.out + '_dump.md')}", flush=True)
    save_json(args.out, {"args": vars(args), "tags": rows, "exclusive": rows_x, "later": rows_l, "categories": rows2,
                         "records": [{k: v for k, v in r.items() if k not in ("ctx", "draft", "truth", "top5")}
                                     for r in recs]})


if __name__ == "__main__":
    main()
