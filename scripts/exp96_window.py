"""Experiment 96: does a DFlash draft "heal" after its first error? Oracle window repair (offline, T=0).

exp95: in the target's own text a changed token mostly affects the next 1-3 tokens. If DFlash drafts likewise become
right again a few positions after their first error, one round could accept far beyond the first error by repairing a
short window and keeping the rest of the draft.

Per block (target greedy text t = truth, plain DFlash draft y with real target features), first error e:
  Table 1  oracle window repair: y[:e] + t[e:e+w] + y[e+w:], accepted = longest match + 1, for w = 0 (plain), 1, 2,
           3, 4, 6, 8; and with up to R successive repairs (each repairs the next error with the same w)
  Table 2  tail correctness after the first error: P(y[e+o] == t[e+o]) for offsets o = 1..12, next to the drafter's
           ordinary positional accuracy at the same block positions (blocks without an error before them)

  python scripts/exp96_window.py
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
from hspec.utils import print_table, save_json  # noqa: E402


def repaired(y, t, w, R):
    """Accepted length after up to R oracle window repairs of width w (each at the next error)."""
    c = list(y)
    for _ in range(R):
        e = matched(c, t)
        if e >= len(c):
            break
        c[e : e + w] = t[e : e + w]
        c = c[: len(y)]
    return matched(c, t) + 1


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--repairs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="exp96_window")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)

    acc = defaultdict(list)
    tail = defaultdict(lambda: [0, 0])          # offset -> [right, total] after the first error
    pos = defaultdict(lambda: [0, 0])           # position -> [right, total] when everything before is right
    n_blocks = 0
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            xl = x.tolist()
            for s in range(ids.shape[1], len(x) - bs - 1, args.stride):
                n_blocks += 1
                t = xl[s + 1 : s + 1 + D]
                y = blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].argmax(-1).tolist()
                e = matched(y, t)
                acc[(0, 0)].append(e + 1)
                for w in args.widths:
                    for R in args.repairs:
                        acc[(w, R)].append(repaired(y, t, w, R))
                for k in range(min(e + 1, D)):
                    pos[k][0] += y[k] == t[k]
                    pos[k][1] += 1
                if e < D:
                    for o in range(1, D - e):
                        tail[o][0] += y[e + o] == t[e + o]
                        tail[o][1] += 1
        print(f"[{dname}] {n_blocks} blocks", flush=True)

    base = sum(acc[(0, 0)]) / len(acc[(0, 0)])
    rows = []
    for w in args.widths:
        row = {"window w": w}
        for R in args.repairs:
            row[f"{R} repair{'s' if R > 1 else ''}"] = sum(acc[(w, R)]) / len(acc[(w, R)])
        rows.append(row)
    print(f"plain DFlash accepted: {base:.3f} ({n_blocks} blocks)")
    print_table(rows, list(rows[0]), "1. accepted per verification after oracle window repair(s), rest of the draft kept")
    rows2 = []
    for o in range(1, 13):
        if tail[o][1]:
            rows2.append({"offset after first error": o, "tail token right": tail[o][0] / tail[o][1],
                          "n": tail[o][1]})
    print_table(rows2, list(rows2[0]), "2. is the draft right again after its first error?")
    rows3 = [{"block position": k + 1, "right (prefix right)": v[0] / v[1]} for k, v in sorted(pos.items()) if v[1]]
    print_table(rows3, list(rows3[0]), "2b. reference: drafter accuracy at each position when everything before is right")
    save_json(args.out, {"args": vars(args), "plain": base, "repairs": rows, "tail": rows2, "positional": rows3})


if __name__ == "__main__":
    main()
