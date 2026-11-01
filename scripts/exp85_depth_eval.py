"""Experiment 85: chained round with REAL shallow features for the in-flight positions (depth lag), no estimation.

Same protocol as exp80-84 (600 rounds, oracle cut n, block 2 at anchor a = s+n, tokens per verification n + m2 + 1).
The in-flight positions s .. a-1 keep only the target taps an early exit after k layers has computed (deeper tap
columns zeroed, hspec.lagtrain.deep_columns); in a real system the target's first k layers run on the in-flight
tokens over the real cache (cost ~k x 0.39 GB in bf16). Use with a drafter trained by train_lag.py --depth-exit k.
Rows: normal, real (all taps), shallow-k (depth lag), and none (in-flight positions dropped, tokens only as a
lagged block).

  python scripts/exp85_depth_eval.py --cache seqs.pt --draft .../drafter_depth2 --depth-exit 2
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

from hspec.lagtrain import deep_columns, draft_logits_depth  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--depth-exit", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--stops", type=int, default=600)
    ap.add_argument("--save", default="exp85_depth_eval")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    val = seqs[: max(int(len(seqs) * 0.2), 1)]
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size

    vr = random.Random(7)
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 8, len(ids) - 3 * bs)]
    vr.shuffle(cand)
    by = {}
    for i, s in cand[: args.stops]:
        by.setdefault(i, []).append(s)

    conds = ["real", "tokens"] + [f"shallow{k}" for k in args.depth_exit]
    recs = []
    with torch.inference_mode():
        for done, (i, anchors) in enumerate(by.items()):
            ids, lo = val[i]
            x = ids.to(dev)
            out = target(x[None], output_hidden_states=True, use_cache=False, logits_to_keep=1)
            f = extract_context_feature(out.hidden_states, lids)
            del out
            masks = {k: deep_columns(draft, k, f.shape[-1], dev) for k in args.depth_exit}
            for s in anchors:
                y1 = blocks_logits(draft, target, f[:, :s], x, [(s, 0)], bs)[0].argmax(-1).tolist()
                n = matched(y1, ids[s + 1 :].tolist())
                r = {"n": n, **{c: 0 for c in conds}}
                if n > 0:
                    a = s + n
                    truth = ids[a + 1 :].tolist()
                    r["real"] = matched(blocks_logits(draft, target, f[:, :a], x, [(a, 0)], bs)[0].argmax(-1).tolist(),
                                        truth)
                    r["tokens"] = matched(blocks_logits(draft, target, f[:, :s], x, [(a, n)], bs)[0]
                                          .argmax(-1).tolist(), truth)
                    for k in args.depth_exit:
                        lg = draft_logits_depth(draft, target, f[:, :a], x, a, n, bs, masks[k])
                        r[f"shallow{k}"] = matched(lg.argmax(-1).tolist(), truth)
                recs.append(r)
            if done % 20 == 0:
                print(f"[eval] {done}/{len(by)} sequences, {len(recs)} rounds, {time.time() - t0:.0f}s", flush=True)
    k_ = len(recs)
    normal = sum(r["n"] + 1 for r in recs) / k_
    rows = [{"in-flight positions get": "none (normal DFlash)", "first token right": "-",
             "tokens / verification": normal, "vs normal": 1.0}]
    pos = [r for r in recs if r["n"] > 0]
    for c in conds:
        t = sum(r["n"] + r[c] + 1 for r in recs) / k_
        rows.append({"in-flight positions get": c, "first token right": sum(r[c] >= 1 for r in pos) / len(pos),
                     "tokens / verification": t, "vs normal": t / normal})
    print_table(rows, list(rows[0]), f"chained round, real shallow features for in-flight positions, {k_} rounds, "
                                     f"oracle cut, drafter {args.draft}")
    save_json(args.save + f"_{Path(args.draft).name}", {"args": vars(args), "rows": rows, "records": recs})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
