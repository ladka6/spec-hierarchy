"""Experiment 86 (G1): where and why does DFlash lose acceptance on reasoning traces?

Data: gen_thinking_vllm.py (paired Qwen3-8B greedy responses, thinking on / off, same prompts). For random anchors in
each response, the standard DFlash round with REAL target features (offline, greedy): tau = matched + 1.
Each anchor is tagged with
  region     think (inside <think>), after (answer after </think>), nothink (answer with thinking off)
  position   token index from the start of the response
  margin     the target's top1 - top2 probability at the first drafted position (how decided the target is)
  marker     whether the next 16 tokens contain a reasoning marker (wait / hmm / alternatively / but / let me ...)
Tables:
  1. tau per region (all, and excluding capped responses)
  2. tau per region x position bucket            -> long-context effect vs reasoning text at equal position
  3. tau per region x target-margin bin          -> intrinsic difficulty (same per bin) vs drafter shift (lower)
  4. tau with the drafter's context limited to the last W positions (exp61) on a subset -> attention dilution
  5. think region: tau with vs without reasoning markers ahead
Nothing is trained; DFlash is the original.

  python scripts/exp86_g1_diag.py --data ~/g1
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

from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

POS_BUCKETS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 10**9)]
MARGIN_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01]
MARKERS = ("wait", "hmm", "alternatively", "but ", "let me", "actually", "however", "double-check", "check")
REGIONS = ("think", "after", "nothink")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", required=True, help="directory with think.pt / nothink.pt")
    ap.add_argument("--anchors", type=int, default=32, help="anchors per response")
    ap.add_argument("--window", type=int, default=1024, help="drafter context window for table 4")
    ap.add_argument("--window-anchors", type=int, default=6, help="anchors per response for table 4")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--save", default="exp86_g1_diag")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    rng = random.Random(0)

    rows_in = []
    for mode in ("think", "nothink"):
        rows_in += torch.load(Path(args.data).expanduser() / f"{mode}.pt")
    rows_in = [r for r in rows_in if len(r["ids"]) <= args.max_len and len(r["ids"]) - r["n_prompt"] > 40]
    if args.max_seqs:
        rng.shuffle(rows_in)
        rows_in = rows_in[: args.max_seqs]
    print(f"[data] {len(rows_in)} responses "
          f"({sum(r['mode'] == 'think' for r in rows_in)} think, {sum(r['mode'] == 'nothink' for r in rows_in)} nothink)",
          flush=True)
    tok = load_tokenizer(args.target)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    head = target.lm_head

    recs = []
    with torch.inference_mode():
        for done, r in enumerate(rows_in):
            ids = r["ids"].long()
            x = ids.to(dev)
            lo, T = r["n_prompt"], len(ids)
            out = target(x[None], output_hidden_states=True, use_cache=False, logits_to_keep=1)
            f = extract_context_feature(out.hidden_states, lids)
            last = out.hidden_states[-1][0]
            del out
            cand = list(range(lo + 1, T - bs - 1))
            anchors = sorted(rng.sample(cand, min(args.anchors, len(cand))))
            if not anchors:
                continue
            L = blocks_logits(draft, target, f, x, [(s, 0) for s in anchors], bs).argmax(-1)
            probs = torch.softmax(head(last[anchors]).float(), -1)
            top2 = probs.topk(2, -1).values
            margins = (top2[:, 0] - top2[:, 1]).tolist()
            win_set = set(rng.sample(anchors, min(args.window_anchors, len(anchors))))
            te = r["think_end"]
            for j, s in enumerate(anchors):
                truth = ids[s + 1 :].tolist()
                tau = matched(L[j].tolist(), truth) + 1
                region = "nothink" if r["mode"] == "nothink" else ("think" if te < 0 or s < te else "after")
                ahead = tok.decode(ids[s + 1 : s + bs].tolist()).lower()
                rec = {"region": region, "dataset": r["dataset"], "capped": bool(r["capped"]), "pos": s - lo,
                       "margin": margins[j], "marker": any(m in ahead for m in MARKERS), "tau": tau}
                if s in win_set and s > args.window:
                    a = s - args.window
                    Lw = blocks_logits(draft, target, f[:, a:s], x[a:], [(args.window, 0)], bs)[0].argmax(-1)
                    rec["tau_win"] = matched(Lw.tolist(), truth) + 1
                recs.append(rec)
            if done % 50 == 0:
                print(f"[run] {done}/{len(rows_in)} responses, {len(recs)} anchors, {time.time() - t0:.0f}s", flush=True)

    def mean(v):
        v = list(v)
        return sum(v) / len(v) if v else None

    def sel(**kw):
        return [x for x in recs if all(x[k] == v for k, v in kw.items())]

    t1 = []
    for reg in REGIONS:
        a = sel(region=reg)
        nc = [x for x in a if not x["capped"]]
        t1.append({"region": reg, "anchors": len(a), "tau": mean(x["tau"] for x in a),
                   "anchors (not capped)": len(nc), "tau (not capped)": mean(x["tau"] for x in nc),
                   "mean position": mean(x["pos"] for x in a)})
    print_table(t1, list(t1[0]), "1. DFlash tau (real features, greedy) per region")

    t2 = []
    for a_, b_ in POS_BUCKETS:
        row = {"position in response": f"{a_}-{b_ if b_ < 10**8 else 'inf'}"}
        for reg in REGIONS:
            v = [x["tau"] for x in recs if x["region"] == reg and a_ <= x["pos"] < b_ and not x["capped"]]
            row[f"{reg} tau"] = mean(v)
            row[f"{reg} n"] = len(v)
        t2.append(row)
    print_table(t2, list(t2[0]), "2. tau per region and position bucket (capped responses excluded)")

    t3 = []
    for lo_, hi_ in zip(MARGIN_BINS[:-1], MARGIN_BINS[1:]):
        row = {"target margin": f"{lo_:.2f}-{min(hi_, 1.0):.2f}"}
        for reg in REGIONS:
            v = [x for x in recs if x["region"] == reg and lo_ <= x["margin"] < hi_ and not x["capped"]]
            row[f"{reg} share"] = len(v) / max(len(sel(region=reg)), 1)
            row[f"{reg} tau"] = mean(y["tau"] for y in v)
        t3.append(row)
    print_table(t3, list(t3[0]), "3. tau per region and target-margin bin (share = fraction of the region's anchors)")
    # intrinsic vs shift: think tau predicted from the nothink per-bin tau and the think region's own margin mix
    pred, act = 0.0, mean(x["tau"] for x in recs if x["region"] == "think" and not x["capped"])
    for row in t3:
        if row["nothink tau"] is not None:
            pred += row["think share"] * row["nothink tau"]
    print(f"[3] think region: actual tau {act}, predicted from the nothink per-margin tau and the think margin mix "
          f"{pred:.3f} (close = intrinsic difficulty, actual lower = drafter shift)", flush=True)

    t4 = []
    for reg in REGIONS:
        v = [x for x in recs if x["region"] == reg and "tau_win" in x and not x["capped"]]
        t4.append({"region": reg, "anchors": len(v), "tau full context": mean(x["tau"] for x in v),
                   f"tau last {args.window}": mean(x["tau_win"] for x in v),
                   "mean position": mean(x["pos"] for x in v)})
    print_table(t4, list(t4[0]), f"4. drafter context limited to the last {args.window} positions (same anchors)")

    t5 = []
    for reg in REGIONS:
        for mk in (True, False):
            v = [x for x in recs if x["region"] == reg and x["marker"] == mk and not x["capped"]]
            t5.append({"region": reg, "reasoning marker ahead": mk, "anchors": len(v), "tau": mean(x["tau"] for x in v)})
    print_table(t5, list(t5[0]), "5. tau with / without a reasoning marker in the next 15 tokens")

    t6 = []
    for ds in sorted({x["dataset"] for x in recs}):
        row = {"dataset": ds}
        for reg in REGIONS:
            row[reg] = mean(x["tau"] for x in recs if x["dataset"] == ds and x["region"] == reg and not x["capped"])
        t6.append(row)
    print_table(t6, list(t6[0]), "6. tau per dataset and region (capped excluded)")

    save_json(args.save, {"args": vars(args), "t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6,
                          "records": recs})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
