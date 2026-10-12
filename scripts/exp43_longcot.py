"""Experiment 43: DFlash on long reasoning traces. Does acceptance hold up deep into thinking-mode outputs?

The official Qwen3 DFlash drafters are trained in non-thinking mode, on responses of moderate length.
Reasoning models produce thousands of tokens of chain-of-thought, which is where speed matters most.
Greedy DFlash decoding (lossless) with Qwen3-8B, thinking on and off; per drafter round we log how many
tokens it produced (accepted + 1) and where in the response it started. Reported: tau overall and per
position bucket, per mode and dataset, plus the share of responses that hit the length cap (greedy
thinking can loop, which inflates acceptance; flagged, not hidden).

  python scripts/exp43_longcot.py --n 6 --max-new 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

BUCKETS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 10**9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["math500", "gsm8k", "humaneval"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--out", default="exp43_longcot")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    recs, gens = [], []
    import torch
    for mode in ("nothink", "think"):
        for d in args.datasets:
            for i, pr in enumerate(load_prompts(d, args.n)):
                text = tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=mode == "think")
                ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
                with torch.inference_mode():
                    res = two_stage_generate(draft, target, target, target, ids, args.max_new, stops)
                pos = 0
                for r in res.rounds:
                    recs.append({"mode": mode, "ds": d, "pos": pos, "tau": r})
                    pos += r
                n_out = len(res.generated)
                gens.append({"mode": mode, "ds": d, "len": n_out, "capped": n_out >= args.max_new,
                             "tau": n_out / max(len(res.rounds), 1)})
                print(f"[{mode} {d} {i}] {n_out} tokens, tau {gens[-1]['tau']:.2f}", flush=True)

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    rows = []
    for mode in ("nothink", "think"):
        g = [x for x in gens if x["mode"] == mode]
        r = [x for x in recs if x["mode"] == mode]
        row = {"mode": mode, "responses": len(g), "mean_len": mean(x["len"] for x in g),
               "capped": mean(float(x["capped"]) for x in g), "tau_all": mean(x["tau"] for x in r)}
        for a, b in BUCKETS:
            sel = [x["tau"] for x in r if a <= x["pos"] < b]
            row[f"tau_{a}-{b if b < 10**8 else 'inf'}"] = mean(sel) if sel else None
        rows.append(row)
    cols = list(rows[0])
    print_table(rows, cols, "DFlash tau (tokens per round) by position in the response, Qwen3-8B, greedy")
    drows = []
    for mode in ("nothink", "think"):
        for d in args.datasets:
            r = [x for x in recs if x["mode"] == mode and x["ds"] == d]
            g = [x for x in gens if x["mode"] == mode and x["ds"] == d]
            row = {"mode": mode, "dataset": d, "mean_len": mean(x["len"] for x in g), "tau_all": mean(x["tau"] for x in r)}
            for a, b in BUCKETS[:5]:
                sel = [x["tau"] for x in r if a <= x["pos"] < b]
                row[f"{a}-{b}"] = mean(sel) if sel else None
            drows.append(row)
    print_table(drows, list(drows[0]), "per dataset")
    save_json(args.out, {"args": vars(args), "summary": rows, "per_dataset": drows, "responses": gens})


if __name__ == "__main__":
    main()
