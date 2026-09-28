"""Experiment 10: acceptance vs lag for the original, control and lag-trained drafters.

Same setup as exp6 (target-greedy trajectories on the evaluation sets, middle-model
features, anchors every `stride` positions) in two modes:
  blind   features for [0, s-g), only the anchor token is given (exp6)
  tokens  features for [0, s-g), the lagged tokens (s-g .. s] are given as inputs
          (what a pipelined drafter actually knows)

  python scripts/exp10_lag_eval.py --drafters orig=z-lab/Qwen3-8B-DFlash-b16 lag16=/path ctrl=/path
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
from exp6_lag import accepted, draft_block  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import free, load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--mid", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--drafters", nargs="+", required=True, help="name=path_or_hf_id")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--lags", nargs="+", type=int, default=[0, 1, 2, 4, 8, 12, 16])
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    mid = load_mid(args.mid)
    stops = stop_ids(target, tok)
    drafters = dict(s.split("=", 1) for s in args.drafters)

    # trajectories (target greedy, via the original drafter) and middle-model features, once
    ref = load_draft(next(iter(drafters.values())))
    trajs = []
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            seq = two_stage_generate(ref, target, target, target, ids, args.max_new, stops).output_ids
            feats = extract_context_feature(mid(seq, output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            ref.target_layer_ids)
            trajs.append((d, ids.shape[1], seq[0], feats))
    del ref
    free()
    print(f"{len(trajs)} trajectories", flush=True)

    acc = defaultdict(list)
    for name, path in drafters.items():
        draft = load_draft(path)
        bs = draft.block_size
        for d, n0, x, feats in trajs:
            for s in range(n0, len(x) - bs - 1, args.stride):
                truth = x[s + 1 : s + bs]
                for g in args.lags:
                    if s - g < 1:
                        continue
                    blind = draft_block(draft, target, feats, s - g, s, x[s], bs)
                    acc[(name, "blind", g, d)].append(accepted(blind, truth))
                    lagged = draft_logits_lagged(draft, target, feats, x, s, g, bs)
                    acc[(name, "tokens", g, d)].append(accepted(lagged, truth))
        print(f"[{name}] done", flush=True)
        del draft
        free()

    rows = []
    for name in drafters:
        for mode in ("blind", "tokens"):
            for g in args.lags:
                row = {"drafter": name, "mode": mode, "lag": g}
                allv = []
                for d in args.datasets:
                    v = acc[(name, mode, g, d)]
                    row[d] = mean(v)
                    allv += v
                row["ALL"] = mean(allv)
                rows.append(row)
    base = {r["lag"]: r["ALL"] for r in rows if r["drafter"] == next(iter(drafters)) and r["mode"] == "blind"}
    for r in rows:
        r["vs_orig_lag0"] = r["ALL"] / base[0] if base.get(0) else float("nan")
    print_table(rows, ["drafter", "mode", "lag", "ALL", "vs_orig_lag0"] + args.datasets,
                "accepted tokens per block vs lag (middle-model features)")
    save_json("exp10_lag_eval", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
