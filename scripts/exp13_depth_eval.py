"""Experiment 13: drafter acceptance when the recent positions only have shallow features.

In the one-GPU three-level loop with an early-exit middle check at layer k, the positions
since the last full target pass have features only from target layers below k. For a block at
anchor s with depth lag g (target-greedy trajectories, anchors every `stride` positions):

  full     all features for [0, s)                          (standard DFlash, the ceiling)
  depth    all features for [0, s-g), shallow only for [s-g, s)
  blind    features for [0, s-g) only, nothing for [s-g, s)  (what exp6 measured)

  python scripts/exp13_depth_eval.py --target facebook/layerskip-llama3-8B --exit-layer 16 \
      --drafters base=/path/drafter_ls depth=/path/drafter_ls_depth16
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
from hspec.lagtrain import deep_columns, draft_logits_depth  # noqa: E402
from hspec.models import free, load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="facebook/layerskip-llama3-8B")
    ap.add_argument("--drafters", nargs="+", required=True, help="name=path; the first one generates trajectories")
    ap.add_argument("--exit-layer", type=int, default=16)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--lags", nargs="+", type=int, default=[0, 2, 4, 8, 16, 24])
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    stops = stop_ids(target, tok)
    drafters = dict(s.split("=", 1) for s in args.drafters)

    ref = load_draft(next(iter(drafters.values())))
    trajs = []
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            seq = two_stage_generate(ref, target, target, target, ids, args.max_new, stops).output_ids
            feats = extract_context_feature(target(seq, output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            ref.target_layer_ids)
            trajs.append((d, ids.shape[1], seq[0], feats))
    del ref
    free()
    print(f"{len(trajs)} trajectories", flush=True)

    acc = defaultdict(list)
    for name, path in drafters.items():
        draft = load_draft(path)
        bs = draft.block_size
        deep = None
        for d, n0, x, feats in trajs:
            if deep is None:
                deep = deep_columns(draft, args.exit_layer, feats.shape[-1], feats.device)
            for s in range(n0, len(x) - bs - 1, args.stride):
                truth = x[s + 1 : s + bs]
                for g in args.lags:
                    if s - g < 1:
                        continue
                    lg = draft_logits_depth(draft, target, feats, x, s, g, bs, deep)
                    acc[(name, "depth", g, d)].append(accepted(lg, truth))
                    if g > 0:
                        lg = draft_block(draft, target, feats, s - g, s, x[s], bs)
                        acc[(name, "blind", g, d)].append(accepted(lg, truth))
        print(f"[{name}] done", flush=True)
        del draft
        free()

    rows = []
    for name in drafters:
        base = mean(sum((acc[(name, "depth", 0, d)] for d in args.datasets), []))
        for mode in ("depth", "blind"):
            for g in args.lags:
                if mode == "blind" and g == 0:
                    continue
                allv, row = [], {"drafter": name, "mode": mode, "lag": g}
                for d in args.datasets:
                    v = acc[(name, mode, g, d)]
                    row[d] = mean(v)
                    allv += v
                row["ALL"] = mean(allv)
                row["vs_lag0"] = row["ALL"] / base if base else float("nan")
                rows.append(row)
    print_table(rows, ["drafter", "mode", "lag", "ALL", "vs_lag0"] + args.datasets,
                f"accepted drafted tokens per block; depth = shallow features only (exit after "
                f"{args.exit_layer} layers) for the last `lag` positions")
    save_json("exp13_depth_eval", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
