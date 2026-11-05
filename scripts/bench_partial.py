"""Partial-pass cost under CUDA graphs: the target's first k layers + final norm + LM head (an early exit over q
drafted tokens on top of a ctx-token KV cache) vs the full forward, same harness as bench_graph.py.

bench_split.py measures the same split in eager mode, where per-layer launch overhead inflates small k; this gives the
numbers for the per-extra-block cost of chained drafting (drafter call + partial pass).

  python scripts/bench_partial.py --target facebook/layerskip-llama3-8B --ks 2 4 8 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_graph import bench_model  # noqa: E402

from hspec.models import load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="facebook/layerskip-llama3-8B")
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--qs", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--save", default="bench_partial")
    args = ap.parse_args()

    model = load_target(args.target)
    cfg = model.config
    L = cfg.num_hidden_layers
    layers = model.model.layers
    types = getattr(cfg, "layer_types", None)
    rows = [bench_model(f"full ({L} layers)", model, args.ctx, args.qs, args.reps)]
    for k in args.ks:
        model.model.layers = torch.nn.ModuleList(list(layers)[:k])
        cfg.num_hidden_layers = k
        if types is not None:
            cfg.layer_types = list(types)[:k]
        rows.append(bench_model(f"first {k} + exit", model, args.ctx, args.qs, args.reps))
    model.model.layers, cfg.num_hidden_layers = layers, L
    if types is not None:
        cfg.layer_types = types
    full = rows[0]
    for r in rows:
        for q in args.qs:
            r[f"r_q{q}"] = r[f"q{q}"] / full[f"q{q}"]
    cols = ["model"] + [f"q{q}" for q in args.qs] + [f"r_q{q}" for q in args.qs] + [f"eager_q{q}" for q in args.qs]
    print_table(rows, cols, f"partial pass vs full forward, CUDA graphs, {args.ctx}-token cache (ms; r_ = ratio to "
                            "full)")
    save_json(args.save, {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
