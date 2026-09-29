"""Cost of split verification: lower k layers (+ exit head) vs upper layers vs full pass.

The one-GPU three-level loop runs the target's first k layers as the middle check (plus the
final norm and LM head to read the exit), and later the upper layers on the accepted tokens,
reusing the lower layers' KV. Verification is memory-bound, so the hope is lower ~ k/L and
upper ~ (L-k)/L of a full pass; the LM head (vocab x hidden) is an extra, non-trivial read.

Eager bf16 with a 512-token KV prefix, median of `--reps` timed calls per point.

  python scripts/bench_split.py --target facebook/layerskip-llama3-8B --exits 16 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import DynamicCache  # noqa: E402

from hspec.models import load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, reps):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="facebook/layerskip-llama3-8B")
    ap.add_argument("--exits", nargs="+", type=int, default=[16, 20])
    ap.add_argument("--qs", nargs="+", type=int, default=[1, 10, 16, 32, 64, 128])
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    model = load_target(args.target)
    base = model.model
    L = model.config.num_hidden_layers
    dev = next(model.parameters()).device
    P = args.prefix
    cache = DynamicCache(config=model.config)
    prefix = torch.randint(0, model.config.vocab_size, (1, P), device=dev)
    model(prefix, past_key_values=cache, use_cache=True, logits_to_keep=1)

    def run(layers, h, pos, mask):
        pe = base.rotary_emb(h, pos)
        for layer in layers:
            h = layer(h, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True,
                      position_embeddings=pe)
            if isinstance(h, tuple):
                h = h[0]
        return h

    rows = []
    for q in args.qs:
        toks = torch.randint(0, model.config.vocab_size, (1, q), device=dev)
        pos = torch.arange(P, P + q, device=dev)[None]
        mask = torch.ones((q, P + q), dtype=torch.bool, device=dev).tril(P)[None, None]
        h0 = base.embed_tokens(toks)

        def full():
            h = run(base.layers, h0, pos, mask)
            model.lm_head(base.norm(h))
            cache.crop(P)

        t_full = timed(full, args.reps)
        row = {"q": q, "full_ms": t_full}
        for k in args.exits:
            hk = run(base.layers[:k], h0, pos, mask)
            cache.crop(P)

            def lower():
                run(base.layers[:k], h0, pos, mask)
                cache.crop(P)

            def lower_exit():
                h = run(base.layers[:k], h0, pos, mask)
                model.lm_head(base.norm(h))
                cache.crop(P)

            t_low = timed(lower, args.reps)
            t_lowx = timed(lower_exit, args.reps)
            # upper layers alone: fill lower-layer KV for the q tokens once, then time the upper part
            run(base.layers[:k], h0, pos, mask)

            def upper_only():
                h = run(base.layers[k:], hk, pos, mask)
                model.lm_head(base.norm(h))
                for i in range(k, L):                       # drop only the upper layers' new entries
                    cache.layers[i].keys = cache.layers[i].keys[..., :P, :]
                    cache.layers[i].values = cache.layers[i].values[..., :P, :]

            t_up = timed(upper_only, args.reps)
            cache.crop(P)
            row.update({f"low{k}+exit": t_lowx, f"low{k}": t_low, f"up{k}": t_up,
                        f"r_low{k}+exit": t_lowx / t_full, f"r_low{k}": t_low / t_full, f"r_up{k}": t_up / t_full})
        rows.append(row)
        print(f"q={q} done", flush=True)

    cols = ["q", "full_ms"]
    for k in args.exits:
        cols += [f"r_low{k}+exit", f"r_low{k}", f"r_up{k}"]
    print_table(rows, cols, f"split verification cost, eager bf16, {P}-token prefix (ms; r_ = ratio to full)")
    save_json("bench_split", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
