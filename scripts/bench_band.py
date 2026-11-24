"""Cost of the pieces of a chained round, same harness as bench_graph.py / bench_partial.py (CUDA graphs, batch 1).

The cost model of the band pass used estimates (drafter 3.3 ms, verify 19 ms, band = 28% of a pass). This measures
them on the actual target, on a ctx-token cache:
  full verify       all layers + LM head over q new tokens (q = 16, 32, 48: one, two, three blocks in one verify)
  body n            n decoder layers over q tokens, NO LM head: the band pass (which n layers does not matter for
                    the time, only how many; the layers of the band shape are used)
  body n + head     the same with the final norm and LM head (a shallow verify)
  bridge            the affine + MLP bridge over q tokens
  drafter           one drafter call (block of --block, c fresh context features) + LM head on the drafted positions
Derived: a chain stage = body n + bridge + drafter; a round of B blocks = B drafter calls + (B - 1) band passes +
one full verify over 16 B tokens. The table gives the time of 1-, 2- and 3-block rounds per band size and the tokens
per verify a chain must reach to break even with one block per verify.

  python scripts/bench_band.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_graph import _time  # noqa: E402

from hspec.bridges import Bridge  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def graph_time(step, reps):
    """eager and CUDA-graph time of step() in ms (graph = nan if the capture fails)"""
    eager = _time(step, reps)
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        t = _time(g.replay, reps)
        del g
        return eager, t
    except Exception as e:  # noqa: BLE001
        print(f"graph capture failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return eager, float("nan")


@torch.inference_mode()
def bench_pass(name, model, ctx, qs, reps, head):
    """new tokens on a ctx-token dynamic cache, cropped back after every call (as bench_graph.bench_model)"""
    from dflash.model import _make_cache

    from hspec.pipeline import crop

    fwd = model if head else model.model
    dev = next(model.parameters()).device
    vocab = model.config.vocab_size
    cache = _make_cache(model.config)
    prefix = torch.randint(0, vocab, (1, ctx), device=dev)
    fwd(prefix, past_key_values=cache, use_cache=True)
    keep = [(layer.keys, layer.values) for layer in cache.layers]     # keep the prefix tensors alive for the graph
    row = {"piece": name}
    for q in qs:
        ids = torch.randint(0, vocab, (1, q), device=dev)
        pos = torch.arange(ctx, ctx + q, device=dev).unsqueeze(0)

        def step():
            out = fwd(ids, position_ids=pos, past_key_values=cache, use_cache=True)
            crop(cache, ctx)
            return out

        row[f"eager q{q}"], row[f"q{q}"] = graph_time(step, reps)
        crop(cache, ctx)
    del keep
    print(row, flush=True)
    return row


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--bands", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12], help="layers of the band pass")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--qs", nargs="+", type=int, default=[16, 32, 48])
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--one-block", type=float, default=5.10, help="tokens per verify of one block (for break-even)")
    ap.add_argument("--save", default="bench_band")
    args = ap.parse_args()

    model = load_target(args.target)
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.hidden_size
    layers = model.model.layers
    types = getattr(cfg, "layer_types", None)
    dev = next(model.parameters()).device
    rows = [bench_pass(f"full verify ({L} layers + head)", model, args.ctx, args.qs, args.reps, True)]
    c = round(0.75 * L) - 2                                  # the band ends near 3/4 depth
    for n in args.bands:
        a = 2 if n > 2 else 0
        kept = list(range(a)) + list(range(c - (n - a) + 1, c + 1))
        model.model.layers = torch.nn.ModuleList([layers[i] for i in kept])
        cfg.num_hidden_layers = n
        if types is not None:
            cfg.layer_types = [types[i] for i in kept]
        rows.append(bench_pass(f"body {n}", model, args.ctx, args.qs, args.reps, False))
        rows.append(bench_pass(f"body {n} + head", model, args.ctx, args.qs, args.reps, True))
    model.model.layers, cfg.num_hidden_layers = layers, L
    if types is not None:
        cfg.layer_types = types

    br = Bridge(H, "mlp", 1024).to(dev)
    row = {"piece": "bridge (affine + MLP)"}
    for q in args.qs:
        x = torch.randn(1, q, H, device=dev, dtype=torch.bfloat16)
        row[f"eager q{q}"], row[f"q{q}"] = graph_time(lambda x=x: br(x), args.reps)
    rows.append(row)
    print(row, flush=True)

    draft = load_draft(args.draft)
    width = len(draft.target_layer_ids) * H
    head = model.lm_head
    for cn in (1, 16):
        h = torch.randn(1, cn, width, device=dev, dtype=torch.bfloat16)
        noise = torch.randn(1, args.block, H, device=dev, dtype=torch.bfloat16)
        pos = torch.arange(1000, 1000 + cn + args.block, device=dev).unsqueeze(0)

        def step(h=h, noise=noise, pos=pos):
            hid = draft(target_hidden=h, noise_embedding=noise, position_ids=pos, past_key_values=None, use_cache=False)
            return draft.compute_logits(hid[:, 1:], head).argmax(-1)

        e, g = graph_time(step, args.reps)
        row = {"piece": f"drafter call ({cn} new context features)", "eager q16": e, "q16": g}
        rows.append(row)
        print(row, flush=True)

    cols = ["piece"] + [f"q{q}" for q in args.qs] + [f"eager q{q}" for q in args.qs]
    print_table(rows, cols, f"{args.target}: pieces of a chained round, CUDA graphs, {args.ctx}-token cache (ms)")

    def get(piece, q):
        r = next(r for r in rows if r["piece"] == piece)
        return r.get(f"q{q}", float("nan"))

    F1 = get(f"full verify ({L} layers + head)", 16)
    D1, D16 = get("drafter call (1 new context features)", 16), get("drafter call (16 new context features)", 16)
    bq = get("bridge (affine + MLP)", 16)
    base = D1 + F1
    drows = [{"round": "1 block (plain DFlash)", "ms": base, "vs 1 block": 1.0, "break-even tokens": args.one_block}]
    for n in args.bands:
        E = get(f"body {n}", 16) + bq
        for B in (2, 3):
            qf = 16 * B if 16 * B in args.qs else args.qs[-1]
            t = D1 + (B - 1) * (E + D16) + get(f"full verify ({L} layers + head)", qf)
            drows.append({"round": f"{B} blocks, band of {n} layers", "ms": t, "vs 1 block": t / base,
                          "break-even tokens": args.one_block * t / base, "band + bridge ms": E,
                          "band / verify": E / F1})
    print_table(drows, ["round", "ms", "vs 1 block", "break-even tokens", "band + bridge ms", "band / verify"],
                f"Round time and the tokens per verify needed to break even (one block gives {args.one_block})")
    save_json(args.save, {"args": vars(args), "rows": rows, "rounds": drows})


if __name__ == "__main__":
    main()
