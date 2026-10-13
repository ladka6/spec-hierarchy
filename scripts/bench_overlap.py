"""Can the drafter + copy work of the next round hide behind the target's verify pass on ONE GPU?

CUDA graphs (as bench_graph.py) for: the target verifying q tokens after a ctx-token prefix, the copy over a 16-token
block after the same prefix, and the drafter at batch 1. Timed alone and with the target on one stream while
drafter + copy (+ a second drafter call, the restarts) replay on another, at batch 1.

  overlap gain = (t_target + t_small) / t_together        (2.0 = perfectly hidden, 1.0 = no overlap)

  python scripts/bench_overlap.py --copies bnb4:Qwen/Qwen3-8B tao4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import _make_cache  # noqa: E402

from hspec.models import free, load_draft, load_mid, load_target  # noqa: E402
from hspec.pipeline import crop  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


KEEP = []


def timed(fn, reps=30):
    out = []
    for i in range(reps + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if i >= 5:
            out.append((time.perf_counter() - t0) * 1000)
    out.sort()
    return out[len(out) // 2]


@torch.inference_mode()
def capture_model(model, ctx, q):
    """CUDA graph of a forward over q tokens after a ctx-token prefix (cache cropped back each time)."""
    dev = next(model.parameters()).device
    vocab = model.config.vocab_size
    cache = _make_cache(model.config)
    model(torch.randint(0, vocab, (1, ctx), device=dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
    ids = torch.randint(0, vocab, (1, q), device=dev)
    pos = torch.arange(ctx, ctx + q, device=dev).unsqueeze(0)

    def step():
        model(ids, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=q)
        crop(cache, ctx)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        model(ids, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=q)
    crop(cache, ctx)
    KEEP.extend([ids, pos, cache])          # a graph reads its inputs from fixed addresses: keep them alive
    return g, cache


@torch.inference_mode()
def capture_draft(draft, target, ctx_feat=64, batch=1):
    dev = next(draft.parameters()).device
    hidden = target.config.hidden_size
    width = len(draft.target_layer_ids) * hidden
    bs = draft.block_size
    h = torch.randn(batch, ctx_feat, width, device=dev, dtype=torch.bfloat16)
    noise = torch.randn(batch, bs, hidden, device=dev, dtype=torch.bfloat16)
    pos = torch.arange(1000, 1000 + ctx_feat + bs, device=dev).unsqueeze(0).expand(batch, -1)

    def step():
        hid = draft(target_hidden=h, noise_embedding=noise, position_ids=pos, past_key_values=None, use_cache=False)
        return draft.compute_logits(hid[:, 1:], target.lm_head).argmax(-1)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    KEEP.extend([h, noise, pos])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copies", nargs="+", default=["bnb4:Qwen/Qwen3-8B", "tao4:Qwen/Qwen3-8B"])
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--verify-q", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--restart-batch", type=int, default=3)
    ap.add_argument("--out", default="bench_overlap")
    args = ap.parse_args()

    target = load_target(args.target)
    draft = load_draft(args.draft)
    gd = capture_draft(draft, target)
    gr = capture_draft(draft, target, batch=args.restart_batch)
    gts = {q: capture_model(target, args.ctx, q) for q in args.verify_q}
    rows = []
    for spec in args.copies:
        try:
            copy = load_mid(spec)
            gc, ccache = capture_model(copy, args.ctx, draft.block_size)
        except Exception as e:  # noqa: BLE001
            print(f"[{spec}] failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        sa, sb = torch.cuda.Stream(), torch.cuda.Stream()

        def small():
            gd.replay()
            gc.replay()
            gr.replay()

        t_d, t_c, t_r = timed(gd.replay), timed(gc.replay), timed(gr.replay)
        t_small = timed(small)
        for q, (gt, _) in gts.items():
            t_t = timed(gt.replay)

            def together():
                with torch.cuda.stream(sa):
                    gt.replay()
                with torch.cuda.stream(sb):
                    small()
                torch.cuda.current_stream().wait_stream(sa)
                torch.cuda.current_stream().wait_stream(sb)

            t_both = timed(together)
            row = {"copy": spec, "verify_q": q, "target_ms": t_t, "drafter_ms": t_d, "copy_ms": t_c,
                   "restarts_ms": t_r, "small_total_ms": t_small, "together_ms": t_both,
                   "sequential_ms": t_t + t_small, "overlap_gain": (t_t + t_small) / t_both,
                   "hidden_share": (t_t + t_small - t_both) / t_small}
            rows.append(row)
            print(row, flush=True)
        del gc, ccache
        free(copy)
    print_table(rows, list(rows[0]), f"one A100, CUDA graphs, ctx {args.ctx}: target verify alone, drafter + copy + "
                                     "restarts alone, and both on two streams (hidden_share = how much of the small "
                                     "work disappears behind the target)")
    save_json(args.out, {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
