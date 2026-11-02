"""Cost per drafting call at batch 1: DFlash vs deep narrow drafters (PARD-style), CUDA graphs and eager.

Same methodology as bench_draft.py: a call is one forward over a short static input without a KV cache (the
memory-bound cost: weights + LM head; attention over a cached prefix is small at these lengths), timed eager and
replayed from a CUDA graph.
  DFlash      C context features + a masked block of bs, LM head (the target's, 4096 x V) on bs-1 positions
  LM drafter  n_new accepted tokens + bs-1 mask tokens through a causal LM, LM head on bs-1 positions
LM drafters: real checkpoints (--models, e.g. amd/PARD-Qwen3-0.6B) and randomly initialised Qwen3 shapes
(--shapes layers x hidden, own tied head) to map time against depth and width.

  python scripts/bench_depth.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def timed(step, reps):
    with torch.inference_mode():
        eager = []
        for i in range(reps + 5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            if i >= 5:
                eager.append((time.perf_counter() - t0) * 1000)
        graph_ms = float("nan")
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
            gt = []
            for i in range(reps + 5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                if i >= 5:
                    gt.append((time.perf_counter() - t0) * 1000)
            graph_ms = _median(gt)
            del g
        except Exception as e:  # noqa: BLE001
            print(f"graph capture failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return _median(eager), graph_ms


def n_params(m, skip=()):
    return sum(p.numel() for n, p in m.named_parameters() if not any(k in n for k in skip)) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--models", nargs="*", default=["amd/PARD-Qwen3-0.6B", "amd/PARD2-Qwen3-8B"])
    ap.add_argument("--shapes", nargs="*",
                    default=["28x1024", "14x1024", "8x1024", "16x1536", "8x2048", "12x2048", "5x4096", "8x4096"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n-new", type=int, default=8, help="accepted tokens added to the LM drafter's context per call")
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM

    bs, D = args.bs, args.bs - 1
    rows = []

    target = load_target(args.target)
    head = target.lm_head
    hidden = target.config.hidden_size
    draft = load_draft(args.draft)
    dev = next(draft.parameters()).device
    width = len(draft.target_layer_ids) * hidden
    h = torch.randn(1, 8, width, device=dev, dtype=torch.bfloat16)
    noise = torch.randn(1, bs, hidden, device=dev, dtype=torch.bfloat16)
    pos = torch.arange(1000, 1000 + 8 + bs, device=dev).unsqueeze(0)

    def dstep():
        hid = draft(target_hidden=h, noise_embedding=noise, position_ids=pos, past_key_values=None, use_cache=False)
        return draft.compute_logits(hid[:, 1:], head).argmax(-1)

    e, g = timed(dstep, args.reps)
    rows.append({"drafter": "DFlash-b16", "layers": len(getattr(draft, "layers", [])) or None,
                 "hidden": hidden, "params (M, excl. shared head/emb)": n_params(draft), "head (M)": head.weight.numel()
                 / 1e6, "eager ms": e, "graph ms": g})
    print(rows[-1], flush=True)
    del draft, target, head
    torch.cuda.empty_cache()

    ids = torch.randint(0, 150000, (1, args.n_new + D), device=dev)

    def lm_row(name, m):
        def step():
            return m(ids, logits_to_keep=D, use_cache=False).logits.argmax(-1)
        e, g = timed(step, args.reps)
        cfg = m.config
        out = m.get_output_embeddings().weight.numel() / 1e6
        rows.append({"drafter": name, "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
                     "params (M, excl. shared head/emb)": n_params(m, ("embed_tokens", "lm_head")), "head (M)": out,
                     "eager ms": e, "graph ms": g})
        print(rows[-1], flush=True)

    for mid in args.models:
        try:
            m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map=dev,
                                                     attn_implementation="sdpa").eval()
            lm_row(mid.split("/")[-1], m)
        except Exception as ex:  # noqa: BLE001
            print(f"[{mid}] failed: {type(ex).__name__}: {ex}", flush=True)
        m = None
        torch.cuda.empty_cache()
    base = AutoConfig.from_pretrained(args.target)
    for sh in args.shapes:
        L, H = (int(v) for v in sh.split("x"))
        cfg = AutoConfig.from_pretrained(args.target)
        cfg.num_hidden_layers, cfg.hidden_size = L, H
        cfg.intermediate_size = 3 * H
        cfg.num_attention_heads = max(H // 128, 1)
        cfg.num_key_value_heads = max(min(base.num_key_value_heads, cfg.num_attention_heads), 1)
        cfg.head_dim = 128
        cfg.tie_word_embeddings = True
        if hasattr(cfg, "layer_types"):
            cfg.layer_types = ["full_attention"] * L
        with torch.device(dev):
            m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        lm_row(f"random {sh}", m)
        m = None
        torch.cuda.empty_cache()

    print_table(rows, list(rows[0]), f"drafting call at batch 1 (block {bs}, LM drafters: {args.n_new} new + {D} mask "
                "tokens; DFlash 2.6 ms in earlier runs)")
    save_json("bench_depth", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
