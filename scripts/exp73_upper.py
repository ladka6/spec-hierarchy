"""Experiment 73: upper-layer completion. Can the target's own top layers turn an approximate mid-layer state into the
right tokens? (offline, no training, Colab-ready)

Idea: the drafter does not output tokens. It predicts the target's hidden state at layer k for the block positions
(anchor + 14 drafted) in one pass; then the target's own layers k..L-1 run on those positions, attending to the real
context through the KV cache that verification already built at those layers, and the target's LM head gives the
tokens. The upper layers are causal over the block, so later positions see earlier ones (dependency for free), and
the output states at the drafted positions are partly genuine target computation.

Test without any drafter: take the target's REAL layer-k states at block positions s .. s+14 (they predict
x[s+1 .. s+15]), replace each by a vector at exact cosine c (same norm, random orthogonal direction, as exp71), run
layers k..L-1 over [real prefix states at layer k ; perturbed block] (the prefix part reproduces the real KV), then
norm + LM head. Measured per (k, c):
  acc       top-1 accuracy per block position on the true token, averaged
  acc@1-4   positions 1-4 only
  tau       matched prefix of the greedy chain + 1 (as if the drafter predicted the states of the correct tokens
            with this error)
  cos_out   cosine of the resulting last-layer state to the real one: do the upper layers shrink the error
            (cos_out > c) or amplify it?
Reference row k = L ("final"): the same noise put directly on the last layer's output (no upper layers to help).
Cost of the upper layers ~ (L - k) / L of a verify over the same 16 tokens (printed).

  python scripts/exp73_upper.py --cache /content/seqs.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

COS = (1.0, 0.995, 0.99, 0.97, 0.95, 0.9, 0.8, 0.7)


def at_cosine(f, c, gen):
    """Rows at cosine exactly c to f with the same norm (c = 1 returns f)."""
    if c >= 1.0:
        return f
    u = F.normalize(f.float(), dim=-1)
    r = torch.randn(f.shape, generator=gen, device=f.device)
    r = F.normalize(r - (r * u).sum(-1, keepdim=True) * u, dim=-1)
    return ((c * u + (1 - c * c) ** 0.5 * r) * f.float().norm(dim=-1, keepdim=True)).to(f.dtype)


def matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences, e.g. from exp69")
    ap.add_argument("--ks", nargs="+", type=int, default=[20, 24, 28, 32, 34])
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--anchors", type=int, default=500)
    ap.add_argument("--save", default="exp73_upper")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(0)

    seqs = torch.load(args.cache)
    target = load_target(args.target, dev)
    layers = target.model.layers
    L = len(layers)
    B = args.block - 1                                  # block positions s .. s+B-1 predict x[s+1 .. s+B]
    ks = [k for k in args.ks if 0 < k < L] + [L]

    pre_norm = {}
    target.model.norm.register_forward_hook(lambda m, i, o: pre_norm.__setitem__("h", i[0]))

    def run_upper(h, k):
        """Run decoder layers k..L-1 on hidden states h [1, T, H] (full causal attention, no cache), then norm + head."""
        pos = torch.arange(h.shape[1], device=dev)[None]
        pe = target.model.rotary_emb(h, pos)
        for layer in layers[k:]:
            out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe, use_cache=False,
                        cache_position=pos[0])
            h = out[0] if isinstance(out, tuple) else out
        return h

    cand = [(i, s) for i, (ids, lo) in enumerate(seqs) for s in range(lo + 4, len(ids) - args.block - 1)]
    random.Random(0).shuffle(cand)
    by_seq = {}
    for i, s in sorted(cand[: args.anchors]):
        by_seq.setdefault(i, []).append(s)

    stats = {(k, c): {"acc": 0.0, "acc4": 0.0, "tau": 0.0, "cos_out": 0.0, "n": 0} for k in ks for c in COS}
    sanity = []
    with torch.inference_mode():
        for n_done, (i, anchors) in enumerate(by_seq.items()):
            ids, lo = seqs[i]
            x = ids.to(dev)
            out = target(x[None], output_hidden_states=True, use_cache=False)
            hs = out.hidden_states                      # hs[k] = output of layer k (k < L); hs[L] is normed
            last = pre_norm["h"]                        # [1, T, H] last layer output before the final norm
            real_tok = out.logits[0].argmax(-1)
            del out
            for s in anchors:
                truth = x[s + 1 : s + B + 1]
                sanity.append(float((real_tok[s : s + B] == truth).float().mean()))
                for k in ks:
                    for c in COS:
                        if k == L:                      # noise straight on the last layer, no upper layers
                            hb = at_cosine(last[0, s : s + B], c, gen)
                            hout = hb
                        else:
                            hb = at_cosine(hs[k][0, s : s + B], c, gen)
                            h = torch.cat([hs[k][:, : s], hb[None]], 1)
                            hout = run_upper(h, k)[0, s : s + B]
                        lg = target.lm_head(target.model.norm(hout))
                        pred = lg.argmax(-1)
                        st = stats[(k, c)]
                        right = (pred == truth).float()
                        st["acc"] += float(right.mean())
                        st["acc4"] += float(right[:4].mean())
                        st["tau"] += matched(pred.tolist(), truth.tolist()) + 1
                        st["cos_out"] += float(F.cosine_similarity(hout.float(), last[0, s : s + B].float(),
                                                                   dim=-1).mean())
                        st["n"] += 1
            if n_done % 20 == 0:
                print(f"[run] {n_done}/{len(by_seq)} sequences, {time.time()-t0:.0f}s", flush=True)

    print(f"\n[sanity] real states -> real tokens agree on {sum(sanity)/len(sanity):.4f} of block positions "
          f"(greedy text, so should be ~1)")
    rows = []
    for k in ks:
        for c in COS:
            st = stats[(k, c)]
            n = max(st["n"], 1)
            rows.append({"k": "final" if k == L else k, "upper_layers": L - k, "cost_vs_verify": (L - k) / L,
                         "cos_in": c, "acc": st["acc"] / n, "acc@1-4": st["acc4"] / n, "tau": st["tau"] / n,
                         "cos_out": st["cos_out"] / n})
    print_table(rows, list(rows[0]), f"upper-layer completion: layer-k states of the block at cosine c to the real "
                                     f"ones, then the target's layers k..{L-1} + LM head (k = final: noise on the last "
                                     f"layer, no upper layers). tau is out of {B + 1}")
    save_json(args.save, {"args": vars(args), "rows": rows, "sanity": sum(sanity) / len(sanity),
                          "n_anchors": sum(len(v) for v in by_seq.values())})
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
