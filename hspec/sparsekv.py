"""A KV-sparse copy: the target's (or a low-bit copy's) weights over a selected subset of the target's own KV cache.

At long context the verify pass is dominated by reading the KV cache, not the weights (Qwen3-8B: about 147 KB of
cache per token, 4.7 GB at 32K, against 16 GB of bf16 weights / 5 GB in 4 bit). The copies of exp27/exp56 attend to
the target's full cache, so their relative cost grows with the context. Here the copy attends only to
  sinks        the first `sinks` positions (StreamingLLM)
  window       the last `window` positions before the anchor
  top-k        the `topk` prefix positions with the highest attention mass from the last `probe` query positions
               before the anchor, per layer and per KV head (SnapKV-style selection on the target's own attention)
Keys are already rotated (RoPE applied at write time), so selecting positions is just a gather; the copy's block
tokens get their true position ids. The selection is an index set over the SHARED cache: no second cache.

Helpers:
  prefill_features(model, x, layer_ids, chunk)   full-context features of the drafter's layers + the KV cache,
                                                 prefilled in chunks so only those layers' states are kept
  attention_scores(model, cache, x, s, probe)    per layer, per KV head attention mass of the last `probe`
                                                 queries over the prefix [0, s) (eager attention, hooks)
  select_indices(scores, s, sinks, window, topk) sorted index tensor [Hkv, S] per layer
  sparse_cache(cache, idx, s)                    DynamicCache view of the selected positions
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


def _layer_output(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.inference_mode()
def prefill_features(model, x: torch.Tensor, layer_ids, chunk: int = 2048):
    """x: 1-D ids. Returns (feats [1, T, len(layer_ids) * H], cache over all of x)."""
    layers = model.model.layers
    grabbed = {}
    hooks = []
    for lid in layer_ids:
        # extract_context_feature reads hidden_states[lid + 1] = the output of decoder layer `lid`
        def hook(mod, inp, out, lid=lid):
            grabbed.setdefault(lid, []).append(_layer_output(out))
        hooks.append(layers[lid].register_forward_hook(hook))
    cache = DynamicCache()
    try:
        for a in range(0, len(x), chunk):
            ids = x[a : a + chunk][None]
            pos = torch.arange(a, a + ids.shape[1], device=x.device)[None]
            model(ids, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
    finally:
        for h in hooks:
            h.remove()
    feats = torch.cat([torch.cat(grabbed[lid], 1) for lid in layer_ids], -1)
    return feats, cache


def prefix_cache(cache, n: int) -> DynamicCache:
    return DynamicCache(ddp_cache_data=[(l.keys[:, :, :n], l.values[:, :, :n]) for l in cache.layers])


def _set_attn(model, impl: str):
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation(impl)
            return
        except Exception:
            pass
    model.config._attn_implementation = impl
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = impl


@torch.inference_mode()
def attention_scores(model, cache, x: torch.Tensor, s: int, probe: int = 16):
    """Attention mass over the prefix [0, s) from the queries at positions s-probe+1 .. s.

    Returns a list (per layer) of float tensors [Hkv, s]. Runs the model on the probe tokens with eager attention,
    reading the shared cache up to s-probe+1, and reduces each layer's weights inside a hook so the [H, probe, s]
    tensors are never kept."""
    cfg = model.config
    H, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    g = H // Hkv
    a = max(0, s - probe + 1)
    scores = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        def hook(mod, inp, out, li=li):
            w = out[1]                                              # [1, H, q, a + q]
            if w is None:
                raise RuntimeError("eager attention did not return weights")
            w = w[0, :, :, : s].float().sum(1)                      # [H, s] mass per key over the probe queries
            scores[li] = w.view(Hkv, g, -1).sum(1)                  # [Hkv, s]
        hooks.append(layer.self_attn.register_forward_hook(hook))
    prev = getattr(cfg, "_attn_implementation", "sdpa")
    _set_attn(model, "eager")
    try:
        ids = x[a : s + 1][None]
        pos = torch.arange(a, s + 1, device=x.device)[None]
        model(ids, position_ids=pos, past_key_values=prefix_cache(cache, a), use_cache=True, logits_to_keep=1)
    finally:
        for h in hooks:
            h.remove()
        _set_attn(model, prev)
    return [scores[i] for i in range(len(model.model.layers))]


def select_indices(scores, s: int, sinks: int, window: int, topk: int):
    """Per layer: sorted LongTensor [Hkv, S] of selected prefix positions (S = same for every layer/head)."""
    out = []
    lo = min(sinks, s)
    hi = max(lo, s - window)
    for sc in scores:
        Hkv = sc.shape[0]
        base = torch.cat([torch.arange(0, lo, device=sc.device), torch.arange(hi, s, device=sc.device)])
        k = min(topk, max(0, hi - lo))
        if k > 0:
            mid = sc[:, lo:hi].topk(k, -1).indices + lo                # [Hkv, k]
            idx = torch.cat([base[None].expand(Hkv, -1), mid], 1)
        else:
            idx = base[None].expand(Hkv, -1)
        out.append(idx.sort(-1).values)
    return out


def sparse_cache(cache, idx, s: int) -> DynamicCache:
    """A cache holding, per layer and KV head, only the selected positions of [0, s)."""
    data = []
    for layer, ix in zip(cache.layers, idx):
        k, v = layer.keys[:, :, :s], layer.values[:, :, :s]
        gi = ix[None, :, :, None].expand(1, ix.shape[0], ix.shape[1], k.shape[-1])
        data.append((torch.gather(k, 2, gi), torch.gather(v, 2, gi)))
    return DynamicCache(ddp_cache_data=data)
