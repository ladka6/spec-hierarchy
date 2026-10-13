"""DFlash drafting over a selected subset of its feature context (training-free long-context drafting).

The DFlash drafter injects the target's features of every context position into the K/V of each of its layers and
was trained on sequences of ~3K tokens. At 32K its acceptance falls from ~8.8 to ~2.4 (exp60). Here the drafter sees
only a selected set of context positions:
  recent   sinks + the last positions                   (= running DFlash on a truncated context)
  attn     sinks + a recent window + top positions by the target's attention mass from the queries just before
           the anchor (summed over layers and heads; hspec.sparsekv.attention_scores)
and the positions are either kept (`orig`) or renumbered 0 .. S-1 so the context falls inside the trained range
(`compact`, StreamingLLM-style). The block itself sits right after the context in both cases.

  draft_block(draft, head_src, ctx_feats, ctx_pos, anchor_tok, anchor_pos, bs)  logits [bs-1, V] of one block
  context_for(feats, idx, extra, s, mode)                                       (ctx_feats, ctx_pos, anchor_pos)
  select_positions(score, s, budget, kind, sinks, window)                       sorted LongTensor of positions < s
"""

from __future__ import annotations

import torch

from dflash.model import _draft_value, _output_head, _raw_input_embeddings


def draft_block(draft, head_src, ctx_feats, ctx_pos, anchor_tok: int, anchor_pos: int, bs: int):
    """One block [anchor, MASK x (bs-1)] at positions anchor_pos .. anchor_pos+bs-1 over arbitrary context features
    ctx_feats [1, S, W] at positions ctx_pos [S]. Returns logits [bs-1, V]."""
    dev = ctx_feats.device
    toks = torch.full((1, bs), draft.mask_token_id, dtype=torch.long, device=dev)
    toks[0, 0] = int(anchor_tok)
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    noise = _raw_input_embeddings(head_src, toks, scale)
    S = ctx_feats.shape[1]
    pos = torch.cat([ctx_pos.to(dev), torch.arange(anchor_pos, anchor_pos + bs, device=dev)])[None]
    mask = torch.ones((1, 1, bs, S + bs), dtype=torch.bool, device=dev)
    hid = draft(target_hidden=ctx_feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                past_key_values=None, use_cache=False)
    return draft.compute_logits(hid[0, 1:bs], _output_head(head_src))


def select_positions(score, s: int, budget: int, kind: str, sinks: int = 4, window: int = 512):
    """score: [s] attention mass per prefix position (needed for kind 'attn'). Returns sorted positions < s,
    at most `budget` of them (all of [0, s) if s <= budget)."""
    dev = score.device if score is not None else "cpu"
    if s <= budget:
        return torch.arange(s, device=dev)
    sk = min(sinks, s)
    if kind == "recent":
        return torch.cat([torch.arange(sk, device=dev), torch.arange(s - (budget - sk), s, device=dev)])
    if kind == "attn":
        w = min(window, budget - sk)
        lo, hi = sk, s - w
        k = budget - sk - w
        base = torch.cat([torch.arange(sk, device=dev), torch.arange(hi, s, device=dev)])
        if k <= 0 or hi <= lo:
            return base
        top = score[lo:hi].topk(min(k, hi - lo)).indices + lo
        return torch.cat([base, top]).sort().values
    raise ValueError(kind)


def context_for(feats, idx, extra, s: int, mode: str):
    """Drafter context = features of the selected positions idx (< s) followed by `extra` features for positions
    s, s+1, ... (e.g. a copy's features over the drafted prefix, or None). Returns (ctx_feats, ctx_pos, anchor_pos)
    where anchor_pos is the position of the token right after the context."""
    dev = feats.device
    n_extra = 0 if extra is None else extra.shape[1]
    parts = [feats[:, idx.to(dev)]] + ([extra] if n_extra else [])
    ctx = torch.cat(parts, 1)
    if mode == "orig":
        pos = torch.cat([idx.to(dev), torch.arange(s, s + n_extra, device=dev)])
        return ctx, pos, s + n_extra
    if mode == "compact":
        S = ctx.shape[1]
        return ctx, torch.arange(S, device=dev), S
    raise ValueError(mode)


def score_positions(scores_per_layer):
    """Sum hspec.sparsekv.attention_scores over layers and KV heads, each layer normalised to unit mass."""
    tot = None
    for sc in scores_per_layer:
        m = sc.float().sum(0)
        m = m / m.sum().clamp_min(1e-9)
        tot = m if tot is None else tot + m
    return tot
