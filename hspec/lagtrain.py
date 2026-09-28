"""Training a DFlash drafter to draft ahead of its verifier (lag-tolerant drafting).

Standard DFlash block: hidden states (features) for every position before the anchor s,
then [x_s, MASK x (bs-1)], predict positions s+1 .. s+bs-1.

Lagged block (lag g): features only for positions [0, s-g); the tokens of the lagged
positions (s-g .. s] are given as ordinary token inputs (in a pipeline the drafter knows
the tokens it drafted ahead, it just has no hidden states for them yet):

    input  = [x_{s-g}, ..., x_{s-1}, x_s, MASK x (bs-1)]     positions s-g .. s+bs-1
    labels =                              x_{s+1} .. x_{s+bs-1}

g = 0 is exactly the standard DFlash block. Many blocks of one sequence are packed into
one forward pass: every block attends to the features before its own lag point and,
bidirectionally, to its own inputs only.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from dflash.model import _draft_value, _output_head, _raw_input_embeddings


def block_inputs(draft, ids: torch.Tensor, s: int, g: int, bs: int):
    """Token ids and positions of one lagged block (ids: 1-D sequence)."""
    ctx_end = s - g
    toks = torch.full((g + bs,), draft.mask_token_id, dtype=torch.long, device=ids.device)
    toks[: g + 1] = ids[ctx_end : s + 1]
    pos = torch.arange(ctx_end, s + bs, device=ids.device)
    return toks, pos, ctx_end


def pack(draft, head_src, feats: torch.Tensor, ids: torch.Tensor, blocks, bs: int):
    """blocks: list of (s, g). feats: [1, T, W] features for the whole sequence.

    Returns (noise_emb [1,Q,H], position_ids [1, T+Q], mask [1,1,Q,T+Q] bool,
    query index of each block's first MASK position, block lengths)."""
    T = feats.shape[1]
    toks, poss, spans, starts = [], [], [], []
    q = 0
    for s, g in blocks:
        t, p, ctx_end = block_inputs(draft, ids, s, g, bs)
        toks.append(t)
        poss.append(p)
        spans.append((q, q + len(t), ctx_end))
        starts.append(q + g + 1)                  # first MASK position of this block
        q += len(t)
    toks = torch.cat(toks)[None]
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    noise = _raw_input_embeddings(head_src, toks, scale)
    pos = torch.cat([torch.arange(T, device=ids.device)] + poss)[None]
    mask = torch.zeros((q, T + q), dtype=torch.bool, device=ids.device)
    for a, b, ctx_end in spans:
        mask[a:b, :ctx_end] = True
        mask[a:b, T + a : T + b] = True
    return noise, pos, mask[None, None], starts


def draft_logits_lagged(draft, head_src, feats, ids, s, g, bs):
    """Logits [bs-1, V] for one block at anchor s with lag g (evaluation)."""
    noise, pos, mask, starts = pack(draft, head_src, feats[:, : s - g], ids, [(s, g)], bs)
    hidden = draft(target_hidden=feats[:, : s - g], noise_embedding=noise, position_ids=pos,
                   attention_mask=mask, past_key_values=None, use_cache=False)
    return draft.compute_logits(hidden[0, starts[0] : starts[0] + bs - 1], _output_head(head_src))


def sample_blocks(n_prompt: int, T: int, n_blocks: int, bs: int, max_lag: int, p_zero: float, rng):
    """Anchors in the response part; lag 0 with probability p_zero, else uniform 1..max_lag."""
    lo, hi = n_prompt, T - 2
    if hi <= lo:
        return []
    out = []
    for s in rng.sample(range(lo, hi), min(n_blocks, hi - lo)):
        g = 0 if max_lag == 0 or rng.random() < p_zero else rng.randint(1, max_lag)
        g = min(g, s - 1)                          # keep at least one feature position
        out.append((s, g))
    return out


def block_loss(draft, head_src, feats, ids, blocks, bs, gamma=0.9):
    """Weighted cross-entropy over all MASK positions of the packed blocks.
    Position k (1-based) in a block gets weight gamma^(k-1): early positions matter most
    for acceptance. Returns (loss, top-1 accuracy at the first position)."""
    T = ids.shape[0]
    noise, pos, mask, starts = pack(draft, head_src, feats, ids, blocks, bs)
    hidden = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                   past_key_values=None, use_cache=False)[0]
    rows, labels, weights, first = [], [], [], []
    for (s, _g), st in zip(blocks, starts):
        for k in range(1, bs):
            if s + k >= T:
                break
            rows.append(st + k - 1)
            labels.append(int(ids[s + k]))
            weights.append(gamma ** (k - 1))
            if k == 1:
                first.append(len(rows) - 1)
    rows = torch.tensor(rows, device=ids.device)
    labels = torch.tensor(labels, device=ids.device)
    w = torch.tensor(weights, device=ids.device, dtype=torch.float32)
    logits = draft.compute_logits(hidden[rows], _output_head(head_src)).float()
    ce = F.cross_entropy(logits, labels, reduction="none")
    loss = (ce * w).sum() / w.sum()
    acc1 = (logits[first].argmax(-1) == labels[first]).float().mean()
    return loss, acc1


__all__ = ["block_inputs", "pack", "draft_logits_lagged", "sample_blocks", "block_loss", "random"]
