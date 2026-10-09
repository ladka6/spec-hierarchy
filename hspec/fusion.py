"""Fusion vs K/V injection of the target's hidden features into the DFlash drafter.

DFlash turns the target's context features (positions before the anchor) into extra keys and
values in every drafter layer (K/V injection). The alternative is to fuse a target feature into
the drafter's hidden stream (residual), as EAGLE does. Modes:

  kv        DFlash as is: context features as K/V, no fusion (control)
  fuse_in   no K/V context; the anchor's feature (position s-1, the last one the target has
            computed) is added to every block position before the first layer
  fuse_all  no K/V context; the anchor's feature is added before every layer
  kv_fuse   K/V context and the anchor's feature added before every layer

The fused feature goes through the drafter's own fc + hidden_norm (as for K/V) and then a
per-layer linear map initialised to zero, so kv_fuse starts exactly at DFlash.
Only lag 0 blocks: [x_s, MASK x (bs-1)], labels x_{s+1} .. x_{s+bs-1}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dflash.model import _output_head

from hspec.lagtrain import pack

MODES = ("kv", "fuse_in", "fuse_all", "kv_fuse")


class Fuser(nn.Module):
    def __init__(self, hidden: int, n_layers: int, mode: str):
        super().__init__()
        assert mode in MODES, mode
        self.mode = mode
        n = {"kv": 0, "fuse_in": 1, "fuse_all": n_layers, "kv_fuse": n_layers}[mode]
        self.proj = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(n))
        for p in self.proj:
            nn.init.zeros_(p.weight)

    @property
    def use_ctx(self) -> bool:
        return self.mode in ("kv", "kv_fuse")


def fused_hidden(draft, fuser: Fuser, ctx, anchor, noise, pos, mask):
    """Drafter forward with optional fusion. ctx [1,T,W] context features (T may be 0),
    anchor [1,Q,W] the fused feature for every query position. Returns hidden [1,Q,H]."""
    h = noise
    th = draft.hidden_norm(draft.fc(ctx))
    fz = draft.hidden_norm(draft.fc(anchor)) if len(fuser.proj) else None
    pe = draft.rotary_emb(h, pos)
    for i, layer in enumerate(draft.layers):
        if i < len(fuser.proj):
            h = h + fuser.proj[i](fz)
        h = layer(hidden_states=h, target_hidden=th, attention_mask=mask, position_ids=pos,
                  past_key_value=None, use_cache=False, position_embeddings=pe)
    return draft.norm(h)


def block_forward(draft, fuser, head_src, feats, ids, anchors, bs):
    """Packed blocks at the given anchors (each s >= 1). Returns (hidden [Q,H], starts)."""
    noise, pos, mask, starts = pack(draft, head_src, feats, ids, [(s, 0) for s in anchors], bs)
    T = feats.shape[1]
    if fuser.use_ctx:
        ctx = feats
    else:                                   # no K/V context: blocks see only themselves
        ctx, pos, mask = feats[:, :0], pos[:, T:], mask[..., T:]
    anchor = feats[0, torch.tensor([s - 1 for s in anchors], device=feats.device)]
    anchor = anchor.repeat_interleave(bs, 0)[None]
    return fused_hidden(draft, fuser, ctx, anchor, noise, pos, mask)[0], starts


def block_loss(draft, fuser, head_src, feats, ids, anchors, bs, gamma=0.9):
    """Weighted CE (gamma^(k-1) at offset k) and per-block greedy stats.
    Returns (loss, correct [n_blocks, bs-1] bool or None where past the sequence end)."""
    T = ids.shape[0]
    hidden, starts = block_forward(draft, fuser, head_src, feats, ids, anchors, bs)
    rows, labels, weights, where = [], [], [], []
    for b, (s, st) in enumerate(zip(anchors, starts)):
        for k in range(1, bs):
            if s + k >= T:
                break
            rows.append(st + k - 1)
            labels.append(int(ids[s + k]))
            weights.append(gamma ** (k - 1))
            where.append((b, k - 1))
    rows = torch.tensor(rows, device=ids.device)
    labels = torch.tensor(labels, device=ids.device)
    w = torch.tensor(weights, device=ids.device, dtype=torch.float32)
    logits = draft.compute_logits(hidden[rows], _output_head(head_src)).float()
    ce = F.cross_entropy(logits, labels, reduction="none")
    loss = (ce * w).sum() / w.sum()
    hit = (logits.argmax(-1) == labels).tolist()
    correct = [[None] * (bs - 1) for _ in anchors]
    for (b, k), h in zip(where, hit):
        correct[b][k] = h
    return loss, correct


def accepted(row) -> int:
    """Accepted tokens of one block under greedy verification: leading correct drafts + 1 bonus."""
    n = 0
    for c in row:
        if not c:
            break
        n += 1
    return n + 1
