"""PDraft: a block diffusion drafter built on a pretrained small LM of the target's family.

DFlash keeps DFlash's mechanism (target features as context, one block of masked positions drafted
in parallel) but its drafter layers start from scratch. PDraft starts from Qwen3-0.6B (all or the
first n layers, its own embeddings and LM head):

  context stream, positions 0 .. s-1:  embed(x_t) + proj(target features at t)   (causal; a KV cache at
                                       inference, extended by the newly verified tokens each round)
  block, positions s .. s+B-1:         [embed(x_s), mask, ..., mask]; attends to context[0, s) and,
                                       bidirectionally, to its own block
  output:                              the small LM's head at mask position s+k predicts x_{s+k}

proj is zero-initialised, so before training PDraft is the plain small LM with masked block slots.
--no-feats gives the same drafter without target features (ablation).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class PDraft(nn.Module):
    def __init__(self, small, d_feat: int, n_layers: int | None = None, use_feats: bool = True):
        super().__init__()
        cfg = small.config
        H = cfg.hidden_size
        n = n_layers or cfg.num_hidden_layers
        self.embed = copy.deepcopy(small.model.embed_tokens)
        self.layers = nn.ModuleList(copy.deepcopy(small.model.layers[i]) for i in range(n))
        self.norm = copy.deepcopy(small.model.norm)
        self.rotary = copy.deepcopy(small.model.rotary_emb)
        self.head = copy.deepcopy(small.lm_head)
        if cfg.tie_word_embeddings:
            self.head.weight = self.embed.weight
        self.use_feats = use_feats
        self.feat_norm = nn.LayerNorm(d_feat)
        self.proj = nn.Linear(d_feat, H)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.mask_emb = nn.Parameter(self.embed.weight.mean(0).clone())

    def new_params(self):
        return list(self.feat_norm.parameters()) + list(self.proj.parameters()) + [self.mask_emb]

    def pretrained_params(self):
        ids = {id(p) for p in self.new_params()}
        return [p for p in self.parameters() if id(p) not in ids]

    def forward(self, ids, feats, anchors, B):
        """ids [T], feats [T, d_feat] real target features, anchors: block anchors s (s >= 1).
        Returns logits [nb, B-1, V] for positions s+1 .. s+B-1."""
        T = ids.shape[0]
        dev = ids.device
        nb = len(anchors)
        h_ctx = self.embed(ids)
        if self.use_feats:
            h_ctx = h_ctx + self.proj(self.feat_norm(feats.float())).to(h_ctx.dtype)
        blk = self.mask_emb.to(h_ctx.dtype).expand(nb, B, -1).clone()
        blk[:, 0] = self.embed(ids[torch.tensor(anchors, device=dev)])
        h = torch.cat([h_ctx, blk.reshape(nb * B, -1)], 0)[None]
        pos = torch.cat([torch.arange(T, device=dev)] + [torch.arange(s, s + B, device=dev) for s in anchors])[None]
        Q = T + nb * B
        mask = torch.zeros(Q, Q, dtype=torch.bool, device=dev)
        mask[:T, :T] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
        for b, s in enumerate(anchors):
            r = slice(T + b * B, T + (b + 1) * B)
            mask[r, :s] = True
            mask[r, r] = True
        pe = self.rotary(h, pos)
        for layer in self.layers:
            h = layer(h, attention_mask=mask[None, None], position_embeddings=pe, position_ids=pos)
        h = self.norm(h)[0, T:].reshape(nb, B, -1)[:, 1:]
        return self.head(h).float()


def block_ce(logits, ids, anchors, gamma=0.9):
    """Gamma-weighted CE over block positions and per-block correctness (bool lists)."""
    T = ids.shape[0]
    nb, L, V = logits.shape
    lab = torch.full((nb, L), -100, dtype=torch.long, device=ids.device)
    for b, s in enumerate(anchors):
        n = min(L, T - 1 - s)
        if n > 0:
            lab[b, :n] = ids[s + 1 : s + 1 + n]
    w = gamma ** torch.arange(L, device=ids.device, dtype=torch.float32)
    ce = F.cross_entropy(logits.reshape(-1, V), lab.reshape(-1), ignore_index=-100, reduction="none").view(nb, L)
    valid = (lab != -100).float()
    loss = (ce * w * valid).sum() / (w * valid).sum()
    hit = (logits.argmax(-1) == lab) & (lab != -100)
    return loss, hit, lab != -100
