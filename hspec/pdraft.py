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

feat_mode="kv" (pdraft_kv): target features are NOT added to the input embeddings. Instead, as in
DFlash, ctx = norm(fc(features)) is turned into extra keys/values in EVERY layer with that layer's own
k_proj/v_proj (+ k_norm, rotary at the feature's position). Queries attend to them in a separate
attention whose output is added through a per-layer tanh gate initialised at 0 (Flamingo-style),
so at init the model is exactly the pretrained small LM. Rows at position p see features [0, p)
for block rows (anchor s: features of 0..s-1) and [0, t] for context rows.

ctx_tokens=False (pdraft_featonly): block rows do not attend to the drafter's own token context;
they see only the anchor, their mask slots and the injected target features (as in DFlash).
Separates "the prior crowds out the features" from "the small LM cannot use the features".
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, rotate_half


def _attn(q, k, v, mask):
    """q [1, Hq, Q, D], k/v [1, Hk, K, D] (GQA), mask [Q, K] bool."""
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])


class PDraft(nn.Module):
    def __init__(self, small, d_feat: int, n_layers: int | None = None, use_feats: bool = True,
                 feat_mode: str = "input", ctx_tokens: bool = True):
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
        self.feat_mode = feat_mode
        self.ctx_tokens = ctx_tokens
        assert ctx_tokens or feat_mode == "kv", "features-only needs kv injection"
        self.feat_norm = nn.LayerNorm(d_feat)
        if feat_mode == "kv":
            self.fc = nn.Linear(d_feat, H, bias=False)
            nn.init.normal_(self.fc.weight, 0.0, 0.02)
            self.fc_norm = nn.RMSNorm(H, eps=cfg.rms_norm_eps)
            self.gates = nn.Parameter(torch.zeros(n))
        else:
            self.proj = nn.Linear(d_feat, H)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.mask_emb = nn.Parameter(self.embed.weight.mean(0).clone())

    def gate_params(self):
        return [self.gates] if self.feat_mode == "kv" else []

    def new_params(self):
        if self.feat_mode == "kv":
            mods = list(self.feat_norm.parameters()) + list(self.fc.parameters()) + list(self.fc_norm.parameters())
        else:
            mods = list(self.feat_norm.parameters()) + list(self.proj.parameters())
        return mods + [self.mask_emb]

    def pretrained_params(self):
        ids = {id(p) for p in self.new_params() + self.gate_params()}
        return [p for p in self.parameters() if id(p) not in ids]

    def _layer(self, li, layer, h, pe, mask, fctx, fpe, fmask):
        """Qwen3 decoder layer, written out so the features can be injected as gated extra K/V."""
        a = layer.self_attn
        x = layer.input_layernorm(h)
        shp = (*x.shape[:-1], -1, a.head_dim)
        q = a.q_norm(a.q_proj(x).view(shp)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(shp)).transpose(1, 2)
        v = a.v_proj(x).view(shp).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *pe)
        o = _attn(q, k, v, mask)
        if fctx is not None:
            fshp = (*fctx.shape[:-1], -1, a.head_dim)
            fk = a.k_norm(a.k_proj(fctx).view(fshp)).transpose(1, 2)
            fv = a.v_proj(fctx).view(fshp).transpose(1, 2)
            fk = fk * fpe[0].unsqueeze(1) + rotate_half(fk) * fpe[1].unsqueeze(1)
            o = o + torch.tanh(self.gates[li]).to(o.dtype) * _attn(q, fk, fv, fmask)
        o = a.o_proj(o.transpose(1, 2).reshape(*x.shape[:-1], -1))
        h = h + o
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def forward(self, ids, feats, anchors, B):
        """ids [T], feats [T, d_feat] real target features, anchors: block anchors s (s >= 1).
        Returns logits [nb, B-1, V] for positions s+1 .. s+B-1."""
        T = ids.shape[0]
        dev = ids.device
        nb = len(anchors)
        h_ctx = self.embed(ids)
        kv = self.use_feats and self.feat_mode == "kv"
        if self.use_feats and not kv:
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
            if self.ctx_tokens:
                mask[r, :s] = True
            mask[r, r] = True
        pe = self.rotary(h, pos)
        fctx = fpe = fmask = None
        if kv:
            fctx = self.fc_norm(self.fc(self.feat_norm(feats.float())).to(h.dtype))[None]
            fpe = self.rotary(fctx, torch.arange(T, device=dev)[None])
            fmask = torch.zeros(Q, T, dtype=torch.bool, device=dev)
            fmask[:T] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
            for b, s in enumerate(anchors):
                fmask[T + b * B : T + (b + 1) * B, :s] = True
        for li, layer in enumerate(self.layers):
            h = self._layer(li, layer, h, pe, mask, fctx, fpe, fmask)
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
