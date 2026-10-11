"""Latent block diffusion over the verifier's hidden states (feasibility model).

Instead of denoising masked tokens (DFlash), the drafter denoises the TARGET's hidden states for the
next block of positions. Tokens are read off with the target's own head; the predicted states are
also the context a further block can be conditioned on.

Setup for one block: the target has processed tokens up to position s (real states h_0..h_s); its
next token x_{s+1} is known (argmax of the head on h_s). The denoiser predicts the states at
positions s+1 .. s+B (B = block size), each state = the target's state after reading the token at
that position, which also determines the next token. Drafted tokens: x_{s+2} .. x_{s+B+1}.

State vector per position: the DFlash feature slices (target layers 1, 9, 17, 25, 33) plus the last
hidden state, each standardised per dimension (statistics from training data).

Network: a small pretrained LM of the same family (Qwen3-0.6B, optionally only its first n layers):
  context positions 0..s:   token embedding + proj_in(real state)          (causal)
  block positions s+1..s+B: token embedding of x_{s+1} at s+1, mask embedding elsewhere,
                            + proj_in(noisy state x_t) + time embedding     (attend context + block)
  output: proj_out(hidden) = predicted clean states at block positions (x0 prediction)
Diffusion: x_t = t * x0 + (1 - t) * eps, t in [0, 1); deterministic DDIM-style sampler with K steps
from pure noise (t = 0). K = 1 is a single-pass predictor; "regress" mode trains only at t = 0.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def target_states(hidden_states, layer_ids):
    """[1, T, (len(layer_ids)+1) * H]: DFlash slices + the last hidden state."""
    return torch.cat([hidden_states[i + 1] for i in layer_ids] + [hidden_states[-1]], -1)


class StateStats:
    """Per-dimension mean / std for standardising target states."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, xs):
        x = torch.cat([v.float() for v in xs], 0)
        return cls(x.mean(0), x.std(0).clamp_min(1e-2 * x.std(0).median()))

    def norm(self, x):
        return (x.float() - self.mean) / self.std

    def denorm(self, z):
        return z * self.std + self.mean


def time_embed(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[..., None] * 1000 * f
    return torch.cat([a.sin(), a.cos()], -1)


class StateDenoiser(nn.Module):
    def __init__(self, small, d_state: int, n_layers: int | None = None):
        super().__init__()
        cfg = small.config
        H = cfg.hidden_size
        n = n_layers or cfg.num_hidden_layers
        self.embed = copy.deepcopy(small.model.embed_tokens)
        self.layers = nn.ModuleList(copy.deepcopy(small.model.layers[i]) for i in range(n))
        self.norm = copy.deepcopy(small.model.norm)
        self.rotary = copy.deepcopy(small.model.rotary_emb)
        self.mask_emb = nn.Parameter(torch.zeros(H))
        self.proj_ctx = nn.Linear(d_state, H)
        self.proj_noisy = nn.Linear(d_state, H)
        self.time = nn.Sequential(nn.Linear(256, H), nn.SiLU(), nn.Linear(H, H))
        self.proj_out = nn.Linear(H, d_state)
        for m in (self.proj_ctx, self.proj_noisy, self.proj_out):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(m.in_features / H))
            nn.init.zeros_(m.bias)

    def pretrained_params(self):
        return list(self.embed.parameters()) + list(self.layers.parameters()) + list(self.norm.parameters())

    def new_params(self):
        ids = {id(p) for p in self.pretrained_params()}
        return [p for p in self.parameters() if id(p) not in ids]

    def forward(self, ids, ctx_states, blocks, noisy, t, B):
        """ids [T] true tokens (context part used; x_{s+1} given at block start),
        ctx_states [T, D] standardised real states, blocks: list of anchors s,
        noisy [nb, B, D] standardised x_t, t [nb]. Returns predicted x0 [nb, B, D]."""
        T = ids.shape[0]
        dev = ids.device
        nb = len(blocks)
        h_ctx = self.embed(ids) + self.proj_ctx(ctx_states)
        btok = []
        bpos = []
        for s in blocks:
            e = self.mask_emb.expand(B, -1).clone()
            e[0] = self.embed(ids[s + 1])
            btok.append(e)
            bpos.append(torch.arange(s + 1, s + 1 + B, device=dev))
        h_blk = torch.stack(btok) + self.proj_noisy(noisy) + self.time(time_embed(t, 256))[:, None]
        h = torch.cat([h_ctx, h_blk.reshape(nb * B, -1)], 0)[None]
        pos = torch.cat([torch.arange(T, device=dev)] + bpos)[None]
        Q = T + nb * B
        mask = torch.zeros(Q, Q, dtype=torch.bool, device=dev)
        mask[:T, :T] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
        for b, s in enumerate(blocks):
            r = slice(T + b * B, T + (b + 1) * B)
            mask[r, : s + 1] = True
            mask[r, r] = True
        pe = self.rotary(h, pos)
        m4 = mask[None, None]
        for layer in self.layers:
            h = layer(h, attention_mask=m4, position_embeddings=pe, position_ids=pos)
        h = self.norm(h)[0, T:].reshape(nb, B, -1)
        return self.proj_out(h).float()

    @torch.no_grad()
    def sample(self, ids, ctx_states, blocks, B, D, K, generator=None):
        nb = len(blocks)
        dev = ids.device
        eps = torch.randn(nb, B, D, device=dev, generator=generator)
        x = eps
        for k in range(K):
            tk = torch.full((nb,), k / K, device=dev)
            x0 = self.forward(ids, ctx_states, blocks, x, tk, B)
            if k == K - 1:
                return x0
            t_next = (k + 1) / K
            eps_hat = (x - (k / K) * x0) / (1 - k / K)
            x = t_next * x0 + (1 - t_next) * eps_hat
        return x0


def decode_tokens(pred_last, stats_last, target, normed):
    """Greedy tokens from predicted last hidden states (de-standardised) through the target head."""
    h = stats_last.denorm(pred_last).to(target.lm_head.weight.dtype)
    if not normed:
        h = target.model.norm(h)
    return target.lm_head(h).float()
