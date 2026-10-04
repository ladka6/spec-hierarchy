"""Faker: a small causal model that emulates the target's context vectors for drafted tokens.

DFlash conditions every block on c_p = hidden_norm(fc(target features at p)) for all positions p
before the block anchor. For tokens the target has not processed yet (a drafted block, a tree
path) c_p does not exist. The faker produces a stand-in in one parallel pass:

    inputs   real c for the last W verified positions  (s-W .. s-1)
             embeddings of the tokens x_s .. x_{s+L-1}  (anchor + drafted tokens, known)
    model    small pre-norm transformer, causal over [context ; tokens]
    output   c_hat_p for p = s .. s+L-1   (in the drafter's normalized context space)

c_p is a deterministic function of the tokens, so this is regression, not generation. Training
matches the frozen drafter's behaviour: the drafter's logits for blocks anchored at s+i given
[real c up to s ; c_hat for s .. s+i-1] should equal its logits given the real c (KL), plus a
small cosine / relative-MSE term on c_hat itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dflash.model import _output_head

from hspec.feathead import feat_loss, raw_context, token_emb
from hspec.lagtrain import pack


class Faker(nn.Module):
    def __init__(self, hidden: int, dim: int = 1024, layers: int = 3, heads: int = 16, window: int = 128,
                 max_len: int = 64):
        super().__init__()
        self.window, self.max_len = window, max_len
        self.ctx_in = nn.Linear(hidden, dim)
        self.tok_in = nn.Linear(hidden, dim)
        self.pos = nn.Embedding(window + max_len, dim)
        self.seg = nn.Embedding(2, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, hidden)
        self.cfg = {"hidden": hidden, "dim": dim, "layers": layers, "heads": heads, "window": window,
                    "max_len": max_len}

    def forward(self, c_ctx, e_tok, hidden_norm):
        """c_ctx [B, C, H] real context vectors (C <= window), e_tok [B, L, H] token embeddings.
        Returns c_hat [B, L, H]."""
        B, C, _ = c_ctx.shape
        L = e_tok.shape[1]
        dev = c_ctx.device
        pos = torch.arange(self.window - C, self.window + L, device=dev)
        typ = torch.cat([torch.zeros(C, dtype=torch.long, device=dev), torch.ones(L, dtype=torch.long, device=dev)])
        h = torch.cat([self.ctx_in(c_ctx), self.tok_in(e_tok)], 1) + self.pos(pos) + self.seg(typ)
        mask = torch.triu(torch.ones(C + L, C + L, dtype=torch.bool, device=dev), 1)
        h = self.body(h, mask=mask)
        return hidden_norm(self.out(self.norm(h[:, C:])))


def fake_context(faker, draft, head_src, c, ids, s, L):
    """c_hat [L, H] for positions s .. s+L-1 from real c [T, H] (only c[:s] is used) and ids."""
    dt = next(faker.parameters()).dtype
    lo = max(0, s - faker.window)
    e = token_emb(draft, head_src, ids[s : s + L][None]).to(dt)
    return faker(c[lo:s][None].to(dt), e, draft.hidden_norm)[0]


def block_logits_ctx(draft, head_src, ctx, ids, anchors, bs):
    """Logits [n, bs-1, V] of standard blocks at the given anchors, context vectors ctx [1, T, H]
    (a block at a sees ctx[:a])."""
    noise, pos, mask, starts = pack(draft, head_src, ctx, ids, [(a, 0) for a in anchors], bs)
    with raw_context(draft):
        hid = draft(target_hidden=ctx, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                    past_key_values=None, use_cache=False)[0]
    rows = torch.stack([hid[st : st + bs - 1] for st in starts])
    return draft.compute_logits(rows, _output_head(head_src))


def faker_loss(faker, draft, head_src, c, ids, s, bs, kl_weight=1.0, offsets=None):
    """Loss for one anchor s. c: [T, H] real context vectors of the sequence (no grad).
    Blocks re-anchored at s+i (i in offsets, default 1 .. bs-2) see real c before s and c_hat after.
    Returns (loss, stats)."""
    L = bs - 1
    offsets = offsets or list(range(1, bs - 1))
    c_hat = fake_context(faker, draft, head_src, c, ids, s, L)
    real = c[s : s + L]
    fl, cos = feat_loss(c_hat, real)
    anchors = [s + i for i in offsets]
    dd = next(draft.parameters()).dtype
    with torch.no_grad():
        ref = block_logits_ctx(draft, head_src, c[: s + L][None].to(dd), ids, anchors, bs).float()
    ctx = torch.cat([c[:s].to(dd), c_hat.to(dd)], 0)[None]
    lg = block_logits_ctx(draft, head_src, ctx, ids, anchors, bs).float()
    kl = F.kl_div(torch.log_softmax(lg, -1), torch.log_softmax(ref, -1), log_target=True, reduction="none")
    kl = kl.sum(-1).mean()
    agree = (lg.argmax(-1) == ref.argmax(-1)).float().mean()
    return fl + kl_weight * kl, {"feat": float(fl), "kl": float(kl), "cos": cos.detach(), "agree": float(agree)}


def save_faker(faker, path):
    torch.save({"state": faker.state_dict(), "cfg": faker.cfg}, path)


def load_faker(path, device, dtype=torch.float32):
    d = torch.load(path, map_location="cpu")
    f = Faker(**d["cfg"])
    f.load_state_dict(d["state"])
    return f.to(device=device, dtype=dtype).eval()
