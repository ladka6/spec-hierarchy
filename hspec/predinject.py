"""Predecessor injection into a DFlash block (DSpine-style, 2609.36173) with the predecessor taken from different sources.

DSpine adds, after every drafter layer, a gated message from position t-1 into position t of the block. Its message is
the drafter's own prediction of token t-1; its diagnostic shows that a CORRECT predecessor lifts the accuracy at
position 12 from ~51% to ~93%. Here the same injection runs in a second drafter pass over the block, after a cheap
copy of the target has read the first-pass draft, and the predecessor message comes from:

  none       no message (LoRA fine-tune of the drafter only: the control)
  self       the drafter's own first-pass distribution at t-1, as a soft token embedding (DSpine's source)
  copy_soft  the copy's next-token distribution for t-1, as a soft token embedding
  copy_feat  the copy's features at position t-1, depth-aligned: drafter layer l gets the copy's tap-layer-l feature
  copy_both  copy_soft and copy_feat together
  oracle     the true token at t-1 (upper bound: DSpine's "correct predecessor")

The first predecessor (the anchor) is always the true anchor token. Each variant has its own zero-initialised injection
modules and its own LoRA adapters on the drafter's linears (MultiLoRA, selected through ACTIVE["v"]), so all variants
can be trained in one loop over the same data.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

ACTIVE = {"v": None}
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VARIANTS = ("none", "self", "copy_soft", "copy_feat", "copy_both", "oracle")


class MultiLoRA(torch.nn.Module):
    """A frozen Linear plus one low-rank adapter per variant; ACTIVE["v"] picks the adapter (None: base only)."""

    def __init__(self, base: torch.nn.Linear, variants, rank: int, alpha: float):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.A = torch.nn.ParameterDict({v: torch.nn.Parameter(
            torch.randn(rank, base.in_features, device=dev) / math.sqrt(base.in_features)) for v in variants})
        self.B = torch.nn.ParameterDict({v: torch.nn.Parameter(
            torch.zeros(base.out_features, rank, device=dev)) for v in variants})
        self.scale = alpha / rank

    def forward(self, x):
        y = self.base(x)
        v = ACTIVE["v"]
        if v is None or v not in self.A:
            return y
        return y + ((x.float() @ self.A[v].T) @ self.B[v].T * self.scale).to(y.dtype)


def add_multilora(draft, variants, rank=16, alpha=32.0):
    """MultiLoRA on every attention / MLP linear of the drafter. Returns {variant: [params]}."""
    params = {v: [] for v in variants}
    for layer in draft.layers:
        for parent, names in ((layer.self_attn, LINEARS[:4]), (layer.mlp, LINEARS[4:])):
            for nm in names:
                m = MultiLoRA(getattr(parent, nm), variants, rank, alpha)
                setattr(parent, nm, m)
                for v in variants:
                    params[v] += [m.A[v], m.B[v]]
    return params


class Inject(torch.nn.Module):
    """LN -> low-rank MLP -> hidden; last projection zero-initialised (starts as the plain drafter)."""

    def __init__(self, in_dim: int, hidden: int, rank: int = 512):
        super().__init__()
        self.down = torch.nn.Linear(in_dim, rank)
        self.up = torch.nn.Linear(rank, hidden)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, m):
        m = F.layer_norm(m.float(), m.shape[-1:])
        return self.up(F.silu(self.down(m)))


def make_injectors(variant: str, hidden: int, n_layers: int, rank: int, device):
    if variant == "none":
        return None
    dim = {"self": hidden, "copy_soft": hidden, "oracle": hidden, "copy_feat": hidden, "copy_both": 2 * hidden}[variant]
    return torch.nn.ModuleList([Inject(dim, hidden, rank) for _ in range(n_layers)]).to(device)


def injected_forward(draft, feats, noise, pos, mask, rows, msgs, injectors):
    """DFlashDraftModel.forward with a message added to `rows` of the block after every layer.
    msgs: list (one per layer) of [len(rows), D] tensors, or None."""
    hs = noise
    target_hidden = draft.hidden_norm(draft.fc(feats))
    pe = draft.rotary_emb(hs, pos)
    for li, layer in enumerate(draft.layers):
        hs = layer(hidden_states=hs, target_hidden=target_hidden, attention_mask=mask, position_ids=pos,
                   past_key_value=None, use_cache=False, position_embeddings=pe)
        if injectors is not None:
            add = injectors[li](msgs[li]).to(hs.dtype)
            hs = hs.clone()
            hs[0, rows] = hs[0, rows] + add
    return draft.norm(hs)


def soft_embed(emb, logp_top, idx_top):
    """Expected token embedding under a (renormalised) top-k distribution. logp_top, idx_top: [N, k]."""
    p = torch.softmax(logp_top.float(), -1)
    return torch.einsum("nk,nkh->nh", p, emb(idx_top).float())
