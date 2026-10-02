"""Feature prediction for a DFlash drafter (EAGLE's idea inside a block drafter).

DFlash conditions on one context vector per earlier position,
    c_t = hidden_norm(fc([h_l1, ..., h_l5] at t))   (target hidden states of 5 layers),
which every drafter layer attends to. Positions the target has not processed yet have no c_t.
A feature head predicts them from what the drafter already has after a normal round:

    c_hat_p = hidden_norm(MLP([drafter hidden at block position p+1 ; embedding of token x_p]))

(the drafter's hidden at p+1 predicts token p+1, like the target's state at p). With c_hat for
the suspected / accepted tokens the next block can be drafted without waiting for the target.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from dflash.model import _draft_value, _output_head, _raw_input_embeddings

from hspec.lagtrain import pack


class FeatHead(nn.Module):
    def __init__(self, hidden: int, mult: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * hidden, mult * hidden), nn.GELU(), nn.Linear(mult * hidden, hidden))

    def forward(self, h_draft, e_tok, hidden_norm):
        return hidden_norm(self.net(torch.cat([h_draft, e_tok], -1)))


def context_vectors(draft, feats):
    """[1, T, H] context vectors c_t from raw concatenated target features."""
    return draft.hidden_norm(draft.fc(feats))


def token_emb(draft, head_src, toks):
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    return _raw_input_embeddings(head_src, toks, scale)


@contextlib.contextmanager
def raw_context(draft):
    """Let the drafter take context vectors directly (skip fc + hidden_norm)."""
    fc, hn = draft.fc, draft.hidden_norm
    draft.fc, draft.hidden_norm = nn.Identity(), nn.Identity()
    try:
        yield
    finally:
        draft.fc, draft.hidden_norm = fc, hn


def block_hidden(draft, head_src, feats, ids, anchors, bs):
    """Drafter hidden states [n, bs-1, H] of fresh (lag 0) blocks at the given anchors
    (rows = block positions s+1 .. s+bs-1)."""
    noise, pos, mask, starts = pack(draft, head_src, feats, ids, [(s, 0) for s in anchors], bs)
    hidden = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                   past_key_values=None, use_cache=False)[0]
    return torch.stack([hidden[st : st + bs - 1] for st in starts])


def predict_context(head, draft, head_src, hid, ids, s, bs):
    """c_hat for positions s .. s+bs-2 from the hidden rows of the block at anchor s."""
    toks = ids[s : s + bs - 1]
    dt = next(head.parameters()).dtype
    e = token_emb(draft, head_src, toks[None])[0].to(dt)
    return head(hid.to(dt), e, draft.hidden_norm)


def draft_logits_ctx(draft, head_src, ctx, ids, s, bs):
    """Logits [bs-1, V] for a block at anchor s given context vectors ctx [1, s, H]."""
    noise, pos, mask, starts = pack(draft, head_src, ctx, ids, [(s, 0)], bs)
    with raw_context(draft):
        hidden = draft(target_hidden=ctx, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                       past_key_values=None, use_cache=False)
    return draft.compute_logits(hidden[0, starts[0] : starts[0] + bs - 1], _output_head(head_src))


def feat_loss(pred, real):
    cos = F.cosine_similarity(pred.float(), real.float(), dim=-1)
    mse = (pred.float() - real.float()).pow(2).mean(-1) / real.float().pow(2).mean(-1).clamp_min(1e-6)
    return (1 - cos).mean() + mse.mean(), cos


def save_head(head, path, hidden, mult):
    torch.save({"state": head.state_dict(), "hidden": hidden, "mult": mult}, path)


def load_head(path, device, dtype):
    d = torch.load(path, map_location="cpu")
    h = FeatHead(d["hidden"], d["mult"])
    h.load_state_dict(d["state"])
    return h.to(device=device, dtype=dtype).eval()
