"""The target's own first k layers as the feature source for the drafter (option 2).

Instead of a separate low-bit copy, the refresh pass runs only the target's first k layers over
the drafted tokens (exact, bf16, the target's own weights). A small trained adapter turns the
output after k layers into what the drafter reads:

  * the feature slices of target layers that were not executed (DFlash reads hidden_states[id+1]
    for ids [1, 9, 17, 25, 33]; slices with id+1 <= k are exact and copied through)
  * a stand-in for the final hidden state, giving next-token predictions through the target's
    norm + lm_head (the correction in the copy-corrected restart)

Adapter: n trainable decoder layers initialised from target layers k .. k+n-1 (n = 0: a per-
position MLP), then per-output linear heads added to the residual stream (zero-initialised, so
an untrained adapter returns h_k for every missing slice). Adapter weights are fp32.

At inference the k-layer activations of drafted tokens that the target later accepts are the
same as the verify pass would compute, so the verify can reuse them (not implemented here).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ShallowAdapter(nn.Module):
    def __init__(self, target, layer_ids, exit_layer: int, n_layers: int = 2, mlp_mult: int = 2):
        super().__init__()
        cfg = target.config
        H = cfg.hidden_size
        self.k = exit_layer
        self.layer_ids = list(layer_ids)
        self.missing = [j for j, lid in enumerate(self.layer_ids) if lid + 1 > exit_layer]
        self.n_layers = n_layers
        if n_layers:
            self.layers = nn.ModuleList(copy.deepcopy(target.model.layers[exit_layer + i]).float()
                                        for i in range(n_layers))
            for p in self.layers.parameters():
                p.requires_grad_(True)
            self.trunk = None
        else:
            self.layers = None
            self.trunk = nn.Sequential(nn.RMSNorm(H, eps=cfg.rms_norm_eps), nn.Linear(H, mlp_mult * H),
                                       nn.SiLU(), nn.Linear(mlp_mult * H, H))
        self.heads = nn.ModuleList(nn.Linear(H, H, bias=False) for _ in range(len(self.missing) + 1))
        self.head_norm = nn.RMSNorm(H, eps=cfg.rms_norm_eps)
        for h in self.heads:
            nn.init.zeros_(h.weight)

    def forward(self, h_k: torch.Tensor, rotary, position_ids: torch.Tensor):
        """h_k [B, T, H] (output after k target layers, full causal sequence).
        Returns (missing slices [B, T, H] list, final pre-norm hidden [B, T, H]), fp32."""
        x = h_k.float()
        if self.layers is not None:
            pe = rotary(x, position_ids)
            z = x
            for layer in self.layers:
                z = layer(z, attention_mask=None, position_embeddings=pe, position_ids=position_ids)
        else:
            z = x + self.trunk(x)
        z = self.head_norm(z)
        outs = [x + h(z) for h in self.heads]
        return outs[:-1], outs[-1]


def assemble(hidden_states, layer_ids, k: int, missing_pred) -> torch.Tensor:
    """DFlash feature vector [B, T, W]: exact slices for executed layers, predicted otherwise."""
    it = iter(missing_pred)
    parts = []
    for lid in layer_ids:
        parts.append(hidden_states[lid + 1].float() if lid + 1 <= k else next(it))
    return torch.cat(parts, -1)


# ---------------------------------------------------------------------------
# Training / evaluation on depth-lagged blocks: a block at anchor s with lag g sees the real
# features for [0, s-g) and the alternative (shallow + adapter) features for [s-g, s), which is
# what the restart sees at inference (real features up to the last verify, refreshed ones over
# the drafted tokens since).
# ---------------------------------------------------------------------------

import torch.nn.functional as F  # noqa: E402

from dflash.model import _draft_value, _output_head, _raw_input_embeddings  # noqa: E402


def pack_alt(draft, head_src, real, alt, ids, blocks, bs):
    """Each block (s, g) gets its own copy of alt[:, s-g:s] after the shared real context.
    Returns (ctx, noise, position_ids, mask, starts)."""
    T = real.shape[1]
    dev = ids.device
    parts, ppos, pspans = [], [], []
    c = T
    for s, g in blocks:
        if g > 0:
            parts.append(alt[:, s - g : s])
            ppos.append(torch.arange(s - g, s, device=dev))
        pspans.append((c, c + g))
        c += g
    ctx = torch.cat([real] + [p.to(real.dtype) for p in parts], 1) if parts else real
    C = ctx.shape[1]
    toks, qpos = [], []
    for s, _g in blocks:
        t = torch.full((bs,), draft.mask_token_id, dtype=torch.long, device=dev)
        t[0] = ids[s]
        toks.append(t)
        qpos.append(torch.arange(s, s + bs, device=dev))
    q = bs * len(blocks)
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    noise = _raw_input_embeddings(head_src, torch.cat(toks)[None], scale)
    pos = torch.cat([torch.arange(T, device=dev)] + ppos + qpos)[None]
    mask = torch.zeros((q, C + q), dtype=torch.bool, device=dev)
    for b, ((s, g), (pa, pb)) in enumerate(zip(blocks, pspans)):
        rows = slice(b * bs, (b + 1) * bs)
        mask[rows, : s - g] = True
        mask[rows, pa:pb] = True
        mask[rows, C + b * bs : C + (b + 1) * bs] = True
    return ctx, noise, pos, mask[None, None], [b * bs + 1 for b in range(len(blocks))]


def drafter_block_loss(draft, head_src, real, alt, ids, blocks, bs, gamma=0.9):
    """Gamma-weighted CE of the drafter on depth-lagged blocks + per-block correctness rows."""
    T = ids.shape[0]
    ctx, noise, pos, mask, starts = pack_alt(draft, head_src, real, alt, ids, blocks, bs)
    ddt = next(draft.parameters()).dtype              # fp32 when the drafter is being trained
    hid = draft(target_hidden=ctx.to(ddt), noise_embedding=noise.to(ddt), position_ids=pos, attention_mask=mask,
                past_key_values=None, use_cache=False)[0].to(_output_head(head_src).weight.dtype)
    rows, labels, weights, where = [], [], [], []
    for b, ((s, _g), st) in enumerate(zip(blocks, starts)):
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
    logits = draft.compute_logits(hid[rows], _output_head(head_src)).float()
    loss = (F.cross_entropy(logits, labels, reduction="none") * w).sum() / w.sum()
    hit = (logits.argmax(-1) == labels).tolist()
    correct = [[None] * (bs - 1) for _ in blocks]
    for (b, k), h in zip(where, hit):
        correct[b][k] = h
    return loss, correct


def adapter_aux(adapter, target, missing_pred, final_pre, hidden_states, tgt_pred, rows):
    """Feature loss (1 - cosine to the real slices) and prediction loss (CE of norm + lm_head of
    the predicted final state against the target's own greedy token), on positions `rows`.
    Returns (feat_loss, pred_loss, agreement of the predicted token with the target's)."""
    fl = []
    for j, p in zip(adapter.missing, missing_pred):
        real = hidden_states[adapter.layer_ids[j] + 1][0, rows].float()
        fl.append(1 - F.cosine_similarity(p[0, rows], real, dim=-1).mean())
    feat = torch.stack(fl).mean() if fl else torch.zeros((), device=final_pre.device)
    h = target.model.norm(final_pre[0, rows].to(target.lm_head.weight.dtype))
    logits = target.lm_head(h).float()
    lab = tgt_pred[rows]
    pred = F.cross_entropy(logits, lab)
    agree = (logits.argmax(-1) == lab).float().mean()
    return feat, pred, agree
