"""Early-exit heads on the target: the target's first k layers plus a head act as the middle
verifier (HiSpec-style), so the check runs on the same GPU and its layer-1..k KV is exactly
what the full target would compute for the accepted tokens.

kinds
  raw    target's final norm + LM head applied to the layer-k hidden state (logit lens)
  lin    h + W h (W zero-initialised, 4096x4096), own copy of the final norm, target LM head
  layer  a trainable copy of the target's last decoder layer on top of layer k, own norm,
         target LM head (Kangaroo-style adapter; costs about one extra layer)

k counts executed layers: the head reads hidden_states[k] of a HF forward with
output_hidden_states=True (index 0 is the embedding output).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExitHead(nn.Module):
    def __init__(self, target, kind: str, k: int):
        super().__init__()
        self.kind, self.k = kind, k
        self._t = [target]                       # not a submodule: frozen, not saved
        base = target.model
        if kind == "raw":
            self.norm = None
        else:
            self.norm = copy.deepcopy(base.norm)
        if kind == "lin":
            H = target.config.hidden_size
            p = next(target.parameters())
            self.proj = nn.Linear(H, H, bias=False, device=p.device, dtype=p.dtype)
            nn.init.zeros_(self.proj.weight)
        elif kind == "layer":
            self.layer = copy.deepcopy(base.layers[-1])
        elif kind != "raw":
            raise ValueError(kind)

    @property
    def target(self):
        return self._t[0]

    def hidden(self, h, position_ids=None):
        """h: [B, T, H] layer-k hidden states -> normalized hidden [B, T, H]."""
        base = self.target.model
        if self.kind == "raw":
            return base.norm(h)
        if self.kind == "lin":
            h = h + self.proj(h)
        else:
            if position_ids is None:
                position_ids = torch.arange(h.shape[1], device=h.device)[None]
            pe = base.rotary_emb(h, position_ids)
            h = self.layer(h, attention_mask=None, position_embeddings=pe, position_ids=position_ids)
            if isinstance(h, tuple):
                h = h[0]
        return self.norm(h)

    def logits(self, h, position_ids=None, rows=None):
        """rows: optional index of positions to return logits for (after the sequence pass)."""
        x = self.hidden(h, position_ids)
        if rows is not None:
            x = x[:, rows]
        return self.target.lm_head(x)

    def trainable(self):
        return [p for n, p in self.named_parameters() if not n.startswith("_t")]


def head_name(kind: str, k: int) -> str:
    return f"{kind}{k}"


def save_heads(heads: dict, path):
    torch.save({name: {"kind": h.kind, "k": h.k, "state": h.state_dict()} for name, h in heads.items()
                if h.kind != "raw"}, path)


def load_heads(target, path, names=None) -> dict:
    blob = torch.load(path, map_location=next(target.parameters()).device)
    out = {}
    for name, d in blob.items():
        if names and name not in names:
            continue
        h = ExitHead(target, d["kind"], d["k"])
        h.load_state_dict(d["state"])
        out[name] = h.eval()
    return out


def kl_loss(student_logits, teacher_logprobs):
    """KL(teacher || student), mean over rows."""
    ls = F.log_softmax(student_logits.float(), -1)
    return (teacher_logprobs.exp() * (teacher_logprobs - ls)).sum(-1).mean()


def block_outcome(d, t, e, t_bonus=None, e_bonus=None):
    """One drafted block checked by a middle verifier, on a target-greedy trajectory.

    d: drafted tokens [n]; t: target tokens at those positions [n]; e: the verifier's greedy
    predictions there, computed teacher-forced on the target trajectory (valid while the draft
    matches the target, which is the only region where they matter).
    Returns (delivered, leak, perfect):
      delivered  correct tokens the verifier hands to the target (accepted draft + its bonus)
      leak       1 if it hands the target a wrong token (false accept or wrong correction)
      perfect    what a verifier identical to the target would deliver (acc + 1)
    """
    n = len(d)
    acc = 0
    while acc < n and d[acc] == t[acc]:
        acc += 1
    for j in range(acc):
        if e[j] != d[j]:                     # false reject: its correction e_j != t_j
            return j, 1, acc + 1
    if acc == n:
        if t_bonus is None:
            return n, 0, n
        ok = int(e_bonus == t_bonus)
        return n + ok, 1 - ok, n + 1
    if e[acc] == d[acc]:                      # false accept of the target's first mismatch
        return acc, 1, acc + 1
    ok = int(e[acc] == t[acc])
    return acc + ok, 1 - ok, acc + 1
