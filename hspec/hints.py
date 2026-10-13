"""Copy features as gated hints at the drafter's MASK positions (DFlow-style relay, 2609.06498, from a cheap copy).

In a refresh restart at anchor a, the drafter gets context features for positions < a (real ones before the round's
anchor s, the copy's after it) and predicts a+1 .. a+bs-1 from [token_a, MASK x (bs-1)]. The copy has also computed
features for positions a .. s+bs-1, on the round-1 drafted tokens, which the restart treats as uncertain (conf
restarts) or wrong (pc restart: the token at a was replaced). Those features are thrown away today.

Here they enter as hints: the input at MASK position p becomes
    E_mask + beta * LN(h) + gamma,    h = hidden_norm(fc(f_copy[p-1]))   (the drafter's own feature projection)
    [beta, gamma] = MLP([h, emb(token the copy saw at p-1), emb(anchor token), onehot(p - a)])
The MLP's last layer starts at zero, so the untrained module is exactly the plain restart. The drafter stays frozen;
the MLP learns how far to trust an approximate feature computed on a possibly wrong prefix.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dflash.model import _output_head
from hspec.lagtrain import pack


class HintMLP(torch.nn.Module):
    def __init__(self, hidden: int, bs: int, width: int = 1024):
        super().__init__()
        self.bs = bs
        self.inp = torch.nn.Linear(3 * hidden + bs, width)
        self.out = torch.nn.Linear(width, 2 * hidden)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, h, e_seen, e_anchor, offset):            # [N, H] x3, offset [N] in 1 .. bs-1
        x = torch.cat([h, e_seen, e_anchor, F.one_hot(offset, self.bs).float()], -1)
        beta, gamma = self.out(F.silu(self.inp(x))).chunk(2, -1)
        return beta * F.layer_norm(h, h.shape[-1:]) + gamma


def hinted_logits(draft, head_src, feats, ids, anchors, bs, hint=None, cfeat=None, seen=None, last=None):
    """Logits [n, bs-1, V] for restart blocks at anchors (lag 0) over context feats [1, T, W] and ids (1-D).

    hint: HintMLP or None (plain restart). cfeat [1, P, W]: copy features for positions first .. first+P-1 with
    first = last - P + 1; seen: the tokens the copy saw at those positions (1-D, same length); last: last position with
    a copy feature. MASK position p of a block gets a hint if p-1 <= last."""
    noise, pos, mask, starts = pack(draft, head_src, feats, ids, [(a, 0) for a in anchors], bs)
    if hint is not None:
        first = last - cfeat.shape[1] + 1
        rows, src, anc, off = [], [], [], []
        for st, a in zip(starts, anchors):
            for i in range(bs - 1):
                p = a + 1 + i
                if p - 1 > last:
                    break
                rows.append(st + i)
                src.append(p - 1 - first)
                anc.append(int(ids[a]))
                off.append(i + 1)
        if rows:
            dev = noise.device
            src_t = torch.tensor(src, device=dev)
            h = draft.hidden_norm(draft.fc(cfeat[0, src_t])).float()
            emb = head_src.get_input_embeddings()
            e_seen = emb(seen[src_t]).float()
            e_anc = emb(torch.tensor(anc, device=dev)).float()
            add = hint(h, e_seen, e_anc, torch.tensor(off, device=dev))
            noise = noise.clone()
            noise[0, torch.tensor(rows, device=dev)] += add.to(noise.dtype)
    hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                past_key_values=None, use_cache=False)[0]
    return draft.compute_logits(torch.stack([hid[st : st + bs - 1] for st in starts]), _output_head(head_src))
