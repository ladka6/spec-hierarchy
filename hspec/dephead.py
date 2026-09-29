"""Dependency head for a block-diffusion drafter (idea B).

DFlash predicts every position of a block independently. A dependency head (TreeFlash-style
one-token lookback) corrects the drafter's hidden state at depth d+1 with the embedding of the
token chosen at depth d:

    h'_{d+1} = h_{d+1} + W_down( silu(W_gate [n(h), n(e_d)]) * W_up [n(h), n(e_d)] )

(zero-initialised W_down, so an untrained head is the plain drafter). The drafter backbone
runs once per block as usual; the head runs on the few candidate tokens per depth, in one
batch, and a tree is built from the resulting conditional (first-order) distributions.

Two training regimes (scripts/train_dephead.py):
  tf     teacher forcing: prior token = the true token, target = the target model's
         distribution after the true prefix (what TreeFlash does)
  tree   on-policy: prior tokens = the nodes of the tree the drafter actually proposes
         (DDTree over the backbone marginals), target = the target model's distribution after
         each node's path, from one tree-masked target forward per block; the true path is
         added too. This trains the head on the contexts it meets at inference, including
         wrong prefixes, which teacher forcing never shows it.
"""

from __future__ import annotations

import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F

from dflash.model import _output_head


class DepHead(nn.Module):
    def __init__(self, hidden: int, inter: int = 2048):
        super().__init__()
        self.hidden, self.inter = hidden, inter
        self.norm_h = nn.RMSNorm(hidden, eps=1e-6)
        self.norm_e = nn.RMSNorm(hidden, eps=1e-6)
        self.gate = nn.Linear(2 * hidden, inter, bias=False)
        self.up = nn.Linear(2 * hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, h, e):
        x = torch.cat([self.norm_h(h), self.norm_e(e)], dim=-1)
        return h + self.down(F.silu(self.gate(x)) * self.up(x))


def head_logits(draft, head_src, dep, h, prior_tok):
    """h: [N, H] backbone hidden rows; prior_tok: [N] token ids -> logits [N, V]."""
    e = head_src.model.embed_tokens(prior_tok).to(h.dtype)
    hh = dep(h, e) if dep is not None else h
    return draft.compute_logits(hh, _output_head(head_src))


def save_dephead(dep, path):
    torch.save({"hidden": dep.hidden, "inter": dep.inter, "state": dep.state_dict()}, path)


def load_dephead(path, device, dtype):
    blob = torch.load(path, map_location=device)
    dep = DepHead(blob["hidden"], blob["inter"]).to(device=device, dtype=dtype)
    dep.load_state_dict(blob["state"])
    return dep.eval()


@torch.no_grad()
def dep_tree(draft, head_src, dep, hidden, lq_backbone, anchor_tok, budget, M):
    """Best-first tree from first-order conditionals.

    hidden: [D, H] backbone rows (row d predicts depth d+1); lq_backbone: [D, V] log-probs.
    Candidate tokens at depth d+1 = top-M of the backbone marginal there; the head scores
    each candidate child given its parent's token. Returns {path tuple: log-prob}."""
    D = hidden.shape[0]
    dev = hidden.device
    cand = [lq_backbone[d].topk(M).indices for d in range(D)]            # children at depth d+1
    priors = [torch.tensor([anchor_tok], device=dev)] + [cand[d] for d in range(D - 1)]
    rows_h = torch.cat([hidden[d : d + 1].expand(len(priors[d]), -1) for d in range(D)])
    rows_t = torch.cat(priors)
    lg = head_logits(draft, head_src, dep, rows_h, rows_t).float()
    lp = torch.log_softmax(lg, -1)
    cond, r = [], 0
    for d in range(D):                                                   # cond[d][prior] -> sorted children
        table = {}
        for t in priors[d].tolist():
            sc = lp[r, cand[d]]
            order = torch.argsort(sc, descending=True)
            table[t] = [(int(cand[d][o]), float(sc[o])) for o in order.tolist()]
            r += 1
        cond.append(table)
    out = {}
    first = cond[0][int(anchor_tok)]
    heap = [(-first[0][1], (first[0][0],), 0, 0, 0.0)]                   # (-score, path, depth idx, rank, parent score)
    while heap and len(out) < budget:
        neg, path, d, rank, pscore = heapq.heappop(heap)
        out[path] = -neg
        parent_tok = int(anchor_tok) if d == 0 else path[-2]
        sibs = cond[d][parent_tok]
        if rank + 1 < len(sibs):
            c, s = sibs[rank + 1]
            heapq.heappush(heap, (-(pscore + s), path[:-1] + (c,), d, rank + 1, pscore))
        if d + 1 < D and path[-1] in cond[d + 1]:
            c, s = cond[d + 1][path[-1]][0]
            heapq.heappush(heap, (-(-neg + s), path + (c,), d + 1, 0, -neg))
    return out


def backbone_rows(draft, head_src, feats, ids, blocks, bs):
    """Drafter hidden rows [n_blocks, bs-1, H] for standard blocks at anchors `blocks` (lag 0),
    packed into one forward (hspec.lagtrain.pack)."""
    from hspec.lagtrain import pack

    noise, pos, mask, starts = pack(draft, head_src, feats, ids, [(s, 0) for s in blocks], bs)
    h = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
              past_key_values=None, use_cache=False)[0]
    return torch.stack([h[st : st + bs - 1] for st in starts])
