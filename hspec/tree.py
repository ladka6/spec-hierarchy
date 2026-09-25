"""Tree-shaped verification on the target (shared by DDTree and three-stage branching).

A tree is rooted at the anchor token (already chosen, not yet processed by the model).
Nodes are listed so that every parent comes before its children. The target processes
[root, node_0, node_1, ...] in one forward pass: node positions are anchor_pos + depth and
each node attends to the cached prefix, the root, its ancestors and itself.

After the greedy walk the KV cache is compacted: the rows of the accepted path are kept
and the rest dropped. A node's keys/values depend only on its ancestors, so this is exact.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import torch


@dataclass
class Tree:
    tokens: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)   # -1 = root
    depths: list[int] = field(default_factory=list)    # root = 0, its children = 1

    def __len__(self) -> int:
        return len(self.tokens)

    def add(self, token: int, parent: int) -> int:
        self.tokens.append(int(token))
        self.parents.append(parent)
        self.depths.append(1 if parent < 0 else self.depths[parent] + 1)
        return len(self.tokens) - 1

    def children(self) -> dict[int, dict[int, int]]:
        ch: dict[int, dict[int, int]] = {-1: {}}
        for i, (t, p) in enumerate(zip(self.tokens, self.parents)):
            ch.setdefault(p, {})[t] = i
            ch.setdefault(i, {})
        return ch


def chain(tokens) -> Tree:
    tree = Tree()
    parent = -1
    for t in tokens:
        parent = tree.add(int(t), parent)
    return tree


def tree_mask(tree: Tree, past_len: int, device, dtype=torch.bool) -> torch.Tensor:
    """[1, 1, q, past_len + q] mask, q = 1 + len(tree). True = may attend."""
    q = 1 + len(tree)
    allowed = torch.zeros((q, past_len + q), dtype=torch.bool)
    allowed[:, :past_len] = True
    anc: list[list[int]] = [[0]]                       # row 0 = root
    for i, p in enumerate(tree.parents):
        anc.append(anc[p + 1] + [i + 1])
    for r, cols in enumerate(anc):
        allowed[r, [past_len + c for c in cols]] = True
    allowed = allowed.to(device)
    if dtype == torch.bool:
        return allowed[None, None]
    m = torch.zeros(allowed.shape, dtype=dtype, device=device)
    m.masked_fill_(~allowed, torch.finfo(dtype).min)
    return m[None, None]


def tree_inputs(root_token, tree: Tree, anchor_pos: int, device):
    ids = torch.tensor([[int(root_token)] + tree.tokens], dtype=torch.long, device=device)
    pos = torch.tensor([[anchor_pos] + [anchor_pos + d for d in tree.depths]], device=device)
    return ids, pos


@torch.inference_mode()
def verify_tree(model, cache, root_token, tree: Tree, anchor_pos: int, *, hidden=False,
                mask_dtype=torch.bool):
    """One forward over [root] + tree. The cache must hold exactly [0, anchor_pos)."""
    device = next(model.parameters()).device
    ids, pos = tree_inputs(root_token, tree, anchor_pos, device)
    mask = tree_mask(tree, anchor_pos, device, mask_dtype)
    return model(ids, position_ids=pos, attention_mask=mask, past_key_values=cache,
                 use_cache=True, output_hidden_states=hidden)


def compact(cache, keep: int, rows) -> None:
    """Cache holds [0, keep) followed by the rows of a tree forward. Keep [0, keep) plus the
    given tree rows (0 = root), in that order."""
    rows = list(rows)
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", True) or layer.keys.numel() == 0:
            continue
        idx = torch.tensor(list(range(keep)) + [keep + r for r in rows], device=layer.keys.device)
        layer.keys = layer.keys.index_select(-2, idx)
        layer.values = layer.values.index_select(-2, idx)


def greedy_walk(tree: Tree, logits: torch.Tensor, start: int = -1) -> tuple[list[int], int]:
    """logits: [1, 1 + len(tree), V]. Walks from node ``start`` (-1 = root).
    Returns (accepted node indices below start, bonus token)."""
    ch = tree.children()
    pred = logits[0].argmax(dim=-1).tolist()
    path: list[int] = []
    cur, row = start, start + 1
    while True:
        t = pred[row]
        nxt = ch.get(cur, {}).get(t)
        if nxt is None:
            return path, t
        path.append(nxt)
        cur, row = nxt, nxt + 1


def build_ddtree(logits: torch.Tensor, budget: int, top_k: int = 16) -> Tree:
    """Best-first draft tree from per-position drafter logits [L, V] (DDTree, Alg. 1).

    Returns the `budget` most probable prefixes under the factorized drafter distribution;
    they always form a valid (prefix-closed) tree."""
    L = logits.shape[0]
    k = min(top_k, budget, logits.shape[-1])
    vals, ids = torch.log_softmax(logits.float(), dim=-1).topk(k, dim=-1)
    logq = vals.cpu().tolist()
    tok = ids.cpu().tolist()
    tree = Tree()
    node_of: dict[tuple, int] = {}
    heap = [(-logq[0][0], (0,))]
    while heap and len(tree) < budget:
        neg, rho = heapq.heappop(heap)
        d = len(rho)
        parent = node_of[rho[:-1]] if d > 1 else -1
        node_of[rho] = tree.add(tok[d - 1][rho[-1]], parent)
        score = -neg
        if rho[-1] + 1 < k:
            s = score - logq[d - 1][rho[-1]] + logq[d - 1][rho[-1] + 1]
            heapq.heappush(heap, (-s, rho[:-1] + (rho[-1] + 1,)))
        if d < L:
            heapq.heappush(heap, (-(score + logq[d][0]), rho + (0,)))
    return tree


def expected_accept(tree: Tree, logits: torch.Tensor) -> float:
    """Surrogate expected acceptance under the drafter's factorized distribution (for logs)."""
    lp = torch.log_softmax(logits.float(), dim=-1).cpu()
    mass = []
    for i, (t, p) in enumerate(zip(tree.tokens, tree.parents)):
        d = tree.depths[i]
        mass.append((mass[p] if p >= 0 else 0.0) + float(lp[d - 1, t]))
    return sum(math.exp(m) for m in mass)
