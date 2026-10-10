"""Small same-family model as the predictor of the verifier's frontier states (proposal 1).

A frozen small model (e.g. Qwen3-0.6B, same tokenizer) runs over the sequence; a mapper turns its
hidden states at a few layers into the target's DFlash feature vector (target layers 1, 9, 17, 25,
33 concatenated). The mapper is a linear map initialised from the closed-form ridge fit plus a
zero-initialised residual MLP, so before training it is exactly the ridge map of exp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dflash.model import extract_context_feature


def small_layers(n_layers: int, n_sel: int) -> list[int]:
    """hidden_states indices (1 .. n_layers) spread evenly, last layer included."""
    return sorted({round(1 + i * (n_layers - 1) / max(n_sel - 1, 1)) for i in range(n_sel)})


def small_inputs(small, ids, sel):
    """(regressors [T, Dx] float32, the small model's greedy predictions [T])."""
    out = small(ids[None], output_hidden_states=True)
    x = torch.cat([out.hidden_states[i][0] for i in sel], -1).float()
    return x, out.logits[0].argmax(-1)


class SmallMapper(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 2048):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.mlp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return self.lin(x) + self.mlp(x)


@torch.no_grad()
def ridge_fit(small, target, lids, sel, seqs, device, alphas=(1e-3, 1e-2, 1e-1), log=print):
    """Closed-form ridge map from small-model regressors to target features (exp32), lambda chosen
    on every 10th sequence. Returns (W [Dx, Dy], b [Dy]) in float32."""
    Dx = small.config.hidden_size * len(sel) + 1
    Dy = len(lids) * target.config.hidden_size
    XtX = torch.zeros(Dx, Dx, dtype=torch.float64, device=device)
    XtY = torch.zeros(Dx, Dy, dtype=torch.float64, device=device)
    hold = []
    for i, d in enumerate(seqs):
        ids = d["ids"].long().to(device)
        real = extract_context_feature(target(ids[None], output_hidden_states=True).hidden_states, lids)[0].double()
        x, _ = small_inputs(small, ids, sel)
        x = torch.cat([x.double(), torch.ones_like(x[:, :1], dtype=torch.float64)], -1)
        r = slice(max(d["n_prompt"], 4), len(ids))
        if i % 10 == 0 and len(hold) < 20:
            hold.append((x[r].clone(), real[r].clone()))
            continue
        XtX += x[r].T @ x[r]
        XtY += x[r].T @ real[r]
    scale = torch.trace(XtX) / Dx
    best = None
    for a in alphas:
        W = torch.linalg.solve(XtX + a * scale * torch.eye(Dx, dtype=XtX.dtype, device=device), XtY)
        c = sum(torch.nn.functional.cosine_similarity(xh @ W, yh, dim=-1).mean().item() for xh, yh in hold) / len(hold)
        log(f"ridge alpha {a:g}: held-out cosine {c:.4f}")
        if best is None or c > best[0]:
            best = (c, W)
    W = best[1].float()
    return W[:-1], W[-1]
