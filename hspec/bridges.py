"""Bridged copy of the target: keep some decoder layers, replace each removed span by a small trained module.

The copy is the target model itself with patched layer forwards:
  - every removed layer returns its input unchanged (identity skip),
  - the LAST removed layer of a span adds bridge(h) to that input, so the hidden state entering the next kept layer
    is h_in + bridge(h_in): the bridge learns what the whole removed span did.
The outputs of the kept layers (and so the features the DFlash drafter reads at its tap layers) are produced by the
real layers. With BRIDGE["on"] = False the hooks pass everything through and the model is the full target again, so
teacher and student share one copy of the weights during training.

Every bridge has an affine part h -> h (I + A) + b (fp32, starts at zero), the form of Ghosted Layers (2605.15491)
and SHIFT-LLM (2608.25068), which can be fitted in closed form (fit_linear_: ridge regression of the span's residual
update on its input, from target activations). On top of it, optional nonlinear parts trained end to end:
  lin   affine part only (training-free if only fitted)
  mlp   affine part, then RMSNorm -> SwiGLU MLP (H -> r -> H), last projection zero-initialised
  attn  affine part, then a small causal self-attention (n heads x d, zero-init output), then the MLP
The attn kind attends over the tokens of the current call only: valid when the copy recomputes the whole prefix
(exp27, kv=own), not with a shared prefix KV cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

BRIDGE = {"on": True}


def spans_of(kept: list[int], n_layers: int) -> list[tuple[int, int]]:
    """Consecutive runs of removed layers (first, last), within 0 .. n_layers-1."""
    keep = set(kept)
    out, start = [], None
    for i in range(n_layers):
        if i not in keep and start is None:
            start = i
        if i in keep and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, n_layers - 1))
    return out


class RMSNorm(torch.nn.Module):
    def __init__(self, h, eps=1e-6):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(h))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.w).to(x.dtype)


class Bridge(torch.nn.Module):
    def __init__(self, hidden: int, kind: str = "mlp", rank: int = 2048, heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.kind = kind
        self.lin = torch.nn.Linear(hidden, hidden, bias=True)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)
        if kind == "lin":
            return
        self.n1 = RMSNorm(hidden)
        self.gate = torch.nn.Linear(hidden, rank, bias=False)
        self.up = torch.nn.Linear(hidden, rank, bias=False)
        self.down = torch.nn.Linear(rank, hidden, bias=False)
        torch.nn.init.zeros_(self.down.weight)
        if kind == "attn":
            self.n0 = RMSNorm(hidden)
            self.h, self.d = heads, head_dim
            self.qkv = torch.nn.Linear(hidden, 3 * heads * head_dim, bias=False)
            self.o = torch.nn.Linear(heads * head_dim, hidden, bias=False)
            torch.nn.init.zeros_(self.o.weight)

    def forward(self, x):                                   # x [B, T, H]
        dt = x.dtype
        x = x.float()
        x = x + self.lin(x)
        if self.kind == "lin":
            return x.to(dt)
        if self.kind == "attn":
            B, T, _ = x.shape
            q, k, v = self.qkv(self.n0(x)).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.o(a.transpose(1, 2).reshape(B, T, self.h * self.d))
        y = self.n1(x)
        x = x + self.down(F.silu(self.gate(y)) * self.up(y))
        return x.to(dt)


def attach_bridges(model, kept: list[int], kind: str = "mlp", rank: int = 2048) -> torch.nn.ModuleDict:
    """Install skip hooks and bridges on a decoder-only HF model (model.model.layers). Returns the bridges."""
    layers = model.model.layers
    H = model.config.hidden_size
    dev = next(model.parameters()).device
    bridges = torch.nn.ModuleDict()
    for a, b in spans_of(kept, len(layers)):
        bridges[str(b)] = Bridge(H, kind, rank).to(dev)          # fp32 parameters
    end_of = {int(k): m for k, m in bridges.items()}

    def make_forward(i, orig):
        def forward(hidden_states, *args, **kwargs):
            if not BRIDGE["on"]:
                return orig(hidden_states, *args, **kwargs)
            return end_of[i](hidden_states) if i in end_of else hidden_states    # skipped: no compute at all
        return forward

    removed = {i for a, b in spans_of(kept, len(layers)) for i in range(a, b + 1)}
    for i in sorted(removed):
        layers[i].forward = make_forward(i, layers[i].forward)
    model.bridges = bridges
    model.kept_layers = sorted(kept)
    return bridges


@torch.no_grad()
def fit_linear_(model, bridges, seqs, ridge: float = 1e-3, device="cuda"):
    """Closed-form affine part of every bridge: ridge regression of the span's update hs[b+1] - hs[a] on hs[a]
    (target activations, bridges off), each span fitted independently. Rows are weighted by 1 / mean(x^2) so the
    attention-sink tokens with huge norms do not dominate. seqs: list of 1-D id tensors."""
    spans = spans_of(model.kept_layers, len(model.model.layers))
    H = model.config.hidden_size
    acc = {b: (torch.zeros(H + 1, H + 1, dtype=torch.float64, device=device),
               torch.zeros(H + 1, H, dtype=torch.float64, device=device)) for _, b in spans}
    BRIDGE["on"] = False
    try:
        for ids in seqs:
            hs = model(ids.to(device)[None], output_hidden_states=True, logits_to_keep=1).hidden_states
            for a, b in spans:
                x = hs[a][0].double()
                d = hs[b + 1][0].double() - x
                w = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                xa = torch.cat([x, torch.ones_like(x[:, :1])], -1) * w
                xtx, xtd = acc[b]
                xtx += xa.T @ xa
                xtd += xa.T @ (d * w)
    finally:
        BRIDGE["on"] = True
    for a, b in spans:
        xtx, xtd = acc[b]
        lam = ridge * xtx.diagonal().mean()
        sol = torch.linalg.solve(xtx + lam * torch.eye(H + 1, dtype=xtx.dtype, device=xtx.device), xtd)
        m = bridges[str(b)]
        m.lin.weight.copy_(sol[:H].T.float())
        m.lin.bias.copy_(sol[H].float())


def save_bridges(out: str | Path, bridges, meta: dict):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.state_dict() for k, v in bridges.items()}, out / "bridges.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load_bridged(model, path: str | Path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    bridges = attach_bridges(model, meta["kept"], meta["kind"], meta["rank"])
    sd = torch.load(path / "bridges.pt", map_location=next(model.parameters()).device)
    for k, m in bridges.items():
        m.load_state_dict(sd[k])
    bridges.eval()
    return model
