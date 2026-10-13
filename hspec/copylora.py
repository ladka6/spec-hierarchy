"""LoRA for a cheap copy of the target (exp55): low-rank updates on the decoder-layer linears.

The copy (simulated low-bit or layer-skipped, hspec.models.load_mid) is trained so that it serves speculation:
the drafter conditioned on the copy's features should behave as with the target's real features, and the copy's
next token should match the target's. Saved as {dir}/meta.json + {dir}/lora.pt; load_mid("lora:{dir}") rebuilds
the base copy and merges the update (quality evaluation; a deployment would keep the adapter unmerged).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.A = torch.nn.Parameter(torch.randn(rank, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(x.dtype)


def trainable_layers(model) -> list[int]:
    """Decoder layers that are not skipped (a skipped layer has zeroed o_proj and down_proj)."""
    out = []
    for i, layer in enumerate(model.model.layers):
        if layer.self_attn.o_proj.weight.abs().sum() > 0 or layer.mlp.down_proj.weight.abs().sum() > 0:
            out.append(i)
    return out


def add_lora(model, rank: int = 16, alpha: float = 32.0, layers: list[int] | None = None) -> dict[str, LoRALinear]:
    layers = trainable_layers(model) if layers is None else layers
    mods = {}
    for i in layers:
        layer = model.model.layers[i]
        for parent, names in ((layer.self_attn, LINEARS[:4]), (layer.mlp, LINEARS[4:])):
            for nm in names:
                lo = LoRALinear(getattr(parent, nm), rank, alpha)
                setattr(parent, nm, lo)
                mods[f"{i}.{nm}"] = lo
    return mods


def save_lora(out: str | Path, mods: dict, meta: dict):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: (m.A.detach().cpu(), m.B.detach().cpu()) for k, m in mods.items()}, out / "lora.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


@torch.no_grad()
def merge_lora_(model, path: str | Path):
    """Add the saved low-rank updates into the copy's weights (in place)."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    scale = meta["alpha"] / meta["rank"]
    for key, (A, B) in torch.load(path / "lora.pt").items():
        i, nm = key.split(".")
        layer = model.model.layers[int(i)]
        lin = getattr(layer.self_attn if nm in LINEARS[:4] else layer.mlp, nm)
        w = lin.weight
        w.add_((B.to(w.device).float() @ A.to(w.device).float() * scale).to(w.dtype))
    return meta
