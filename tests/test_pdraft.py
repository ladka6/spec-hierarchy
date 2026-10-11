"""CPU checks for PDraft (tiny random Qwen3): manual layers == HF layers, kv mode at init == no features,
gradients reach the injection path once the gate opens, and block rows never see their own future features."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.pdraft import PDraft, block_ce  # noqa: E402


def tiny():
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


def main():
    torch.manual_seed(0)
    small = tiny()
    T, B, D = 40, 8, 48
    ids = torch.randint(0, 97, (T,))
    feats = torch.randn(T, D)
    anc = [5, 17, 30]

    # 1. manual layer path == HF decoder layers (pure LM, no blocks: context rows == HF causal forward)
    m = PDraft(small, D, use_feats=False).eval()
    with torch.no_grad():
        h = m.embed(ids)[None]
        pos = torch.arange(T)[None]
        pe = m.rotary(h, pos)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        a = h
        for li, layer in enumerate(m.layers):
            a = m._layer(li, layer, a, pe, mask, None, None, None)
        ref = small(ids[None], output_hidden_states=True).hidden_states[-1]
        err = (m.norm(a) - ref).abs().max().item()
    print(f"manual vs HF layers max err {err:.2e}")
    assert err < 1e-4

    # 2. kv mode at init (gates 0) == no-feature model
    torch.manual_seed(1)
    kv = PDraft(small, D, feat_mode="kv").eval()
    with torch.no_grad():
        d = (kv(ids, feats, anc, B) - m(ids, feats, anc, B)).abs().max().item()
    print(f"kv at init vs nofeat max diff {d:.2e}")
    assert d < 1e-4

    # 3. gradient flow: gates get grad at init; fc gets grad once a gate is open
    kv.train()
    loss, _, _ = block_ce(kv(ids, feats, anc, B), ids, anc)
    loss.backward()
    assert kv.gates.grad.abs().sum() > 0, "gates got no gradient"
    kv.zero_grad()
    with torch.no_grad():
        kv.gates.fill_(0.5)
    loss, _, _ = block_ce(kv(ids, feats, anc, B), ids, anc)
    loss.backward()
    assert kv.fc.weight.grad.abs().sum() > 0, "fc got no gradient"
    print("gradients reach gates and fc")

    # 4. no leakage: block at anchor s must not depend on features at positions >= s
    kv.eval()
    with torch.no_grad():
        f2 = feats.clone()
        f2[anc[0]:] += 5 * torch.randn_like(f2[anc[0]:])
        o1 = kv(ids, feats, [anc[0]], B)
        o2 = kv(ids, f2, [anc[0]], B)
        leak = (o1 - o2).abs().max().item()
        f3 = feats.clone()
        f3[: anc[0]] += 5 * torch.randn_like(f3[: anc[0]])
        used = (o1 - kv(ids, f3, [anc[0]], B)).abs().max().item()
    print(f"future-feature leak {leak:.2e}, past-feature effect {used:.2e}")
    assert leak < 1e-5 and used > 1e-3
    print("OK")


if __name__ == "__main__":
    main()
