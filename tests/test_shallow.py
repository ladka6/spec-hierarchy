"""CPU checks for the shallow-feature adapter (tiny random models).

  - exit = all layers: the adapter path gives the real features and the drafter output equals
    plain DFlash on the same blocks
  - untrained adapter (zero heads) returns h_k for every missing slice
  - gradients reach the adapter through the frozen drafter, the feature loss and the prediction loss

  python tests/test_shallow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_cpu import L, V, tiny_draft, tiny_target  # noqa: E402

from dflash.model import extract_context_feature  # noqa: E402

from hspec.fusion import Fuser, block_forward  # noqa: E402
from hspec.shallow import ShallowAdapter, adapter_aux, assemble, drafter_block_loss, pack_alt  # noqa: E402


def main():
    torch.manual_seed(0)
    target = tiny_target()
    draft = tiny_draft(8)
    target.requires_grad_(False)
    draft.requires_grad_(False)
    bs = draft.block_size
    lids = list(draft.target_layer_ids)
    ids = torch.randint(0, V - 3, (40,))
    out = target(ids[None], output_hidden_states=True)
    hs = tuple(h.detach() for h in out.hidden_states)
    real = extract_context_feature(hs, lids)
    tp = out.logits[0].argmax(-1)
    pos = torch.arange(40)[None]
    fails = 0
    print(f"tiny target: {L} layers, drafter reads {lids}")

    # 1. all layers executed: adapter features == real, drafter output == plain DFlash
    full = ShallowAdapter(target, lids, L, n_layers=0)
    miss, fin = full(hs[L], target.model.rotary_emb, pos)
    alt = assemble(hs, lids, L, miss)
    ok = len(miss) == 0 and torch.allclose(alt, real.float())
    blocks = [(10, 4), (20, 8), (30, 1)]
    with torch.no_grad():
        ctx, noise, p, mask, starts = pack_alt(draft, target, real, alt, ids, blocks, bs)
        h1 = draft(target_hidden=ctx, noise_embedding=noise, position_ids=p, attention_mask=mask)[0]
        h0, st0 = block_forward(draft, Fuser(draft.config.hidden_size, 2, "kv"), target, real, ids,
                                [s for s, _ in blocks], bs)
    ok = ok and torch.allclose(h1, h0, atol=1e-5)
    fails += not ok
    print(f"exit = all layers: adapter path == plain DFlash: {ok}")

    # 2. untrained adapter returns h_k for missing slices
    k = 2
    for n in (0, 1):
        ad = ShallowAdapter(target, lids, k, n_layers=n)
        miss, fin = ad(hs[k], target.model.rotary_emb, pos)
        ok = all(torch.allclose(m, hs[k].float()) for m in miss) and torch.allclose(fin, hs[k].float())
        ok = ok and len(miss) == sum(l + 1 > k for l in lids)
        fails += not ok
        print(f"untrained adapter (layers={n}) returns h_k for {len(miss)} missing slices: {ok}")

        # 3. gradients through the frozen drafter + aux losses
        for p_ in ad.heads:
            torch.nn.init.normal_(p_.weight, std=0.02)
        miss, fin = ad(hs[k], target.model.rotary_emb, pos)
        alt = assemble(hs, lids, k, miss)
        ld, correct = drafter_block_loss(draft, target, real, alt, ids, blocks, bs)
        rows = torch.tensor(sorted({t for s, g in blocks for t in range(s - g, s)}))
        lf, lp, ag = adapter_aux(ad, target, miss, fin, hs, tp, rows)
        (ld + lf + lp).backward()
        g = [p_.grad for p_ in ad.parameters() if p_.requires_grad]
        ok = all(x is not None for x in g) and all(torch.isfinite(t) for t in (ld, lf, lp)) \
            and sum(float(x.abs().sum()) for x in g) > 0 and draft.layers[0].mlp.down_proj.weight.grad is None
        fails += not ok
        print(f"layers={n}: losses draft {float(ld):.3f} feat {float(lf):.3f} pred {float(lp):.3f}, "
              f"grads on {len(g)} adapter tensors, drafter frozen: {ok}")
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(int(fails > 0))


if __name__ == "__main__":
    main()
