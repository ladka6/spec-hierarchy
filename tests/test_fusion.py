"""CPU checks for the fusion ablation (tiny random models).

  kv mode reproduces DFlash's own packed forward exactly; kv_fuse equals kv at init (zero maps);
  every mode gives finite loss and gets gradients into its fusion maps / drafter.

  python tests/test_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_cpu import V, tiny_draft, tiny_target  # noqa: E402

from dflash.model import extract_context_feature  # noqa: E402

from hspec.fusion import MODES, Fuser, accepted, block_forward, block_loss  # noqa: E402
from hspec.lagtrain import pack  # noqa: E402


def main():
    torch.manual_seed(0)
    target = tiny_target()
    draft = tiny_draft(8)
    bs = draft.block_size
    ids = torch.randint(0, V - 3, (40,))
    feats = extract_context_feature(target(ids[None], output_hidden_states=True).hidden_states,
                                    draft.target_layer_ids).detach()
    anchors = [5, 12, 20, 30]
    H = draft.config.hidden_size
    fails = 0
    with torch.no_grad():
        noise, pos, mask, starts = pack(draft, target, feats, ids, [(s, 0) for s in anchors], bs)
        ref = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask)[0]
        for mode in ("kv", "kv_fuse"):
            h, _ = block_forward(draft, Fuser(H, len(draft.layers), mode), target, feats, ids, anchors, bs)
            ok = torch.allclose(h, ref, atol=1e-5)
            fails += not ok
            print(f"{mode} at init == DFlash forward: {ok}")
        # no-context modes: a block must not depend on other blocks or on the context features
        fu = Fuser(H, len(draft.layers), "fuse_all")
        for p in fu.proj:
            torch.nn.init.normal_(p.weight, std=0.02)
        h1, _ = block_forward(draft, fu, target, feats, ids, anchors, bs)
        f2 = feats.clone()
        f2[:, :4] += 1.0                           # context only (anchor features are at s-1 >= 4)
        h2, _ = block_forward(draft, fu, target, f2, ids, anchors, bs)
        ok = torch.allclose(h1, h2, atol=1e-5)
        fails += not ok
        print(f"fuse_all ignores context features other than the anchor's: {ok}")
        h3, _ = block_forward(draft, fu, target, feats, ids, anchors[:1], bs)
        ok = torch.allclose(h1[:bs], h3, atol=1e-5)
        fails += not ok
        print(f"fuse_all blocks are independent of each other: {ok}")
        f4 = feats.clone()
        f4[:, anchors[0] - 1] += 1.0
        h4, _ = block_forward(draft, fu, target, f4, ids, anchors, bs)
        ok = not torch.allclose(h1[:bs], h4[:bs], atol=1e-4)
        fails += not ok
        print(f"fuse_all uses the anchor feature: {ok}")
    for mode in MODES:
        fu = Fuser(H, len(draft.layers), mode)
        draft.zero_grad()
        loss, correct = block_loss(draft, fu, target, feats, ids, anchors, bs)
        loss.backward()
        g = [p.grad for p in fu.parameters()]
        ok = torch.isfinite(loss).item() and all(x is not None and x.abs().sum() > 0 for x in g) \
            and draft.layers[0].mlp.down_proj.weight.grad is not None
        fails += not ok
        print(f"{mode}: loss {float(loss):.3f}, {len(g)} fusion maps with grad, "
              f"tau {sum(accepted(r) for r in correct) / len(correct):.2f}: {ok}")
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(int(fails > 0))


if __name__ == "__main__":
    main()
