"""CPU test with tiny random models (no downloads): the offline block evaluation leaks nothing and equals the
drafter call DFlash makes at inference.

Every tau in the reports comes from hspec.lagtrain.pack: a block at anchor s with lag g reads the features of
positions [0, s - g), the tokens s - g .. s, and MASK slots. If the mask or the positions were wrong, a block could
see the tokens it is scored on, or the features of positions it is not supposed to have, and every number would be
too good. Checks:

  1. target features are causal: the feature of position p does not change when a later token changes
  2. a lagged block's logits do not change when (a) any token after the anchor changes, (b) any feature at or
     after s - g changes; and they DO change when the anchor or the last allowed feature changes (the test can fail)
  3. at lag 0 the packed, masked call equals the unmasked call dflash_generate makes (features [0, s), block at s)
  4. blocks packed together do not see each other: a block's logits are the same packed alone or with others
  5. pack_depth (extra rows per block): the same leak checks, and the extra rows of one block are invisible to others
  6. the accepted-length count

  python tests/test_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dflash.model import DFlashDraftModel, _output_head, _raw_input_embeddings, extract_context_feature  # noqa: E402

from hspec.lagtrain import deep_columns, draft_logits_lagged, pack, pack_depth  # noqa: E402

torch.manual_seed(0)
V, H, L, BS = 96, 64, 6, 8
TOL = 1e-4


def tiny_target():
    cfg = Qwen3Config(vocab_size=V, hidden_size=H, intermediate_size=128, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=1024, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval()


def tiny_draft():
    cfg = Qwen3Config(vocab_size=V, hidden_size=H, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
    cfg.num_target_layers = L
    cfg.dflash_config = {"block_size": BS, "mask_token_id": V - 1}
    cfg._attn_implementation = "sdpa"
    return DFlashDraftModel(cfg).eval()


def packed_logits(draft, target, feats, ids, blocks):
    noise, pos, mask, starts = pack(draft, target, feats, ids, blocks, BS)
    hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                past_key_values=None, use_cache=False)[0]
    return [draft.compute_logits(hid[st : st + BS - 1], _output_head(target)) for st in starts]


def depth_logits(draft, target, feats, ids, blocks, deep):
    ctx, noise, pos, mask, starts = pack_depth(draft, target, feats, ids, blocks, BS, deep)
    hid = draft(target_hidden=ctx, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                past_key_values=None, use_cache=False)[0]
    return [draft.compute_logits(hid[st : st + BS - 1], _output_head(target)) for st in starts]


@torch.no_grad()
def main():
    target, draft = tiny_target(), tiny_draft()
    lids = list(draft.target_layer_ids)
    T = 40
    ids = torch.randint(0, V - 3, (T,))
    fails = []

    def check(name, ok):
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
        if not ok:
            fails.append(name)

    def feats_of(x):
        return extract_context_feature(target(x[None], output_hidden_states=True).hidden_states, lids)

    feats = feats_of(ids)

    # 1. causal features
    p = 20
    x2 = ids.clone()
    x2[p + 1 :] = torch.randint(0, V - 3, (T - p - 1,))
    f2 = feats_of(x2)
    check("1. features of positions <= p unchanged when later tokens change",
          (feats[:, : p + 1] - f2[:, : p + 1]).abs().max().item() < TOL)
    check("1. (sensitivity) features after p do change", (feats[:, p + 1 :] - f2[:, p + 1 :]).abs().max().item() > 1e-3)

    # 2. lagged block: no leak from the future, none from features it should not have
    for s, g in ((20, 0), (20, 4), (20, 8), (25, 16)):
        base = packed_logits(draft, target, feats, ids, [(s, g)])[0]
        x2 = ids.clone()
        x2[s + 1 :] = torch.randint(0, V - 3, (T - s - 1,))
        check(f"2. lag {g}: logits unchanged when the tokens after the anchor change",
              (packed_logits(draft, target, feats, x2, [(s, g)])[0] - base).abs().max().item() < TOL)
        f2 = feats.clone()
        f2[:, s - g :] = torch.randn_like(f2[:, s - g :])
        check(f"2. lag {g}: logits unchanged when the features at / after s - g change",
              (packed_logits(draft, target, f2, ids, [(s, g)])[0] - base).abs().max().item() < TOL)
        x2 = ids.clone()
        x2[s] = (ids[s] + 1) % (V - 3)
        check(f"2. lag {g}: (sensitivity) logits change with the anchor token",
              (packed_logits(draft, target, feats, x2, [(s, g)])[0] - base).abs().max().item() > 1e-3)
        f2 = feats.clone()
        f2[:, s - g - 1] = torch.randn_like(f2[:, s - g - 1])
        check(f"2. lag {g}: (sensitivity) logits change with the last allowed feature",
              (packed_logits(draft, target, f2, ids, [(s, g)])[0] - base).abs().max().item() > 1e-3)
        if g:
            x2 = ids.clone()
            x2[s - g] = (ids[s - g] + 1) % (V - 3)
            check(f"2. lag {g}: (sensitivity) logits change with an in-flight token",
                  (packed_logits(draft, target, feats, x2, [(s, g)])[0] - base).abs().max().item() > 1e-3)

    # 3. lag 0 equals the call of dflash_generate (no mask, features [0, s), block [x_s, MASK ...])
    s = 20
    blk = torch.full((1, BS), draft.mask_token_id, dtype=torch.long)
    blk[0, 0] = ids[s]
    hid = draft(target_hidden=feats[:, :s], noise_embedding=_raw_input_embeddings(target, blk, 1.0),
                position_ids=torch.arange(s + BS)[None], past_key_values=None, use_cache=False)[:, 1:]
    ref = draft.compute_logits(hid, _output_head(target))[0]
    check("3. lag 0, packed over the whole sequence = the inference call",
          (packed_logits(draft, target, feats, ids, [(s, 0)])[0] - ref).abs().max().item() < TOL)
    check("3. draft_logits_lagged at lag 0 = the inference call",
          (draft_logits_lagged(draft, target, feats, ids, s, 0, BS) - ref).abs().max().item() < TOL)

    # 4. packed blocks are independent
    blocks = [(12, 0), (20, 4), (25, 8), (30, 16), (22, 0)]
    together = packed_logits(draft, target, feats, ids, blocks)
    worst = max((together[i] - packed_logits(draft, target, feats, ids, [b])[0]).abs().max().item()
                for i, b in enumerate(blocks))
    check("4. a block's logits are the same packed alone or with other blocks", worst < TOL)

    # 5. pack_depth: extra rows per block
    deep = deep_columns(draft, 2, feats.shape[-1])
    blocks = [(20, 4), (26, 8), (30, 0)]
    base = depth_logits(draft, target, feats, ids, blocks, deep)
    alone = [depth_logits(draft, target, feats, ids, [b], deep)[0] for b in blocks]
    check("5. depth rows: a block's logits are the same packed alone or with other blocks",
          max((a - b).abs().max().item() for a, b in zip(base, alone)) < TOL)
    x2 = ids.clone()
    x2[21:] = torch.randint(0, V - 3, (T - 21,))
    check("5. depth rows: logits unchanged when the tokens after the anchor change",
          (depth_logits(draft, target, feats, x2, [(20, 4)], deep)[0] - alone[0]).abs().max().item() < TOL)
    f2 = feats.clone()
    f2[:, 20:] = torch.randn_like(f2[:, 20:])
    check("5. depth rows: logits unchanged when the features at / after the anchor change",
          (depth_logits(draft, target, f2, ids, [(20, 4)], deep)[0] - alone[0]).abs().max().item() < TOL)
    f2 = feats.clone()
    f2[:, 16:20, deep] = torch.randn_like(f2[:, 16:20, deep])
    check("5. depth rows: logits unchanged when the removed (deep) part of the in-flight rows changes",
          (depth_logits(draft, target, f2, ids, [(20, 4)], deep)[0] - alone[0]).abs().max().item() < TOL)
    f2 = feats.clone()
    f2[:, 16:20, ~deep] = torch.randn_like(f2[:, 16:20, ~deep])
    check("5. depth rows: (sensitivity) logits change with the kept part of the in-flight rows",
          (depth_logits(draft, target, f2, ids, [(20, 4)], deep)[0] - alone[0]).abs().max().item() > 1e-3)

    # 6. accepted length
    from exp27_refresh import matched
    check("6. matched counts the leading agreement only",
          matched([1, 2, 3, 9, 5], [1, 2, 3, 4, 5]) == 3 and matched([9, 2], [1, 2]) == 0
          and matched([1, 2], [1, 2, 3]) == 2 and matched([1, 2, 3], [1, 2]) == 2)

    print(f"\n{len(fails)} failed" if fails else "\nall passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
