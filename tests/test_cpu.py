"""CPU correctness test with tiny random models (no downloads).

Checks that every decoder is lossless under greedy decoding: two-stage (any feature model)
and three-stage (any window) must reproduce the target's autoregressive greedy output.

  python tests/test_cpu.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import DFlashDraftModel  # noqa: E402

from hspec.pipeline import (  # noqa: E402
    WindowPolicy, ar_generate, ddtree_generate, three_stage_generate, two_stage_generate,
)

torch.manual_seed(0)
V, H, L = 96, 64, 6


def tiny_target():
    cfg = Qwen3Config(vocab_size=V, hidden_size=H, intermediate_size=128, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=1024, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    m = Qwen3ForCausalLM(cfg).eval()
    # sharpen the output distribution so greedy trajectories are not all ties
    with torch.no_grad():
        m.lm_head.weight.mul_(8.0)
    return m


def noisy_copy(model, scale):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * scale * p.std())
    return m


def tiny_draft(block_size=8):
    cfg = Qwen3Config(vocab_size=V, hidden_size=H, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=1024)
    cfg.num_target_layers = L
    cfg.dflash_config = {"block_size": block_size, "mask_token_id": V - 1}
    cfg._attn_implementation = "sdpa"
    return DFlashDraftModel(cfg).eval()


def main():
    target = tiny_target()
    draft = tiny_draft()
    mids = {"exact-copy": copy.deepcopy(target), "noisy-0.05": noisy_copy(target, 0.05),
            "noisy-0.3": noisy_copy(target, 0.3)}
    stops = [V - 2]
    max_new = 80
    failures = 0

    # drafter input must not depend on stale tokens after the anchor (regression test)
    from dflash.model import _make_cache, extract_context_feature
    from hspec.pipeline import propose

    ids = torch.randint(0, V - 3, (1, 12))
    out = target(ids, output_hidden_states=True)
    ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)
    pos = torch.arange(64).unsqueeze(0)
    masked = torch.full((1, 8), V - 1)
    masked[0, 0] = 5
    stale = torch.randint(0, V - 3, (1, 8))
    stale[0, 0] = 5
    p1 = propose(draft, target, ctx, masked, pos, 12, _make_cache(draft.config))
    p2 = propose(draft, target, ctx, stale, pos, 12, _make_cache(draft.config))
    ok = torch.equal(p1, p2)
    failures += not ok
    print(f"propose ignores stale block content: {ok}")

    # tree attention: every node's logits equal a causal forward over its root path
    from hspec.tree import Tree, chain, verify_tree

    prefix = torch.randint(0, V - 3, (1, 10))
    cache = _make_cache(target.config)
    target(prefix, past_key_values=cache, use_cache=True)
    tree = Tree()
    a = tree.add(3, -1); b = tree.add(4, a); tree.add(7, a); tree.add(9, -1); tree.add(1, b)
    tl = verify_tree(target, cache, 5, tree, 10).logits[0]
    ok = True
    for i in range(len(tree)):
        path, j = [], i
        while j >= 0:
            path.append(tree.tokens[j]); j = tree.parents[j]
        seq = torch.cat([prefix, torch.tensor([[5] + path[::-1]])], dim=1)
        ok &= torch.allclose(target(seq).logits[0, -1], tl[i + 1], atol=1e-4)
    cache = _make_cache(target.config)
    target(prefix, past_key_values=cache, use_cache=True)
    ok &= torch.allclose(verify_tree(target, cache, 5, chain([3, 4, 7]), 10).logits[0],
                         target(torch.cat([prefix, torch.tensor([[5, 3, 4, 7]])], 1)).logits[0, 10:],
                         atol=1e-4)
    failures += not ok
    print(f"tree attention matches causal paths: {ok}")
    branch_hits = 0

    for trial in range(4):
        ids = torch.randint(0, V - 3, (1, 7 + 3 * trial))
        ref = ar_generate(target, ids, max_new, stops).generated
        print(f"\nprompt {trial}: AR produced {len(ref)} tokens")

        r = two_stage_generate(draft, target, target, target, ids, max_new, stops)
        ok = torch.equal(r.generated, ref)
        failures += not ok
        print(f"  two-stage T/T          lossless={ok}  tau={sum(r.rounds) / len(r.rounds):.2f}")
        for name, mid in mids.items():
            r = two_stage_generate(draft, target, target, mid, ids, max_new, stops)
            ok = torch.equal(r.generated, ref)
            failures += not ok
            print(f"  two-stage M>T {name:11s} lossless={ok}")
            for w in (1, 4, 16, 64):
                r = three_stage_generate(draft, target, mid, ids, max_new, stops,
                                         WindowPolicy("fixed", window=w))
                ok = torch.equal(r.generated, ref)
                failures += not ok
                s = r.summary()
                print(f"  three-stage {name:11s} P={w:3d} lossless={ok}  "
                      f"target_calls={r.target_calls:3d} acc/check={s.get('mean_accepted_per_check', 0):.2f}")
            r = three_stage_generate(draft, target, mid, ids, max_new, stops,
                                     WindowPolicy("adaptive", window=8))
            ok = torch.equal(r.generated, ref)
            failures += not ok
            print(f"  three-stage {name:11s} adaptive lossless={ok}")
            for w, k, bl in ((16, 4, 8), (64, 8, 4), (4, 2, 0)):
                r = three_stage_generate(draft, target, mid, ids, max_new, stops,
                                         WindowPolicy("fixed", window=w), branch_k=k,
                                         branch_len=bl, branch_margin=1.0)
                ok = torch.equal(r.generated, ref)
                failures += not ok
                s = r.summary()
                branch_hits += s.get("branch_hit_rate", 0) * len(r.checks)
                print(f"  3s+branch {name:11s} P={w:3d} k={k} L={bl} lossless={ok}  "
                      f"target_calls={r.target_calls:3d} hit_rate={s.get('branch_hit_rate', 0):.2f}")
        for budget, bsz in ((1, None), (8, None), (32, None), (64, 16)):
            r = ddtree_generate(draft, target, ids, max_new, stops, budget=budget, block_size=bsz)
            ok = torch.equal(r.generated, ref)
            failures += not ok
            print(f"  ddtree B={budget:3d} bs={bsz}  lossless={ok}  target_calls={r.target_calls:3d} "
                  f"mean_q={r.summary()['mean_target_q']:.1f}")
        r = two_stage_generate(draft, target, target, target, ids, max_new, stops, block_size=16)
        ok = torch.equal(r.generated, ref)
        failures += not ok
        print(f"  two-stage block 16     lossless={ok}")
    if branch_hits == 0:
        failures += 1
        print("branching never recovered a rejection: branch path untested")
    print("\nALL PASSED" if failures == 0 else f"\n{failures} FAILURES")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
