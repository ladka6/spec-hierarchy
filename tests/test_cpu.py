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

from hspec.pipeline import WindowPolicy, ar_generate, three_stage_generate, two_stage_generate  # noqa: E402

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
    print("\nALL PASSED" if failures == 0 else f"\n{failures} FAILURES")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
