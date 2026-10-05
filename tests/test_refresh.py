"""CPU losslessness test for feature-refreshed decoding (tiny random models, no downloads).

refresh_generate must reproduce the target's greedy output for any copy model (exact, noisy,
very noisy), any restart setting and any block size.

  python tests/test_refresh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_cpu import V, noisy_copy, tiny_draft, tiny_target  # noqa: E402

from hspec.pipeline import ar_generate  # noqa: E402
from hspec.refresh import chains_to_tree, refresh_generate  # noqa: E402


def main():
    torch.manual_seed(0)
    t = chains_to_tree([[1, 2, 3], [1, 2, 4], [5], [1, 6, 7]])
    ok = t.tokens == [1, 2, 3, 4, 5, 6, 7] and t.parents == [-1, 0, 1, 1, -1, 0, 5]
    print(f"chains_to_tree builds a prefix trie: {ok}")
    failures = int(not ok)

    target = tiny_target()
    copies = {"exact": noisy_copy(target, 0.0), "noisy-0.05": noisy_copy(target, 0.05),
              "noisy-0.3": noisy_copy(target, 0.3)}
    stops = [V - 2]
    for bsz in (8, 16):
        draft = tiny_draft(bsz)
        for trial in range(3):
            ids = torch.randint(0, V - 3, (1, 6 + 4 * trial))
            for max_new in (5, 60):
                ref = ar_generate(target, ids, max_new, stops).generated
                for cname, cp in copies.items():
                    for ks, pc in ((0, True), (2, True), (2, False)):
                        r = refresh_generate(draft, target, cp, ids, max_new, stops, ks=ks, use_pc=pc,
                                             block_size=bsz)
                        same = torch.equal(r.generated, ref)
                        failures += not same
                        if not same:
                            print(f"MISMATCH bs={bsz} trial={trial} max_new={max_new} copy={cname} ks={ks} pc={pc}: "
                                  f"{r.generated.tolist()} vs {ref.tolist()}")
                print(f"bs={bsz} prompt {trial} max_new={max_new}: AR {len(ref)} tokens checked "
                      f"(last round lens {r.rounds[-3:]})")
    # batched restart logits == separate drafter calls (direct check on a cached state)
    from dflash.model import _make_cache, extract_context_feature
    from hspec.pipeline import crop, draft_logits
    from hspec.refresh import batched_restarts

    draft = tiny_draft(16)
    ids = torch.randint(0, V - 3, (1, 40))
    start = 20
    feats = extract_context_feature(target(ids, output_hidden_states=True).hidden_states, draft.target_layer_ids)
    pos = torch.arange(200)[None]
    dc = _make_cache(draft.config)
    blk0 = torch.full((1, 16), V - 1); blk0[0, 0] = ids[0, start]
    draft_logits(draft, target, feats[:, :start], blk0, pos, start, dc)       # cache now [0, start)
    cfeat = feats[:, start : start + 16]
    specs = [(5, start + 5, 7), (1, start + 1, 3), (12, start + 12, 9)]
    lb = batched_restarts(draft, target, cfeat, specs, dc, start, 16, pos)
    ok = dc.get_seq_length() == start
    for r, (c, a, t) in enumerate(specs):
        blk = torch.full((1, 16), V - 1); blk[0, 0] = t
        ls = draft_logits(draft, target, cfeat[:, :c], blk, pos, a, dc)
        crop(dc, start)
        ok &= torch.allclose(lb[r], ls, atol=1e-4)
    failures += not ok
    print(f"batched_restarts logits == separate calls: {ok}")

    # batched restarts (one drafter call) must give the same output as separate calls
    draft = tiny_draft(16)
    for trial in range(3):
        ids = torch.randint(0, V - 3, (1, 8 + trial))
        a = refresh_generate(draft, target, copies["noisy-0.05"], ids, 60, stops, ks=3, batch_restarts=True,
                             block_size=16)
        b = refresh_generate(draft, target, copies["noisy-0.05"], ids, 60, stops, ks=3, batch_restarts=False,
                             block_size=16)
        ok = torch.equal(a.generated, b.generated) and a.rounds == b.rounds
        failures += not ok
        print(f"batched restarts == separate restarts (trial {trial}): {ok}  rounds {a.rounds[:6]}")

    # with an exact copy the correction is the target's own token: every round (but the last)
    # must accept it plus the bonus, i.e. produce >= 2 tokens; a noisy copy must still help a random drafter
    draft = tiny_draft(16)
    ids = torch.randint(0, V - 3, (1, 9))
    for cname in ("exact", "noisy-0.05"):
        r = refresh_generate(draft, target, copies[cname], ids, 60, [], ks=0, use_pc=True, block_size=16)
        mean = sum(r.rounds) / len(r.rounds)
        ok = min(r.rounds[:-1]) >= 2 if cname == "exact" else mean > 1.2
        failures += not ok
        print(f"copy={cname}: tokens per round {mean:.2f} (min {min(r.rounds[:-1])}) -> {ok}")
    print("ALL PASSED" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
