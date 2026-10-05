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
