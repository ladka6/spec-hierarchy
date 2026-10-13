"""CPU smoke test for exp59 (tiny random models, no downloads).

Checks that collect_anchor runs, that every saved chain's `match` equals the matched prefix against the
trajectory, that tau of each policy is consistent with the verified node sets (an exact copy must make the
copy walk cover the whole accepted spine), and that the replay produces a table for several batch sizes.

  python tests/test_batchprune.py
"""

from __future__ import annotations

import copy as _copy
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_cpu import V, noisy_copy, tiny_draft, tiny_target  # noqa: E402

from dflash.model import extract_context_feature  # noqa: E402
from exp59_batchprune import _matched, collect_anchor, replay_rows, verified_sets  # noqa: E402
from hspec.pipeline import ar_generate  # noqa: E402


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    target = tiny_target()
    draft = tiny_draft(8)
    bs = draft.block_size
    failures = 0
    for cname, cp in (("exact", _copy.deepcopy(target)), ("noisy", noisy_copy(target, 0.3))):
        recs = []
        for trial in range(2):
            ids = torch.randint(0, V - 3, (1, 8))
            x = ar_generate(target, ids, 48, [V - 2]).output_ids[0]
            tout = target(x[None], output_hidden_states=True, logits_to_keep=1, use_cache=True)
            feats = extract_context_feature(tout.hidden_states, draft.target_layer_ids)
            tcache = tout.past_key_values
            for s in range(ids.shape[1], len(x) - 2 * bs - 1, 4):
                rec = collect_anchor(draft, target, cp, feats, tcache, x, s, bs, big=16)
                rec["dataset"] = "tiny"
                for k, c in rec["chains"].items():
                    if c["match"] != _matched(c["tokens"], rec["truth"]):
                        failures += 1
                        print(f"bad match for chain {k}")
                if cname == "exact":
                    # an exact copy walks the tree exactly like the target: its accepted spine prefix is the
                    # drafter's matched prefix (as far as the spine is in the tree)
                    m = _matched(rec["y"], rec["truth"])
                    spine_ok = min(m, len(rec["spine"]))
                    walk_on_spine = [r for r in rec["walk"] if r in rec["spine"]]
                    if len(walk_on_spine) < spine_ok and rec["walk"][: spine_ok] != rec["spine"][:spine_ok]:
                        failures += 1
                        print(f"exact copy walk {rec['walk']} does not cover the matched spine {rec['spine'][:spine_ok]}")
                    sets = verified_sets(rec, widths=[0, 4, 16], ks=[16])
                    if sets["narrow"][0] < m + 1:
                        failures += 1
                        print(f"narrow tau {sets['narrow'][0]} < dflash tau {m + 1} with an exact copy")
                recs.append(rec)
        json.loads(json.dumps(recs))                                    # serialisable
        rows = replay_rows(recs, costs=None, batches=[1, 8], drafter_ms=2.6, drafter_knee=16, restart_ms=1.0,
                           widths=[0, 4, 16], ks=[16])
        pols = {r["policy"] for r in rows}
        for need in ("dflash", "ddtree-16", "pc+c2", "narrow", "prune16-W4+tp"):
            if need not in pols:
                failures += 1
                print(f"missing policy {need}")
        print(f"copy={cname}: {len(recs)} anchors, {len(pols)} policies, "
              f"dflash tau {next(r['tau'] for r in rows if r['policy'] == 'dflash'):.2f}, "
              f"narrow tau {next(r['tau'] for r in rows if r['policy'] == 'narrow'):.2f}")
        # measured-curve path of the replay
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bench.json"
            p.write_text(json.dumps({"rows": [
                {"model": "Qwen/Qwen3-8B", "q_ms": {"1": 12.7, "17": 13.3, "65": 14.2},
                 "real_q_ms": {"1": {"1": 12.7, "17": 13.3, "65": 14.2}, "8": {"1": 14, "17": 20, "65": 60}}},
                {"model": "Qwen/Qwen3-8B-AWQ", "q_ms": {"1": 3, "17": 3, "65": 4.5},
                 "real_q_ms": {"1": {"1": 3, "17": 3, "65": 4.5}, "8": {"1": 3.2, "17": 4, "65": 8}}}]}))
            from exp59_batchprune import load_curves
            costs = (load_curves(p, "Qwen3-8B"), load_curves(p, "AWQ"))
            rows2 = replay_rows(recs, costs, [1, 8, 32], 2.6, 16, 1.0, [0, 4, 16], [16])
            if {r["batch"] for r in rows2} != {1, 8}:
                failures += 1
                print("replay with measured curves should skip batch 32 (no curve) and keep 1 and 8")
    print("ALL PASSED" if not failures else f"{failures} FAILURES")


if __name__ == "__main__":
    main()
