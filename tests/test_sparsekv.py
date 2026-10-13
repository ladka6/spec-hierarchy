"""CPU test for hspec/sparsekv.py and exp60's per-anchor evaluation (tiny random models, no downloads).

  1. prefill_features in chunks == the features of one full forward (and the cache matches).
  2. attention_scores returns [Hkv, s] per layer, non-negative, and the probe mass sums to ~probe per head.
  3. A sparse view that selects ALL prefix positions reproduces the full-cache logits exactly (gather is a no-op),
     and with every position selected the copy "target/sparse" equals "target/full" in eval_anchor.
  4. A real sparse view runs, with the right number of positions read, and its logits differ from the full ones.

  python tests/test_sparsekv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_cpu import V, tiny_draft, tiny_target  # noqa: E402

from dflash.model import extract_context_feature  # noqa: E402
from exp60_sparsekv import eval_anchor, greedy_long  # noqa: E402
from hspec.pipeline import ar_generate  # noqa: E402
from hspec.sparsekv import attention_scores, prefill_features, prefix_cache, select_indices, sparse_cache  # noqa: E402


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    target = tiny_target()
    draft = tiny_draft(8)
    bs = draft.block_size
    failures = 0
    ids = torch.randint(0, V - 3, (1, 10))
    x = ar_generate(target, ids, 70, [V + 10]).output_ids[0]          # no stop token: a long trajectory
    T = len(x)

    # 1. chunked prefill == full forward
    full = target(x[None], output_hidden_states=True, logits_to_keep=1, use_cache=True)
    f_ref = extract_context_feature(full.hidden_states, draft.target_layer_ids)
    feats, cache = prefill_features(target, x, draft.target_layer_ids, chunk=16)
    err = (feats.float() - f_ref.float()).abs().max().item()
    ok = feats.shape == f_ref.shape and err < 1e-3
    print(f"chunked prefill features match: {ok} (max abs err {err:.2e})")
    failures += not ok
    kerr = max((a.keys.float() - b.keys.float()).abs().max().item() for a, b in zip(cache.layers, full.past_key_values.layers))
    print(f"chunked prefill cache matches: {kerr < 1e-3} (max abs err {kerr:.2e})")
    failures += kerr >= 1e-3

    # 2. attention scores
    s = T - 2 * bs - 2
    sc = attention_scores(target, cache, x, s, probe=4)
    Hkv = target.config.num_key_value_heads
    ok = len(sc) == target.config.num_hidden_layers and all(t.shape == (Hkv, s) for t in sc) and all((t >= 0).all() for t in sc)
    g = target.config.num_attention_heads // Hkv
    mass = sc[0].sum(-1) / g                       # per KV head, mass over the prefix [0, s) summed over 4 queries
    print(f"attention scores shape/sign ok: {ok}; prefix mass per head (<= 4 probes): {[round(m, 2) for m in mass.tolist()]}")
    failures += not ok or bool((mass > 4.0 + 1e-3).any())
    # sdpa is restored afterwards
    ok = target.config._attn_implementation == "sdpa"
    print(f"attention implementation restored to sdpa: {ok}")
    failures += not ok

    # 3. selecting everything == full cache
    z = x[s : s + bs]
    pos = torch.arange(s, s + bs)[None]
    ref = target(z[None], past_key_values=prefix_cache(cache, s), position_ids=pos, use_cache=True, logits_to_keep=bs).logits
    idx = select_indices(sc, s, sinks=s, window=0, topk=0)
    ok = all(i.shape == (Hkv, s) for i in idx)
    out = target(z[None], past_key_values=sparse_cache(cache, idx, s), position_ids=pos, use_cache=True, logits_to_keep=bs).logits
    err = (out.float() - ref.float()).abs().max().item()
    print(f"all-positions sparse view == full cache logits: {err < 1e-4} (max abs err {err:.2e})")
    failures += not ok or err >= 1e-4

    # 4. a real sparse view
    idx = select_indices(sc, s, sinks=2, window=4, topk=3)
    S = idx[0].shape[1]
    out2 = target(z[None], past_key_values=sparse_cache(cache, idx, s), position_ids=pos, use_cache=True, logits_to_keep=bs).logits
    diff = (out2.float() - ref.float()).abs().max().item()
    print(f"sparse view reads {S} of {s} positions, logits differ from full: {diff > 1e-4} (max abs diff {diff:.2e})")
    failures += not (S == 2 + 4 + 3) or diff <= 1e-4
    # selected indices are sorted and unique per head
    ok = all(bool((i[:, 1:] > i[:, :-1]).all()) for i in idx)
    print(f"indices sorted and unique: {ok}")
    failures += not ok

    # eval_anchor: 'target' copy with the all-positions view equals the full view
    views = {"full": None, "all": (s, 0, 0), "sparse": (2, 4, 3)}
    res = eval_anchor(draft, target, {"target": target}, feats, cache, x, s, bs, views, probe=4)
    a, b, c = res[("target", "full")], res[("target", "all")], res[("target", "sparse")]
    ok = a[:2] == b[:2] and abs(a[2] - b[2]) < 1e-6 and abs(a[3] - 1.0) < 1e-4 and b[4] == s and c[4] == 2 + 4 + 3
    print(f"eval_anchor: full {a}, all-positions {b}, sparse {c}: consistent {ok}")
    failures += not ok
    # greedy_long (chunked prefill + greedy) == plain AR greedy, with and without a stop token
    for stops in ([V + 10], [int(x[20])]):
        ref = ar_generate(target, ids, 40, stops).output_ids[0]
        got = greedy_long(target, ids[0], 40, stops, chunk=3)
        ok = torch.equal(ref, got)
        print(f"greedy_long == AR greedy (stops {stops}): {ok} ({len(got)} tokens)")
        failures += not ok
    print("ALL PASSED" if not failures else f"{failures} FAILURES")


if __name__ == "__main__":
    main()
