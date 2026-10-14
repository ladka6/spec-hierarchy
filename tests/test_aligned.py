"""CPU test for exp62 (tiny random models, no downloads).

The test sequence is P + A + B + A' where A' repeats A exactly (offline, the sequence itself is the trajectory).
  1. ngram_match at an anchor inside A' finds the matching position inside A, at the right offset, and returns None
     where nothing repeats.
  2. eval_anchor runs for every base / match; at an anchor inside A' the ngram lookup chain matches the trajectory
     for all 15 tokens (lookup tau >= 16), aligned >= lookup >= block1 (tree unions only add chains), and the
     oracle restart uses the same tokens.
  3. Without a match the extra variants fall back to block1.
  4. summarize produces rows.

  python tests/test_aligned.py
"""

from __future__ import annotations

import copy as _copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_cpu import V, tiny_draft, tiny_target  # noqa: E402

from exp62_aligned import eval_anchor, ngram_match, summarize  # noqa: E402
from hspec.sparsekv import prefill_features  # noqa: E402


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    target = tiny_target()
    draft = tiny_draft(16)
    bs = draft.block_size
    D = bs - 1
    failures = 0
    g = torch.Generator().manual_seed(1)
    P = torch.randint(0, V - 3, (20,), generator=g)
    A = torch.randint(0, V - 3, (60,), generator=g)
    B = torch.randint(0, V - 3, (40,), generator=g)
    x = torch.cat([P, A, B, A, torch.randint(0, V - 3, (60,), generator=g)])
    a0, a1 = 20, 120                                   # starts of A and of its repeat
    x_np = x.numpy()

    # 1. matching
    s = a1 + 10
    i, n = ngram_match(x_np, s, D)
    ok = i == a0 + 10 and n == 8
    print(f"ngram_match inside the repeat: {ok} (i={i}, expected {a0 + 10}, n={n})")
    failures += not ok
    s_new = len(x) - 3 * bs - 1                         # inside the random tail: almost surely no 8..2-gram... allow 2-grams
    i2, n2 = ngram_match(x_np, s_new, D, nmin=6)
    ok = i2 is None
    print(f"no 6+-gram match in the random tail: {ok} (i={i2}, n={n2})")
    failures += not ok

    # 2./3. eval_anchor
    feats, cache = prefill_features(target, x, draft.target_layer_ids, chunk=32)
    args = SimpleNamespace(bases=["recent-64-orig", "attn-64-compact", "full"], match=["ngram", "attn"],
                           probe=4, window=16, restart_at=[8, 15], copy_budget=16)
    res, diag = eval_anchor(draft, target, _copy.deepcopy(target), feats, cache, x, x_np, s, bs, args)
    t = res["recent-64-orig|ngram"]
    ok = (diag["ngram"]["r_match"] == D and t["lookup"][0] >= 16 and t["aligned"][0] >= t["lookup"][0]
          >= t["block1"][0] and t["oracle"][0] >= t["lookup"][0] and t["_hit"] == 1)
    print(f"repeat anchor: lookup tau {t['lookup'][0]}, aligned {t['aligned'][0]}, oracle {t['oracle'][0]}, "
          f"block1 {t['block1'][0]}, copy {t['copy'][0]}, aligned+copy {t['aligned+copy'][0]}: {ok}")
    failures += not ok
    ok = all(set(v_) >= {"block1", "copy", "lookup", "aligned", "oracle", "aligned+copy", "_hit"} for v_ in res.values())
    print(f"all bases x matches present with all variants: {ok} ({sorted(res)})")
    failures += not ok
    res2, diag2 = eval_anchor(draft, target, None, feats, cache, x, x_np, s_new,
                              bs, SimpleNamespace(**{**vars(args), "match": ["ngram"]}))
    if diag2["ngram"] is None:
        t2 = res2["recent-64-orig|ngram"]
        ok = t2["lookup"] == t2["block1"] == t2["aligned"] and t2["_hit"] == 0 and "copy" not in t2
        print(f"no match: variants fall back to block1: {ok}")
        failures += not ok
    else:
        print(f"(tail anchor found a 2-gram match at {diag2['ngram']}; fallback check skipped)")

    # 4. summarize
    recs = [{"res": res, "diag": diag}, {"res": res, "diag": diag}]
    rows = summarize(recs, SimpleNamespace(kv_ms_per_ktoken=0.1, drafter_ms=2.6, restart_ms=1.0, copy_ms=3.0), 8192)
    ok = len(rows) == 6 * 6 and all(r["x"] > 0 for r in rows)
    print(f"summarize: {len(rows)} rows: {ok}")
    failures += not ok
    print("ALL PASSED" if not failures else f"{failures} FAILURES")


if __name__ == "__main__":
    main()
