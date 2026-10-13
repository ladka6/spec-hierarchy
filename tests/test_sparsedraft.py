"""CPU test for hspec/sparsedraft.py and exp61 (tiny random models, no downloads).

  1. draft_block over the full context at true positions == draft_logits_lagged (the existing drafter call), and a
     restart with extra copy features == blocks_logits on the concatenated features (exp27/exp56's restarts).
  2. compact positions over the full context == orig positions shifted (RoPE is relative: identical logits).
  3. select_positions: sizes, sinks and window present, sorted, unique, and all of [0, s) when s <= budget.
  4. eval_anchor runs for every variant; 'full' and any variant whose budget covers the context give identical taus.
  5. copy_rate on a toy example.

  python tests/test_sparsedraft.py
"""

from __future__ import annotations

import copy as _copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_cpu import V, tiny_draft, tiny_target  # noqa: E402

from exp27_refresh import blocks_logits  # noqa: E402
from exp60_sparsekv import greedy_long  # noqa: E402
from exp61_sparsedraft import copy_rate, eval_anchor, variants  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.sparsedraft import context_for, draft_block, select_positions  # noqa: E402
from hspec.sparsekv import prefill_features  # noqa: E402


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    target = tiny_target()
    draft = tiny_draft(8)
    bs = draft.block_size
    failures = 0
    ids = torch.randint(0, V - 3, (12,))
    x = greedy_long(target, ids, 80, [V + 10], chunk=16)
    feats, cache = prefill_features(target, x, draft.target_layer_ids, chunk=16)
    s = 60

    # 1. full context, true positions == existing drafter call
    ref = draft_logits_lagged(draft, target, feats, x, s, 0, bs)
    ctx, pos, ap = context_for(feats, torch.arange(s), None, s, "orig")
    got = draft_block(draft, target, ctx, pos, x[s], ap, bs)
    err = (got.float() - ref.float()).abs().max().item()
    print(f"draft_block(full, orig) == draft_logits_lagged: {err < 1e-4} (max abs err {err:.2e})")
    failures += err >= 1e-4
    extra = feats[:, s : s + 3] + 0.1                    # stand-in copy features for positions s .. s+2
    full = torch.cat([feats[:, :s], extra], 1)
    zc = torch.cat([x[: s + 3], torch.tensor([5])])      # anchor token 5 at position s+3
    ref2 = blocks_logits(draft, target, full, zc, [(s + 3, 0)], bs)[0]
    c2, p2, a2 = context_for(feats, torch.arange(s), extra, s, "orig")
    got2 = draft_block(draft, target, c2, p2, 5, a2, bs)
    err = (got2.float() - ref2.float()).abs().max().item()
    print(f"restart with extra features == blocks_logits: {err < 1e-4} (max abs err {err:.2e}, anchor pos {a2})")
    failures += err >= 1e-4 or a2 != s + 3

    # 2. compact over the full context == orig (positions 0..s-1 either way)
    cc, pc, ac = context_for(feats, torch.arange(s), None, s, "compact")
    ok = torch.equal(pc, torch.arange(s)) and ac == s
    got3 = draft_block(draft, target, cc, pc, x[s], ac, bs)
    err = (got3.float() - ref.float()).abs().max().item()
    print(f"compact == orig on the full context: {ok and err < 1e-4}")
    failures += not ok or err >= 1e-4
    #    and compact over a strict subset differs from orig over the same subset (positions renumbered)
    sub = torch.cat([torch.arange(4), torch.arange(s - 20, s)])
    co, po_, ao = context_for(feats, sub, None, s, "orig")
    cm, pm, am = context_for(feats, sub, None, s, "compact")
    lo = draft_block(draft, target, co, po_, x[s], ao, bs)
    lm = draft_block(draft, target, cm, pm, x[s], am, bs)
    d = (lo.float() - lm.float()).abs().max().item()
    print(f"compact != orig on a gapped subset (sinks + window): {d > 1e-4} (max abs diff {d:.2e}; anchor {ao} vs {am})")
    failures += d <= 1e-4 or am != 24

    # 3. selection
    score = torch.rand(s)
    for kind in ("recent", "attn"):
        idx = select_positions(score, s, 24, kind, sinks=4, window=8)
        ok = (len(idx) == 24 and bool((idx[1:] > idx[:-1]).all()) and set(range(4)) <= set(idx.tolist())
              and set(range(s - 8, s)) <= set(idx.tolist()) and int(idx.max()) < s)
        print(f"select_positions {kind}: {ok} ({idx.tolist()})")
        failures += not ok
    idx = select_positions(score, s, 24, "attn", sinks=4, window=8)
    mid = [i for i in idx.tolist() if 4 <= i < s - 8]
    best = sorted(range(4, s - 8), key=lambda i: -float(score[i]))[:12]
    ok = sorted(mid) == sorted(best)
    print(f"attn picks the top-scoring middle positions: {ok}")
    failures += not ok
    ok = torch.equal(select_positions(score, s, 200, "attn"), torch.arange(s))
    print(f"budget >= context keeps everything: {ok}")
    failures += not ok

    # 4. eval_anchor
    args = SimpleNamespace(probe=4, copy_sinks=2, copy_window=4, copy_budget=8, window=8, no_refresh=False)
    vars_ = variants([16, 200])
    res = eval_anchor(draft, target, _copy.deepcopy(target), feats, cache, x, s, bs, vars_, args)
    ok = set(res) == {v[0] for v in vars_}
    big = [k for k in res if "-200-" in k]
    same = all(res[k][:2] == res["full"][:2] for k in big)
    print(f"eval_anchor covers all variants: {ok}; budget-200 variants == full: {same}")
    print("   " + ", ".join(f"{k}: {v[0]}/{v[1]} read {v[2]:.2f}" for k, v in res.items()))
    failures += not ok or not same

    # 5. copy rate
    r = copy_rate(list(range(20)), list(range(5, 30)), n=4)
    ok = abs(r - 12 / 17) < 1e-9
    print(f"copy_rate: {ok} ({r:.3f})")
    failures += not ok
    print("ALL PASSED" if not failures else f"{failures} FAILURES")


if __name__ == "__main__":
    main()
