"""Experiment 41: block-optimal relaxed verification (stage 1, offline).

Lossy acceptance rules decide token by token: MARS accepts the target's runner-up at near-ties,
Fuzzy SD accepts when draft and target distributions are close. But the verify pass already holds the
target's logits for the WHOLE drafted block (conditioned on the drafted prefix), so whether forgiving a
mismatch at j pays off (do the tokens after j match?) is known. Block-optimal rule: accept the prefix
length L maximising  L - lambda * sum_{i<L, mismatch} gap_i,  gap_i = p_target(top1) - p_target(draft_i).
An exact search over L (the block has 15 drafted positions), using only verify-pass quantities.

Here, on target-greedy trajectories (the exact-decoding state distribution), at each anchor:
DFlash drafts a block, the target verifies it (logits at every drafted position). For each rule and
parameter: mean tau (accepted + bonus) and the quality cost it spends, as mean gap per generated token
and forgiven mismatches per 100 tokens. Main table: best tau per rule at matched cost budgets.

  python scripts/exp41_blockrelax.py --data .../hspec_lagft/data
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

BUDGETS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def js(p_log, q_log):
    p, q = p_log.exp(), q_log.exp()
    m = (0.5 * (p + q)).clamp_min(1e-12).log()
    return 0.5 * ((p * (p_log - m)).sum(-1) + (q * (q_log - m)).sum(-1))


def run_policy(blocks, forgive_fn=None, lam=None):
    """forgive_fn(block) -> bool tensor [n] of forgivable positions (myopic rules), or lam for block-optimal.
    Returns mean tau, mean gap per generated token, forgiven per 100 tokens."""
    tau = cost = forg = 0.0
    for b in blocks:
        match, gap = b["match"], b["gap"]
        n = len(match)
        if lam is None:
            ok = match | forgive_fn(b) if forgive_fn is not None else match
            L = int(ok.long().cumprod(0).sum())
        else:
            c = torch.cat([torch.zeros(1), torch.cumsum(gap * (~match).float(), 0)])        # cost of accepting L
            score = torch.arange(n + 1).float() - lam * c
            L = int(score.argmax())
        mism = ~match[:L]
        tau += L + 1
        cost += float((gap[:L] * mism.float()).sum())
        forg += int(mism.sum())
    T = tau
    return tau / len(blocks), cost / T, 100.0 * forg / T


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-seqs", type=int, default=400)
    ap.add_argument("--anchors", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="exp41_blockrelax")
    args = ap.parse_args()
    rng = random.Random(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids = list(draft.target_layer_ids)
    bs = draft.block_size

    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
        if len(data) >= 3 * args.n_seqs:
            break
    data = [d for d in data if len(d["ids"]) <= args.max_len and len(d["ids"]) - d["n_prompt"] > 8]
    rng.shuffle(data)
    data = data[: args.n_seqs]

    blocks = []
    for i, d in enumerate(data):
        ids = d["ids"].long().to(dev)
        T = len(ids)
        out = target(ids[None], output_hidden_states=True, use_cache=True)
        feats = extract_context_feature(out.hidden_states, lids)
        full_logits, cache = out.logits[0], out.past_key_values
        lo, hi = max(d["n_prompt"], 1), T - 1
        for s in rng.sample(range(lo, hi), min(args.anchors, hi - lo)):
            dl = draft_logits_lagged(draft, target, feats, ids, s, 0, bs).float()        # positions s+1 .. s+bs-1
            dtok = dl.argmax(-1)
            sl = DynamicCache(ddp_cache_data=[(l.keys[:, :, : s + 1], l.values[:, :, : s + 1]) for l in cache.layers])
            vl = target(dtok[None], past_key_values=sl, use_cache=True).logits[0].float()   # row i predicts s+2+i
            tl = torch.cat([full_logits[s : s + 1].float(), vl[:-1]])                      # target at s+1 .. s+bs-1
            lp = F.log_softmax(tl, -1)
            p = lp.exp()
            top2 = tl.topk(2, -1)
            pd = p.gather(-1, dtok[:, None])[:, 0]
            blocks.append({
                "match": (dtok == top2.indices[:, 0]).cpu(),
                "gap": (p.max(-1).values - pd).cpu(),
                "is2": (dtok == top2.indices[:, 1]).cpu(),
                "ratio": (top2.values[:, 1] / top2.values[:, 0].clamp_min(1e-6)).cpu(),     # MARS logit ratio
                "js": js(F.log_softmax(dl, -1), lp).cpu(),
            })
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(data)}] {len(blocks)} blocks", flush=True)

    fams = {
        "exact": [("-", dict(forgive_fn=None))],
        "MARS(theta)": [(th, dict(forgive_fn=lambda b, th=th: b["is2"] & (b["ratio"] >= th)))
                        for th in (0.99, 0.98, 0.96, 0.94, 0.92, 0.9, 0.88, 0.85, 0.8)],
        "gap<=d": [(dd, dict(forgive_fn=lambda b, dd=dd: b["gap"] <= dd))
                   for dd in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)],
        "Fuzzy JS<=t": [(t, dict(forgive_fn=lambda b, t=t: b["js"] <= t))
                        for t in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)],
        "block-opt(lambda)": [(lam, dict(lam=lam)) for lam in (200, 100, 50, 30, 20, 15, 10, 7, 5, 3, 2, 1)],
    }
    rows = []
    for fam, params in fams.items():
        for prm, kw in params:
            tau, cost, forg = run_policy(blocks, **kw)
            rows.append({"rule": fam, "param": prm, "tau": tau, "gap_per_tok": cost, "forgiven_per_100": forg})
    print_table(rows, ["rule", "param", "tau", "gap_per_tok", "forgiven_per_100"],
                f"all parameter settings ({len(blocks)} DFlash blocks verified by the target)")

    exact_tau = rows[0]["tau"]
    brows = []
    for B in BUDGETS:
        r = {"budget_gap_per_tok": B, "exact": exact_tau}
        for fam in list(fams)[1:]:
            ok = [x for x in rows if x["rule"] == fam and x["gap_per_tok"] <= B]
            r[fam] = max((x["tau"] for x in ok), default=exact_tau)
        brows.append(r)
    print_table(brows, ["budget_gap_per_tok", "exact"] + list(fams)[1:],
                "best tau at a quality budget (mean target-probability gap spent per generated token)")
    save_json(args.out, {"args": vars(args), "all": rows, "budget": brows, "n_blocks": len(blocks)})


if __name__ == "__main__":
    main()
