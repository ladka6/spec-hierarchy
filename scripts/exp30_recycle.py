"""Experiment 30: recycle the verifier's leftover computation instead of running a copy (T=0).

When the target verifies a chain C at anchor s and rejects at index a (bonus b = its correct
token at s' = s+a+1), it has still run all its layers over the rejected suffix C[a:] (with
the wrong token C[a] at s'). DFlash throws that away. Here the next round reuses it, exactly
like the copy-corrected restart but with the verify pass as the "copy" (no extra model, no
extra pass):

  kept suffix   K = C[a+1:] at positions s'+1 .. (substitution hypothesis: only C[a] was wrong)
  stale feats   the verify pass's features at s' .. end, and its predictions after each of them
  restart (pc)  j = first index where the stale prediction disagrees with K; chain
                R = K[:j] + [stale prediction at j] + 15 drafted tokens, drafter context =
                real features up to s' + stale features s' .. s'+j

Policies, each simulated as a full greedy decoding loop along the target-greedy trajectory
(the emitted text is the trajectory, as in the real lossless loop):

  dflash          plain DFlash every round
  recycle_end     keep all of K, draft 15 after its end from stale features (no correction)
  recycle         the pc restart above (falls back to DFlash after a fully accepted round)
  recycle_tree    verify the fresh DFlash block and the recycled chain together (2 drafter
                  blocks in one call); the next round recycles the winning chain's suffix
  oracle          as recycle, but the features / predictions over K come from an extra exact
                  target pass over the corrected chain (upper bound: no staleness; not free)

Reports tau = tokens per target verify, verify size, recycled-suffix statistics and a cost
estimate (drafter + verify per round; no copy pass in any policy).

  python scripts/exp30_recycle.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched, tree_nodes  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

VLLM_V = [(1, 12.7), (16, 13.3), (32, 14.1), (64, 14.2), (128, 15.8), (256, 19.0)]
HF_V = [(1, 20.8), (16, 22.0), (32, 22.8), (48, 23.3), (64, 23.6), (128, 25.5), (256, 29.0)]
DRAFT_MS = {"vLLM": (2.6, 3.1), "HF+graphs": (3.7, 4.4)}      # (one block, two blocks in one call)
POLICIES = ["dflash", "recycle_end", "recycle", "recycle_tree", "oracle"]


def interp(tab, q):
    for (q0, t0), (q1, t1) in zip(tab, tab[1:]):
        if q <= q1:
            return t0 + (t1 - t0) * (q - q0) / (q1 - q0)
    return tab[-1][1] * q / tab[-1][0]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--max-chain", type=int, default=64, help="cap on a recycled chain's length")
    ap.add_argument("--out", default="exp30_recycle")
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    lids = draft.target_layer_ids
    dev = next(target.parameters()).device
    log("models loaded")

    def run_target(prefix, chain):
        """Target over prefix + chain: features [1, len, W] and greedy predictions (list) per position."""
        z = torch.cat([prefix, torch.tensor(chain, dtype=prefix.dtype, device=dev)])
        out = target(z[None], output_hidden_states=True)
        return extract_context_feature(out.hidden_states, lids), out.logits[0].argmax(-1).tolist()

    def draft_from(ctx, toks, anchor_pos):
        """15 drafted tokens after anchor_pos; ctx covers positions [0, anchor_pos)."""
        return blocks_logits(draft, target, ctx, toks, [(anchor_pos, 0)], bs)[0].argmax(-1).tolist()

    stats = {p: {"tok": 0, "rounds": 0, "q": [], "ndraft": [], "rec_rounds": 0, "rec_won": 0, "j": [],
                 "k_ok": [], "ms": {}} for p in args.policies}
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            L = len(x)
            feats = extract_context_feature(target(x[None], output_hidden_states=True).hidden_states, lids)
            truth_all = x.tolist()
            for pol in args.policies:
                st = stats[pol]
                s = n0
                stale = None              # (s', K, stale feats [1, len(K)+1, W] at s'.., stale preds)
                while s < L - 1:
                    truth = truth_all[s + 1 :]
                    fresh = draft_logits_lagged(draft, target, feats, x, s, 0, bs).argmax(-1).tolist()
                    chains, ndraft = [], 1
                    rec = None
                    if pol != "dflash" and stale is not None and stale[0] == s and len(stale[1]):
                        _, K, sf, sp = stale
                        if pol == "oracle":   # exact target pass over the corrected chain (x[:s+1] + K)
                            of, op = run_target(x[: s + 1], K)
                            sf, sp = of[:, s:], op[s:]
                        if pol == "recycle_end":
                            ctx = torch.cat([feats[:, :s], sf[:, : len(K)]], 1)
                            zt = torch.cat([x[: s + 1], torch.tensor(K, dtype=x.dtype, device=dev)])
                            rec = K + draft_from(ctx, zt, s + len(K))
                        else:
                            j = next((i for i in range(len(K)) if sp[i] != K[i]), len(K))
                            st["j"].append(j)
                            ctx = torch.cat([feats[:, :s], sf[:, : j + 1]], 1)
                            zt = torch.cat([x[: s + 1], torch.tensor(K[:j] + [sp[j]], dtype=x.dtype, device=dev)])
                            rec = K[:j] + [sp[j]] + draft_from(ctx, zt, s + j + 1)
                        st["k_ok"].append(int(K[0] == truth[0]) if truth else 0)
                        st["rec_rounds"] += 1
                    if rec is None:
                        chains = [fresh]
                    elif pol == "recycle_tree":
                        chains, ndraft = [fresh, rec], 2
                    else:
                        chains = [rec]
                    m = [matched(c, truth) for c in chains]
                    w = max(range(len(chains)), key=lambda i: (m[i], i))   # ties -> the recycled chain
                    if rec is not None and chains[w] is rec and (len(chains) == 1 or m[w] > m[0]):
                        st["rec_won"] += 1
                    C, a = chains[w], m[w]
                    produced = min(a + 1, L - 1 - s)
                    st["tok"] += produced
                    st["rounds"] += 1
                    st["q"].append((tree_nodes(chains) if len(chains) > 1 else len(C)) + 1)
                    st["ndraft"].append(ndraft)
                    s_new = s + produced
                    stale = None
                    if a < len(C) and s_new < L - 1 and pol != "dflash":
                        # the verify pass over x[:s+1] + C: features / predictions from s' = s+a+1 on
                        vf, vp = run_target(x[: s + 1], C)
                        K = C[a + 1 :][: max(args.max_chain - bs, 0)]
                        stale = (s_new, K, vf[:, s_new : s_new + len(K) + 1], vp[s_new : s_new + len(K) + 1])
                    s = s_new
            log(f"[{d}] prompt done ({L - n0} tokens)")

    rows, crow = [], []
    for pol, st in stats.items():
        rows.append({"policy": pol, "tau": st["tok"] / max(st["rounds"], 1), "verify_q": mean(st["q"]),
                     "recycled_rounds": st["rec_rounds"] / max(st["rounds"], 1),
                     "recycled_won": st["rec_won"] / max(st["rec_rounds"], 1),
                     "suffix_first_ok": mean(st["k_ok"]), "mean_j": mean(st["j"])})
    base = {r["policy"]: r for r in rows}
    for eng, vt in (("vLLM", VLLM_V), ("HF+graphs", HF_V)):
        d1, d2 = DRAFT_MS[eng]
        tpt = {}
        for pol, st in stats.items():
            ms = sum((d1 if nd == 1 else d2) + interp(vt, q) for q, nd in zip(st["q"], st["ndraft"]))
            tpt[pol] = st["tok"] / ms
        for pol in stats:
            if pol == "oracle":
                continue      # needs an extra exact target pass per recycled round: not a free method
            crow.append({"engine": eng, "policy": pol, "x_vs_dflash": tpt[pol] / tpt["dflash"]})
    print_table(rows, ["policy", "tau", "verify_q", "recycled_rounds", "recycled_won", "suffix_first_ok", "mean_j"],
                "simulated greedy loop, T=0: tau = tokens per target verify (no copy in any policy)")
    print_table(crow, ["engine", "policy", "x_vs_dflash"], "estimated speedup over plain DFlash")
    save_json(args.out, {"args": vars(args), "rows": rows, "costs": crow, "base": base})


if __name__ == "__main__":
    main()
