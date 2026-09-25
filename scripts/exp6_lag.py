"""Experiment 6: can the diffusion drafter draft the next block before the middle model answers?

Two offline measurements on target-greedy trajectories, with middle-model (bnb4) features:

(a) Feature-lag tolerance. Draft a block anchored at s while the drafter only has context
    features for positions [0, s - g) (the last g positions have no features, as when
    drafting ahead of the middle model). Report mean accepted length vs g.

(b) Candidate futures. While the middle model checks the block at s, the drafter could
    draft next blocks for the most likely outcomes (cut after a tokens, correct token t at
    position a+1), scored by the drafter's own probabilities. Report how often the true
    outcome is among the top K candidates, and the acceptance of the block drafted for a
    hit candidate (features only up to s, i.e. lag a+1) vs drafting it after the answer
    (lag 0).

Truth = the target's greedy continuation (the middle model agrees with it ~97-98%).

  python scripts/exp6_lag.py --n 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import _draft_value, _output_head, _raw_input_embeddings, extract_context_feature  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


@torch.inference_mode()
def draft_block(draft, head_src, feats, ctx_end, anchor_pos, anchor_tok, bs):
    """Drafter logits [bs-1, V] for a block anchored at anchor_pos, context features [0, ctx_end)."""
    device = feats.device
    block = torch.full((1, bs), draft.mask_token_id, dtype=torch.long, device=device)
    block[0, 0] = anchor_tok
    noise = _raw_input_embeddings(head_src, block,
                                  float(_draft_value(draft.config, "input_embedding_scale", 1.0)))
    pos = torch.cat([torch.arange(ctx_end, device=device),
                     torch.arange(anchor_pos, anchor_pos + bs, device=device)]).unsqueeze(0)
    hidden = draft(target_hidden=feats[:, :ctx_end], noise_embedding=noise, position_ids=pos,
                   past_key_values=None, use_cache=False)[:, 1 - bs :, :]
    return draft.compute_logits(hidden, _output_head(head_src))[0]


def accepted(logits, truth):
    """Matching prefix length of drafted argmax vs truth tokens."""
    k = min(len(truth), logits.shape[0])
    if k == 0:
        return 0
    m = (logits[:k].argmax(-1) == truth[:k]).long()
    return int(m.cumprod(0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mid", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--lags", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--alt-k", type=int, default=4, help="alternative tokens per cut position")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    mid = load_mid(args.mid)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    kmax = max(args.ks)

    lag_rows, hit_rows = [], []
    for d in args.datasets:
        lag_acc = {g: [] for g in args.lags}
        hits = {k: 0 for k in args.ks}
        n_anchor = n_full = 0
        acc_hit_lagged, acc_hit_fresh, lag_of_hit = [], [], []
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            seq = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids
            T = seq.shape[1]
            mout = mid(seq, output_hidden_states=True, logits_to_keep=1)
            feats = extract_context_feature(mout.hidden_states, draft.target_layer_ids)
            x = seq[0]
            for s in range(n0, T - bs - 1, args.stride):
                truth = x[s + 1 : s + bs]
                # (a) lag tolerance
                for g in args.lags:
                    if s - g < 1:
                        continue
                    lag_acc[g].append(accepted(draft_block(draft, target, feats, s - g, s, x[s], bs), truth))
                # (b) candidate futures from the lag-0 block
                lg = draft_block(draft, target, feats, s, s, x[s], bs).float()
                a_true = accepted(lg, truth)
                n_anchor += 1
                if a_true >= bs - 1:
                    n_full += 1
                    continue
                lp = torch.log_softmax(lg, -1)
                top = lp.topk(args.alt_k + 1, -1)
                pref = torch.cat([torch.zeros(1, device=lp.device), top.values[:, 0].cumsum(0)])
                cands = []
                for a in range(bs - 1):              # cut after a accepted tokens
                    for j in range(1, args.alt_k + 1):
                        cands.append((float(pref[a] + top.values[a, j]), a, int(top.indices[a, j])))
                cands.sort(reverse=True)
                true_tok = int(x[s + a_true + 1])
                rank = next((r for r, (_, a, t) in enumerate(cands[:kmax])
                             if a == a_true and t == true_tok), None)
                for k in args.ks:
                    hits[k] += rank is not None and rank < k
                if rank is not None and rank < 4:
                    pnext = s + a_true + 1
                    if pnext + bs < T:
                        nt = x[pnext + 1 : pnext + bs]
                        acc_hit_lagged.append(accepted(draft_block(draft, target, feats, s, pnext, x[pnext], bs), nt))
                        acc_hit_fresh.append(accepted(draft_block(draft, target, feats, pnext, pnext, x[pnext], bs), nt))
                        lag_of_hit.append(pnext - s)
        for g in args.lags:
            lag_rows.append({"dataset": d, "lag": g, "mean_accept": mean(lag_acc[g]), "n": len(lag_acc[g])})
        hr = {"dataset": d, "anchors": n_anchor, "full_accept": n_full / max(n_anchor, 1)}
        for k in args.ks:
            hr[f"hit@{k}"] = hits[k] / max(n_anchor, 1)
        hr.update({"acc_hit_lagged": mean(acc_hit_lagged), "acc_hit_fresh": mean(acc_hit_fresh),
                   "mean_lag_of_hit": mean(lag_of_hit)})
        hit_rows.append(hr)
        print(f"[{d}] done", flush=True)

    print_table(lag_rows, ["dataset", "lag", "mean_accept", "n"],
                "(a) drafter acceptance when the last g context features are missing")
    print_table(hit_rows, ["dataset", "anchors", "full_accept"] + [f"hit@{k}" for k in args.ks]
                + ["acc_hit_lagged", "acc_hit_fresh", "mean_lag_of_hit"],
                "(b) candidate next blocks: hit rate and acceptance of the pre-drafted block")
    save_json("exp6_lag", {"args": vars(args), "lag": lag_rows, "hits": hit_rows})


if __name__ == "__main__":
    main()
