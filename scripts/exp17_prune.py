"""Experiment 17: early-exit tree pruning, offline.

DDTree's acceptance grows with the tree budget, but a full target pass stops being flat in
cost beyond ~128 tokens. Pruning: build a big DDTree (official builder, B_big nodes), run
only the target's first k layers + exit head on all of it, keep the K most promising nodes
(path score from the exit, optionally plus the drafter's), and verify only those with the
upper layers. Lossless; the question is how much of the big tree's acceptance survives.

Offline on target-greedy trajectories (official DFlash loop), anchors every `stride`
positions: the full target accepts exactly the longest tree path that matches the
trajectory. Reports accepted tokens per pass (incl. the bonus token) for DDTree-K,
DDTree-B_big and pruned B_big -> K, per exit layer and scoring rule.

  python scripts/exp17_prune.py --target facebook/layerskip-llama3-8B --draft .../drafter_ls
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

from dflash.model import extract_context_feature  # noqa: E402
from ddtree_official.ddtree import build_ddtree_tree  # noqa: E402
from ddtree_official.dflash import dflash_generate as off_dflash  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def walk(child_maps, keep, truth):
    """Accepted tokens (+1 bonus) along the truth path using only nodes in `keep`."""
    idx, a = 0, 0
    for t in truth:
        nxt = child_maps[idx].get(t)
        if nxt is None or nxt not in keep:
            break
        idx, a = nxt, a + 1
    return a + 1


def lower_forward(target, layers, ids_prefix, tree_ids, tree_pos, vis):
    """Target layers [0, max(layers)) over prefix (causal) + tree (prefix + ancestors), no cache.
    Returns {k: exit log-probs [1+T, V] for the tree rows after k layers}."""
    base = target.model
    dev = ids_prefix.device
    P, Q = ids_prefix.shape[1], tree_ids.shape[1]
    ids = torch.cat([ids_prefix, tree_ids], 1)
    pos = torch.cat([torch.arange(P, device=dev)[None], tree_pos], 1)
    m = torch.zeros((P + Q, P + Q), dtype=torch.bool, device=dev)
    m[:P, :P] = torch.ones((P, P), dtype=torch.bool, device=dev).tril()
    m[P:, :P] = True
    m[P:, P:] = vis.to(dev)
    h = base.embed_tokens(ids)
    pe = base.rotary_emb(h, pos)
    out = {}
    for i in range(max(layers)):
        h = base.layers[i](h, attention_mask=m[None, None], position_ids=pos, position_embeddings=pe)
        if isinstance(h, tuple):
            h = h[0]
        if i + 1 in layers:
            out[i + 1] = torch.log_softmax(target.lm_head(base.norm(h[0, P:])).float(), -1)
    return out


def prune(parents, node_tok, exit_lp, draft_lp, K, w_draft):
    """Keep K nodes (1-based tree indices) by path score, prefix-closed. exit_lp: [1+T, V]
    (row r = the exit's next-token log-probs after tree row r); draft_lp: per-node drafter
    log-prob of its token."""
    T = len(node_tok)
    score = [0.0] * (T + 1)
    for i in range(1, T + 1):
        par = parents[i]
        score[i] = score[par] + float(exit_lp[par, node_tok[i - 1]]) + w_draft * draft_lp[i - 1]
    keep = set()
    for i in sorted(range(1, T + 1), key=lambda j: -score[j]):
        if len(keep) >= K:
            break
        if parents[i] == 0 or parents[i] in keep:
            keep.add(i)
    return keep


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="facebook/layerskip-llama3-8B")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--big", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--keep", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--exits", nargs="+", type=int, default=[16, 20])
    ap.add_argument("--w-draft", nargs="+", type=float, default=[0.0, 1.0])
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    Bmax = max(args.big)
    budgets = sorted(set(args.big + args.keep))
    acc = defaultdict(list)
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = off_dflash(draft, target, ids, draft.mask_token_id, args.max_new, bs, stops).output_ids[0].to(ids.device)
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            for s in range(n0, len(x) - bs - 1, args.stride):
                truth = x[s + 1 : s + 1 + D].tolist()
                lg = draft_logits_lagged(draft, target, feats, x, s, 0, bs).float()
                lq = torch.log_softmax(lg, -1).cpu()
                node_ids, depths, parents, child_maps, vis, _ = build_ddtree_tree(lg, Bmax)
                node_tok = node_ids.tolist()
                dl = lq[depths - 1, node_ids].tolist() if len(node_tok) else []
                T = len(node_tok)
                for B in budgets:                              # official tree at budget B = first B nodes
                    acc[("ddtree", B, 0, 0.0, d)].append(walk(child_maps, set(range(1, min(B, T) + 1)), truth))
                for Bb in args.big:
                    nb = min(Bb, T)
                    tids = torch.tensor([[int(x[s])] + node_tok[:nb]], device=x.device)
                    tpos = torch.tensor([[s] + [s + int(dd) for dd in depths[:nb].tolist()]], device=x.device)
                    lp = lower_forward(target, args.exits, x[None, :s], tids, tpos, vis[: nb + 1, : nb + 1])
                    for k in args.exits:
                        for K in args.keep:
                            if K >= Bb:
                                continue
                            for w in args.w_draft:
                                keep = prune(parents[: nb + 1], node_tok[:nb], lp[k], dl[:nb], K, w)
                                acc[("prune", Bb, k, w, d, K)].append(walk(child_maps, keep, truth))
                    del lp
        print(f"[{d}] done", flush=True)

    rows = []
    dd_all = {B: mean(sum((acc[("ddtree", B, 0, 0.0, dd)] for dd in args.datasets), [])) for B in budgets}
    for B in budgets:
        rows.append({"method": f"ddtree-{B}", "accepted": dd_all[B], "vs_ddtree_K": float("nan")})
    for Bb in args.big:
        for k in args.exits:
            for K in args.keep:
                if K >= Bb:
                    continue
                for w in args.w_draft:
                    v = mean(sum((acc[("prune", Bb, k, w, dd, K)] for dd in args.datasets), []))
                    rows.append({"method": f"prune {Bb}->{K} exit{k} w_draft{w:g}", "accepted": v,
                                 "vs_ddtree_K": v / dd_all[K], "vs_ddtree_big": v / dd_all[Bb]})
    print_table(rows, ["method", "accepted", "vs_ddtree_K", "vs_ddtree_big"],
                "accepted tokens per pass (incl. bonus): DDTree at each budget and pruned big trees")
    save_json("exp17_prune", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
