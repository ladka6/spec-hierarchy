"""Experiment 90: trees at every level of a block-drafter hierarchy, pruned by calibrated uncertainty (offline, T=0).

exp89 found the hierarchy (block drafter -> copy -> target) limited by the tokens the copy accepts per round (~6.5 of
15): the copy only checked the drafter's single chain. At batch 1 the copy is memory bound like the target, so it can
check a tree for about the cost of a chain, and a block drafter gives the tree for free (per-position marginals).

Per anchor s, for a copy-level tree budget N (chain = greedy 15 tokens, or best-first trees from the drafter's
marginals, DDTree-style, top-4 children per node):
  round r = 1..R   the drafter drafts at the spine end (real target features up to s, copy features after s); the copy
                   walks the tree: the spine grows by the longest prefix of the copy's greedy continuation that is in the
                   tree, plus the copy's next token (exact for greedy; computed with 16 cached copy steps).
  target hedges    none | free: drafter-tree branches the copy did not take, at the K spine positions with the lowest
                   copy margin (already drafted: no extra pass) | paid: exp89's hedges (copy top-2 + a 15-token redraft,
                   one extra drafter pass)
  target           one pass over spine + hedges; accepted = longest match with the target's greedy text + 1.
Cost (ms): verify t_V(nodes) as exp89; drafter --t-draft; copy --t-copy at 16 tokens, scaled like t_V for a tree.

  python scripts/exp90_treehier.py --copy rtn4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import heapq
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched, tree_nodes  # noqa: E402
from exp89_hedge import t_verify  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def best_first_tree(pr, budget, k=4):
    """Node set (tuples of tokens) of a best-first tree over independent marginals pr [D, V]."""
    D = pr.shape[0]
    tv, ti = pr.topk(k, -1)
    tv, ti = tv.log().tolist(), ti.tolist()
    nodes, heap = set(), [(-tv[0][c], (ti[0][c],)) for c in range(k)]
    heapq.heapify(heap)
    while heap and len(nodes) < budget:
        nl, path = heapq.heappop(heap)
        nodes.add(path)
        d = len(path)
        if d < D:
            for c in range(k):
                heapq.heappush(heap, (nl - tv[d][c], path + (ti[d][c],)))
    return nodes


def best_path_through(nodes, prefix, prl=None):
    """Greedy path down the tree from prefix (most likely child under the drafter's marginals), as a list."""
    p = tuple(prefix)
    while True:
        kids = [n for n in nodes if len(n) == len(p) + 1 and n[: len(p)] == p]
        if not kids:
            return list(p)
        if prl is not None and len(p) < len(prl):
            kids.sort(key=lambda n: -prl[len(p)][n[-1]])
        p = kids[0]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="rtn4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 16, 32, 64], help="0 = chain")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--t-draft", type=float, default=2.6)
    ap.add_argument("--t-copy", type=float, default=4.5)
    ap.add_argument("--out", default="exp90_treehier")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    copy = load_mid(args.copy)
    stops = stop_ids(target, tok)
    bs, D, R = draft.block_size, draft.block_size - 1, args.rounds
    lids = list(draft.target_layer_ids)
    dev = next(target.parameters()).device

    rec = defaultdict(list)          # (N, r, hedge, K) -> [(accepted, nodes)]
    added = defaultdict(list)        # N -> tokens added per round
    base, n_anchor = [], 0
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            horizon = R * bs + bs
            for s in range(n0, len(x) - horizon - 1, args.stride):
                n_anchor += 1
                truth = x[s + 1 : s + 1 + horizon].tolist()
                for N in args.budgets:
                    # copy prefix pass with cache: features of positions <= s, logits for the next token
                    co = copy(x[None, : s + 1], output_hidden_states=True, use_cache=True)
                    past = co.past_key_values
                    cf = extract_context_feature(co.hidden_states, lids).to(feats.dtype)      # [1, s+1, W]
                    last_logits = co.logits[0, -1].float()
                    P, info, spine_at, free = [], [], [], []
                    for r in range(R):
                        a = s + len(P)
                        z = torch.cat([x[: s + 1], torch.tensor(P, dtype=x.dtype, device=dev)])
                        mix = feats[:, :s] if r == 0 else torch.cat([feats[:, :s], cf[:, s:a]], 1)
                        lg = blocks_logits(draft, target, mix, z, [(a, 0)], bs)[0].float()
                        pr = torch.softmax(lg, -1)
                        y = pr.argmax(-1).tolist()
                        if r == 0 and N == args.budgets[0]:
                            base.append(matched(y, truth) + 1)
                        tree = {tuple(y[: i + 1]) for i in range(D)} if N == 0 else best_first_tree(pr, N)
                        # copy's greedy continuation with cached steps, until it leaves the tree: the last token is
                        # the copy's correction (every fed token's features are kept; the cache stays exact)
                        g, marg, alt2, hid, logits = [], [], [], [], last_logits
                        for t in range(bs):
                            pc = torch.softmax(logits, -1)
                            v2, i2 = pc.topk(2)
                            g.append(int(i2[0]))
                            marg.append(float(v2[0] - v2[1]))
                            alt2.append(int(i2[1]))
                            st = copy(torch.tensor([[g[-1]]], device=dev), past_key_values=past, use_cache=True,
                                      output_hidden_states=True)
                            past = st.past_key_values
                            hid.append(extract_context_feature(st.hidden_states, lids).to(feats.dtype))
                            logits = st.logits[0, -1].float()
                            if tuple(g) not in tree:
                                break
                        new = g
                        # free hedges: drafter-tree siblings of the copy's choice at each new position
                        prl = pr.tolist() if N else None
                        for k in range(len(new)):
                            pre = tuple(g[:k])
                            sib = [n for n in tree if len(n) == k + 1 and n[:k] == pre and n[k] != g[k]]
                            if sib:
                                if prl is not None:
                                    sib.sort(key=lambda n: -prl[k][n[k]])
                                path = best_path_through(tree, sib[0], prl)
                                free.append((marg[k], len(P) + k, P + list(path)))
                        info += [{"cm": marg[k], "c_alt": alt2[k]} for k in range(len(new))]
                        P += new
                        added[N].append(len(new))
                        spine_at.append(len(P))
                        cf = torch.cat([cf] + hid, 1)
                        last_logits = logits
                    cache = {}

                    def paid(i, alt):
                        if (i, alt) not in cache:
                            zb = torch.cat([x[: s + 1], torch.tensor(P[:i] + [alt], dtype=x.dtype, device=dev)])
                            mixb = torch.cat([feats[:, :s], cf[:, s : s + i + 1]], 1)
                            cont = blocks_logits(draft, target, mixb, zb, [(s + i + 1, 0)], bs)[0].argmax(-1).tolist()
                            cache[(i, alt)] = P[:i] + [alt] + cont
                        return cache[(i, alt)]

                    for r in range(R):
                        L = spine_at[r]
                        spine = P[:L]
                        for K in args.ks:
                            for hedge in (["none"] if K == 0 else ["free", "paid"]):
                                if hedge == "none":
                                    chains = [spine]
                                elif hedge == "free":
                                    cand = sorted((f for f in free if f[1] < L), key=lambda f: f[0])[:K]
                                    chains = [spine] + [f[2] for f in cand]
                                else:
                                    cand = sorted(range(L), key=lambda k: info[k]["cm"])[:K]
                                    chains = [spine] + [paid(k, info[k]["c_alt"]) for k in cand]
                                acc = max(matched(c, truth) for c in chains) + 1
                                rec[(N, r + 1, hedge, K)].append((acc, tree_nodes(chains)))
            print(f"[{dname}] {n_anchor} anchors so far", flush=True)

    t_base = t_verify(bs) + args.t_draft
    tau0 = sum(base) / len(base)
    rate0 = tau0 / t_base
    rows = []
    for N in args.budgets:
        rows.append({"copy tree": "chain" if N == 0 else N, "tokens added / round": sum(added[N]) / len(added[N])})
    print_table(rows, list(rows[0]), "1. copy-level tree: tokens the spine grows per round")
    rows = [{"copy tree": "-", "rounds": 0, "hedge": "plain DFlash", "K": 0, "accepted": tau0, "nodes": bs,
             "time ms": t_base, "tokens/ms": rate0, "vs DFlash": 1.0}]
    for (N, r, hedge, K), v in sorted(rec.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        acc = sum(a for a, _ in v) / len(v)
        nodes = sum(n for _, n in v) / len(v)
        t_c = args.t_copy * t_verify(max(N, D)) / t_verify(bs)
        t = t_verify(nodes) + r * (args.t_draft + t_c) + (args.t_draft if hedge == "paid" else 0.0)
        rows.append({"copy tree": "chain" if N == 0 else N, "rounds": r, "hedge": hedge, "K": K, "accepted": acc,
                     "nodes": nodes, "time ms": t, "tokens/ms": acc / t, "vs DFlash": acc / t / rate0})
    print_table(rows, list(rows[0]), f"2. tree hierarchy ({n_anchor} anchors, copy {args.copy})")
    best = max(rows[1:], key=lambda r: r["tokens/ms"])
    print(f"[best] {best}", flush=True)
    save_json(args.out, {"args": vars(args), "rows": rows,
                         "added": {str(N): sum(v) / len(v) for N, v in added.items()}})


if __name__ == "__main__":
    main()
