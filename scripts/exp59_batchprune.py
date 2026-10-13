"""Experiment 59: the copy as a pruner when target width stops being free (batch size), offline.

Everything so far is batch 1: the target's verify pass costs about the same for 16 or 64 nodes, so the best
round is a wide tree (exp56). At batch size B the target turns compute-bound and every verified node costs
time (DFlash: 5.1x at concurrency 1 -> 2.8x at 32). The 4-bit copy reads ~1/4 of the bytes, so it stays
memory-bound up to roughly 4x the batch at which the target does. Between the two knees width is free for the
copy and expensive for the target, so the copy should look at a WIDE candidate set (a DDTree + restarts) and the
target should verify only the chains the copy scores highest. This script measures the acceptance side of that
trade-off once (GPU) and replays it under measured cost curves for every batch size (CPU).

Phase `collect` (GPU; per anchor s on target-greedy trajectories, real features up to s):
  1. DFlash drafts block 1 (chain y, per-position marginals); a DDTree of --big nodes is built from them
     (its first K nodes are DDTree-K, best-first order).
  2. The copy runs ONCE over the whole DDTree as a tree forward on the target's prefix KV (one pass, free width
     for a memory-bound copy). It gives, per node, the copy's log-prob of the node's token under its parent, its
     own top-1/top-2 next token, and copy features at every node.
  3. Restarts (15 drafted tokens each, drafter conditioned on real features up to s and copy features along the
     kept path):
       pc      correction at the copy's first disagreement on the spine (exp56's pc; the chain case)
       tp      correction at the end of the copy's greedy WALK through the tree (the longest copy-accepted
               path, which may leave the spine), with the copy's top-1 there
       tw2     the same with the copy's top-2 token
       c1, c2  restarts at the drafter's two most likely stop positions on the spine (exp56's cK)
  Everything is saved per anchor (tree, scores, chains, matches against the trajectory), so any verification set
  can be scored later without the GPU.

Phase `replay` (CPU): for every batch size B with measured vLLM cost curves c_T(B, q) and c_C(B, q) (bench_vllm.py
--batches, real text), throughput of
  dflash          verify y (16 nodes)                                   no copy
  ddtree-K        the first K DDTree nodes                              no copy
  pc, pc+c2       y + chain restarts (exp56, copy over the chain: 16 copy nodes)
  prune-W[+r]     copy over the DDTree-K (K copy nodes); the target verifies the W nodes with the highest
                  copy path score, plus the restarts r in {tp, tw2, c1, c2} (their kept prefix is always kept)
  narrow          only the copy's accepted path + tp (W = 0)
tau = longest verified prefix of the trajectory + 1; tokens/ms at batch B = B * tau / round_ms(B);
reported relative to DFlash at the same B. The drafter is assumed memory-bound up to --drafter-knee requests
(cost constant), then linear: this is an assumption until bench_draft.py is run at batch > 1.

  python scripts/exp59_batchprune.py collect --copy bnb4:Qwen/Qwen3-8B --out exp59_dump
  python scripts/exp59_batchprune.py replay --dump results/batchprune/exp59_dump.json \
      --target-costs results/bench_vllm_batch.json --copy-costs results/bench_vllm_batch.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hspec.utils import mean, print_table, save_json  # noqa: E402

# ----------------------------------------------------------------------------------------------------- collect


def _prefix_cache(tcache, n):
    from transformers import DynamicCache

    return DynamicCache(ddp_cache_data=[(l.keys[:, :, :n], l.values[:, :, :n]) for l in tcache.layers])


def _spine_rows(tree):
    """Node indices of the drafter's top-1 chain inside the DDTree, in depth order."""
    ch = tree.children()
    rows, cur = [], -1
    while True:
        kids = ch.get(cur, {})
        if not kids:
            return rows
        # the spine child is the one added first at every depth (best-first order puts top-1 first)
        nxt = min(kids.values())
        rows.append(nxt)
        cur = nxt


def _walk(tree, pred, start=-1):
    """Greedy walk from node `start` with per-row predictions pred (row 0 = root). Returns (path, next token)."""
    ch = tree.children()
    path, cur = [], start
    while True:
        t = pred[cur + 1]
        nxt = ch.get(cur, {}).get(t)
        if nxt is None:
            return path, t
        path.append(nxt)
        cur = nxt


def _matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


@torch.inference_mode()
def collect_anchor(draft, target, copy, feats, tcache, x, s, bs, big):
    """One anchor. Returns a JSON-serialisable record (see module doc)."""
    from dflash.model import extract_context_feature

    from exp27_refresh import blocks_logits
    from hspec.lagtrain import draft_logits_lagged
    from hspec.tree import build_ddtree, verify_tree

    D = bs - 1
    truth = x[s + 1 : s + 2 * bs].tolist()
    L1 = draft_logits_lagged(draft, target, feats, x, s, 0, bs).float()
    lpd = torch.log_softmax(L1, -1)
    conf, y = torch.softmax(L1, -1).max(-1)
    y, conf = y.tolist(), conf.tolist()
    tree = build_ddtree(L1, big)
    N = len(tree)

    # the copy over the whole tree, on the target's prefix KV (one forward)
    co = verify_tree(copy, _prefix_cache(tcache, s), x[s], tree, s, hidden=True)
    logp = torch.log_softmax(co.logits[0].float(), -1)                       # [1+N, V]
    top2 = logp.topk(2, -1)
    ctok = top2.indices.tolist()
    cprob = top2.values.exp().tolist()
    cfeat = extract_context_feature(co.hidden_states, draft.target_layer_ids)[0].to(feats.dtype)   # [1+N, W]
    del co
    lp_copy = [float(logp[p + 1, t]) for t, p in zip(tree.tokens, tree.parents)]
    lp_draft = [float(lpd[d - 1, t]) for t, d in zip(tree.tokens, tree.depths)]
    # on-trajectory flag: the node's whole path equals the trajectory prefix
    ontruth = []
    for i, (t, p, d) in enumerate(zip(tree.tokens, tree.parents, tree.depths)):
        ontruth.append(bool(d - 1 < len(truth) and t == truth[d - 1] and (p < 0 or ontruth[p])))
    pred1 = [c[0] for c in ctok]
    spine = _spine_rows(tree)
    walk, t_walk = _walk(tree, pred1)
    # first copy disagreement on the spine (= chain pc of exp56); the prediction for spine[k] sits on the row of
    # its parent (row 0 = root for k = 0)
    j1 = next((k for k, r in enumerate(spine) if pred1[tree.parents[r] + 1] != tree.tokens[r]), None)

    def restart(path_rows, anchor_tok):
        """Drafter restart after the path (copy features along it) with anchor_tok at the next position."""
        rows = [0] + [r + 1 for r in path_rows]
        full = torch.cat([feats[:, :s], cfeat[rows][None]], 1)              # positions 0 .. s+len(path)
        z = torch.cat([x[: s + 1], torch.tensor([tree.tokens[r] for r in path_rows] + [anchor_tok], device=x.device)])
        a = s + len(path_rows) + 1
        Lr = blocks_logits(draft, target, full, z, [(a, 0)], bs)[0].float()
        pr, tk = torch.softmax(Lr, -1).max(-1)
        return [tree.tokens[r] for r in path_rows] + [anchor_tok] + tk.tolist(), pr.tolist()

    chains = {}
    if j1 is not None:
        kept = spine[:j1]
        corr_row = tree.parents[spine[j1]] + 1
        chains["pc"] = restart(kept, ctok[corr_row][0])
    last_row = (walk[-1] + 1) if walk else 0
    chains["tp"] = restart(walk, ctok[last_row][0])
    chains["tw2"] = restart(walk, ctok[last_row][1])
    # stop-position restarts on the spine (exp56's cK), copy features along the spine
    post, run = {}, 1.0
    for i in range(1, D + 1):
        run *= conf[i - 1]
        post[i] = run * (1.0 - conf[i]) if i < D else run
    cpos = sorted(post, key=lambda i: -post[i])[:2]
    for r_, i in enumerate(cpos):                                    # restart at position s+i: keep y[:i]
        if i - 1 < len(spine):
            chains[f"c{r_ + 1}"] = restart(spine[: i - 1], y[i - 1])
    rec = {
        "truth": truth, "y": y, "conf": conf, "j1": j1,
        "tree": {"tokens": tree.tokens, "parents": tree.parents, "depths": tree.depths,
                 "lp_copy": lp_copy, "lp_draft": lp_draft, "ontruth": ontruth,
                 "ctop1": pred1, "cp1": [c[0] for c in cprob]},
        "walk": walk, "spine": spine,
        "chains": {k: {"tokens": v[0], "conf": v[1], "match": _matched(v[0], truth)} for k, v in chains.items()},
        "copy_agree": float(torch.tensor([pred1[tree.parents[r] + 1] == tree.tokens[r] for r in spine]).float().mean())
        if spine else float("nan"),
    }
    return rec


@torch.inference_mode()
def collect(args):
    from dflash.model import extract_context_feature

    from hspec.data import encode, load_prompts, stop_ids
    from hspec.models import load_draft, load_mid, load_target, load_tokenizer
    from hspec.pipeline import two_stage_generate

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    copy = load_mid(args.copy)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    recs = []
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            tout = target(x[None], output_hidden_states=True, logits_to_keep=1, use_cache=True)
            feats = extract_context_feature(tout.hidden_states, draft.target_layer_ids)
            tcache = tout.past_key_values
            del tout
            for s in range(n0, len(x) - 2 * bs - 1, args.stride):
                rec = collect_anchor(draft, target, copy, feats, tcache, x, s, bs, args.big)
                rec["dataset"] = d
                recs.append(rec)
            print(f"[{d}] {len(recs)} anchors", flush=True)
    quick = replay_rows(recs, costs=None, batches=[1], drafter_ms=2.6, drafter_knee=1e9, restart_ms=1.0,
                        widths=args.widths, ks=args.ks)
    print_table(quick, ["batch", "policy", "tau", "q_T", "q_C", "x_vs_dflash"],
                f"collect: acceptance under exp56's batch-1 vLLM costs (copy {args.copy}, big={args.big})")
    save_json(args.out, {"args": vars(args), "bs": bs, "anchors": recs})


# ------------------------------------------------------------------------------------------------------ replay

VTAB = [(1, 12.7), (16, 13.3), (32, 14.1), (64, 14.2), (128, 15.8), (256, 19.0)]     # exp29 batch-1 fallback
CTAB = [(1, 3.0), (16, 3.0), (32, 3.5), (64, 4.5), (128, 6.0), (256, 9.0)]           # 4-bit AWQ, rough fallback


def _interp(tab, q):
    tab = sorted(tab)
    if q <= tab[0][0]:
        return tab[0][1]
    for (a, ta), (b, tb) in zip(tab, tab[1:]):
        if q <= b:
            return ta + (tb - ta) * (q - a) / (b - a)
    (a, ta), (b, tb) = tab[-2], tab[-1]
    return tb + (tb - ta) * (q - b) / (b - a)                       # linear beyond the last point (compute-bound)


def load_curves(path, model_substr):
    """{B: [(q, ms), ...]} from bench_vllm.py output (real_q_ms per batch, q_ms for batch 1)."""
    if not path:
        return None
    rows = json.loads(Path(path).read_text())["rows"]
    want_awq = "awq" in model_substr.lower()
    row = next((r for r in rows if model_substr.lower() in r["model"].lower()
                and ("awq" in r["model"].lower()) == want_awq), None)
    if row is None:
        raise SystemExit(f"no row for '{model_substr}' in {path}: {[r['model'] for r in rows]}")
    curves = {}
    for B, qm in row.get("real_q_ms", {}).items():
        curves[int(B)] = sorted((int(q), v) for q, v in qm.items())
    if 1 not in curves and row.get("q_ms"):
        curves[1] = sorted((int(q), v) for q, v in row["q_ms"].items())
    return curves


def _union_nodes(seqs):
    nodes = set()
    for c in seqs:
        for k in range(1, len(c) + 1):
            nodes.add(tuple(c[:k]))
    return nodes


def verified_sets(rec, widths, ks):
    """Per policy: (tau, target nodes q_T incl. anchor, copy nodes q_C incl. anchor, restart calls)."""
    tr, truth, y = rec["tree"], rec["truth"], rec["y"]
    N = len(tr["tokens"])
    paths = []                                                      # token path of every node
    for i, (t, p) in enumerate(zip(tr["tokens"], tr["parents"])):
        paths.append((paths[p] if p >= 0 else ()) + (t,))
    chains = rec["chains"]

    def tau_of(node_set):
        best = 0
        for pth in node_set:
            m = _matched(list(pth), truth)
            if m == len(pth):
                best = max(best, m)
        return best + 1

    out = {}
    out["dflash"] = (_matched(y, truth) + 1, 16, 0, 0)
    for K in ks:
        ns = set(paths[:K])
        out[f"ddtree-{K}"] = (tau_of(ns), len(ns) + 1, 0, 0)
    # chain refresh (copy over the 16-token chain only)
    for name, parts in (("pc", ["pc"]), ("pc+c2", ["pc", "c1", "c2"])):
        seqs = [y] + [chains[k]["tokens"] for k in parts if k in chains]
        ns = _union_nodes(seqs)
        out[name] = (tau_of(ns), len(ns) + 1, 16, 1)
    # copy over DDTree-K, target verifies top-W nodes by copy path score + restarts
    score = []
    for i, p in enumerate(tr["parents"]):
        score.append(tr["lp_copy"][i] + (score[p] if p >= 0 else 0.0))
    for K in ks:
        order = sorted(range(K), key=lambda i: -score[i])
        for W in widths:
            keep = set()
            for i in order[:W]:
                keep |= _union_nodes([paths[i]])                  # prefix-closed
            for rname, parts in (("", []), ("+tp", ["tp"]), ("+tp+tw2", ["tp", "tw2"]),
                                 ("+tp+c2", ["tp", "c1", "c2"]), ("+tp+tw2+c2", ["tp", "tw2", "c1", "c2"])):
                ns = set(keep) | _union_nodes([chains[k]["tokens"] for k in parts if k in chains])
                if not ns and not parts:
                    continue
                out[f"prune{K}-W{W}{rname}"] = (tau_of(ns), len(ns) + 1, K + 1, 1 if parts else 0)
    walk_path = _union_nodes([chains["tp"]["tokens"]]) if "tp" in chains else set()
    out["narrow"] = (tau_of(walk_path), len(walk_path) + 1, max(ks) + 1, 1)
    return out


def replay_rows(recs, costs, batches, drafter_ms, drafter_knee, restart_ms, widths, ks):
    """costs: (target_curves, copy_curves) or None for the batch-1 tables."""
    per = defaultdict(list)
    for rec in recs:
        for pol, v in verified_sets(rec, widths, ks).items():
            per[pol].append(v)
    rows = []
    for B in batches:
        if costs is None:
            cT = lambda q: _interp(VTAB, q)                                    # noqa: E731
            cC = lambda q: _interp(CTAB, q)                                    # noqa: E731
        else:
            tcur, ccur = costs
            if B not in tcur or B not in ccur:
                print(f"batch {B}: no measured curve (target {sorted(tcur)}, copy {sorted(ccur)}), skipped")
                continue
            cT = lambda q, t=tcur[B]: _interp(t, q)                             # noqa: E731
            cC = lambda q, c=ccur[B]: _interp(c, q)                             # noqa: E731
        dm = drafter_ms * max(1.0, B / drafter_knee)
        base = None
        for pol, vals in per.items():
            tau = mean(v[0] for v in vals)
            qT = mean(v[1] for v in vals)
            qC = mean(v[2] for v in vals)
            rcalls = mean(v[3] for v in vals)
            ms = dm + (cC(qC) if qC else 0.0) + rcalls * restart_ms + cT(qT)
            tpt = B * tau / ms
            if pol == "dflash":
                base = tpt
            rows.append({"batch": B, "policy": pol, "tau": tau, "q_T": qT, "q_C": qC, "ms": ms, "tok_per_ms": tpt})
        for r in rows:
            if r["batch"] == B:
                r["x_vs_dflash"] = r["tok_per_ms"] / base
    return rows


def replay(args):
    recs = []
    for p in args.dump:
        recs += json.loads(Path(p).read_text())["anchors"]
    costs = None
    if args.target_costs:
        costs = (load_curves(args.target_costs, args.target_model), load_curves(args.copy_costs, args.copy_model))
    rows = replay_rows(recs, costs, args.batches, args.drafter_ms, args.drafter_knee, args.restart_ms,
                       args.widths, args.ks)
    cols = ["batch", "policy", "tau", "q_T", "q_C", "ms", "x_vs_dflash"]
    for B in args.batches:
        rb = [r for r in rows if r["batch"] == B]
        if not rb:
            continue
        rb.sort(key=lambda r: -r["x_vs_dflash"])
        print_table(rb[: args.top], cols, f"batch {B}: best {args.top} policies ({len(recs)} anchors)")
    best = [{"batch": B, "best": max((r for r in rows if r["batch"] == B), key=lambda r: r["x_vs_dflash"])["policy"],
             "x_vs_dflash": max(r["x_vs_dflash"] for r in rows if r["batch"] == B),
             "wide(pc+c2)": next(r["x_vs_dflash"] for r in rows if r["batch"] == B and r["policy"] == "pc+c2"),
             "ddtree-64": next((r["x_vs_dflash"] for r in rows if r["batch"] == B and r["policy"] == "ddtree-64"), float("nan")),
             "narrow": next(r["x_vs_dflash"] for r in rows if r["batch"] == B and r["policy"] == "narrow")}
            for B in args.batches if any(r["batch"] == B for r in rows)]
    print_table(best, list(best[0]) if best else [], "crossover: best policy per batch size, speed vs DFlash at the same batch")
    save_json(args.out, {"args": vars(args), "rows": rows, "best": best})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--target", default="Qwen/Qwen3-8B")
    c.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    c.add_argument("--copy", required=True, help="load_mid spec (needs the target's architecture: bnb4/ao4/lora)")
    c.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    c.add_argument("--n", type=int, default=10)
    c.add_argument("--max-new", type=int, default=512)
    c.add_argument("--stride", type=int, default=8)
    c.add_argument("--big", type=int, default=64, help="DDTree nodes the copy looks at")
    c.add_argument("--widths", nargs="+", type=int, default=[0, 4, 8, 16, 32, 64])
    c.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    c.add_argument("--out", default="exp59_dump")
    r = sub.add_parser("replay")
    r.add_argument("--dump", nargs="+", required=True)
    r.add_argument("--target-costs", default="", help="bench_vllm.py json with real_q_ms per batch for the target")
    r.add_argument("--copy-costs", default="", help="bench_vllm.py json for the AWQ copy")
    r.add_argument("--target-model", default="Qwen3-8B", help="substring of the model name in --target-costs (not AWQ)")
    r.add_argument("--copy-model", default="AWQ")
    r.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    r.add_argument("--drafter-ms", type=float, default=2.6)
    r.add_argument("--drafter-knee", type=float, default=16, help="drafter cost constant up to this batch, then linear")
    r.add_argument("--restart-ms", type=float, default=1.0)
    r.add_argument("--widths", nargs="+", type=int, default=[0, 4, 8, 16, 32, 64])
    r.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    r.add_argument("--top", type=int, default=12)
    r.add_argument("--out", default="exp59_replay")
    args = ap.parse_args()
    if args.phase == "replay" and "AWQ" in args.target_model.upper():
        raise SystemExit("--target-model must not match the AWQ row")
    (collect if args.phase == "collect" else replay)(args)


if __name__ == "__main__":
    main()
