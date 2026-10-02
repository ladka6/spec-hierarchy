"""Experiment 25: suspected-prefix pipelining, offline. How often can the next round start early?

Pipelined speculation: while round t (tree T_t, anchor s) is still being verified, bet on where
it will stop (a "suspected" accepted node n of T_t, depth k) and draft round t+1's tree rooted at n
right away. The S tokens (the path to n) have token ids but no target features yet, so round t+1 is
drafted with lag k: features up to s, the k suspected tokens as inputs. Several hypotheses can
share one verify pass (verification is ~free up to ~128 nodes at batch 1): the next tree is the
union of small subtrees, one per hypothesis, budget split evenly.

A hypothesis n succeeds when (1) the path to n is a prefix of the target's trajectory and (2) its
subtree contains the rest of round t's accepted tokens plus round t's correction / bonus token.
Then round t+1's work stands; its progress = tokens it adds beyond round t (incl. its own bonus).
Otherwise round t+1 is redone fresh after round t returns.

Hypothesis choice: nodes of T_t ranked by the drafter's probability that acceptance stops exactly
there, P(path) * (1 - sum of its children's marginal probs) ("conf"); "oracle" = the true stop node.

Pipeline model (2 GPUs, target split in two halves; d = draft+tree ms, v = verify ms):
  success round d + v/2, failure round d + v;  x = [q*prog + (1-q)*tau] / [q(d+v/2) + (1-q)(d+v)]
  relative to sequential tau / (d + v).

  python scripts/exp25_suspect.py --lag-draft /scratch-shared/$USER/hspec_lagft/drafter_lag16
  python scripts/exp25_suspect.py --subs orig=z-lab/Qwen3-8B-DFlash-b16:depth depth18=/path/drafter_depth18:depth
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp15_blocktree import accepted, tree_paths  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.feathead import block_hidden, context_vectors, draft_logits_ctx, load_head, predict_context  # noqa: E402
from hspec.lagtrain import deep_columns, draft_logits_depth, draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.tree import build_ddtree  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def node_paths(tree):
    """Token path (tuple) of every node, in node order."""
    out = []
    for t, p in zip(tree.tokens, tree.parents):
        out.append((out[p] if p >= 0 else ()) + (t,))
    return out


def stop_scores(tree, logq):
    """[(path, P(acceptance stops exactly at this node))] incl. the root (empty path).
    logq: [D, V] log-probs of the drafter's per-position marginals."""
    paths = node_paths(tree)
    lp = {(): 0.0}
    for path in paths:
        lp[path] = lp[path[:-1]] + float(logq[len(path) - 1, path[-1]])
    child_mass = defaultdict(float)
    for path in paths:
        child_mass[path[:-1]] += math.exp(float(logq[len(path) - 1, path[-1]]))
    return [(pth, math.exp(lp[pth]) * max(0.0, 1.0 - child_mass[pth])) for pth in [()] + paths]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--subs", nargs="+", default=None,
                    help="next-round drafters name=path:mode, mode token (suspected tokens get no features) or "
                         "depth (they get the lower half's features, 2-GPU pipeline), fresh (real features: upper bound) "
                         "or pred@HEAD.pt (features predicted by a feature head); default orig + --lag-draft")
    ap.add_argument("--lag-draft", default=None)
    ap.add_argument("--depth-exit", type=int, default=18, help="layers on the first GPU (depth mode)")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--budget", type=int, default=128, help="round-t tree and next-round tree budget")
    ap.add_argument("--ms", nargs="+", type=int, default=[1, 2, 4, 8], help="number of hypotheses")
    ap.add_argument("--d-ms", type=float, default=8.4, help="draft + tree ms per round (exp24)")
    ap.add_argument("--v-ms", type=float, default=42.5, help="verify ms per round (exp24)")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    subs = args.subs or ([f"orig={args.draft}:token"] + ([f"lag={args.lag_draft}:token"] if args.lag_draft else []))
    drafters, loaded = {}, {args.draft: draft}
    for spec in subs:
        name, rest = spec.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        if path not in loaded:
            loaded[path] = load_draft(path)
        drafters[name] = (loaded[path], mode)
    heads = {}
    for name, (dm, mode) in drafters.items():
        if mode.startswith("pred@"):
            heads[name] = load_head(mode[5:], next(dm.parameters()).device, next(dm.parameters()).dtype)
    deep = None
    stops = stop_ids(target, tok)
    bs = draft.block_size
    B = args.budget
    mmax = max(args.ms)

    rec = defaultdict(list)       # (method, drafter, m) -> [(success, progress)]
    taus = []
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            if deep is None:
                deep = deep_columns(draft, args.depth_exit, feats.shape[-1], feats.device)
            c_real = context_vectors(draft, feats) if heads else None
            for s in range(n0, len(x) - 2 * bs - 2, args.stride):
                truth = x[s + 1 : s + 2 * bs].tolist()
                lg = draft_logits_lagged(draft, target, feats, x, s, 0, bs)
                logq = torch.log_softmax(lg.float(), -1).cpu()
                tree = build_ddtree(lg, B)
                a = accepted(set(tree_paths(tree, logq)), truth)     # accepted + bonus
                taus.append(a)
                ranked = sorted(stop_scores(tree, logq), key=lambda kv: -kv[1])
                true_node = tuple(truth[: a - 1])
                cand = {"conf": [pth for pth, _ in ranked[:mmax]], "oracle": [true_node]}
                # subtree for each candidate node on the true path, per drafter and budget
                cache, logit_cache, chat = {}, {}, {}

                def outcome(pth, dname, budget):
                    k = len(pth)
                    if pth != tuple(truth[:k]):
                        return None                                   # hypothesis off the true path
                    key = (k, dname, budget)
                    if key not in cache:
                        if (k, dname) not in logit_cache:
                            dm, mode = drafters[dname]
                            if mode == "depth":
                                sl = draft_logits_depth(dm, target, feats, x, s + k, k, bs, deep)
                            elif mode == "fresh" or (mode.startswith("pred@") and k == 0):
                                sl = draft_logits_lagged(dm, target, feats, x, s + k, 0, bs)
                            elif mode.startswith("pred@"):
                                if dname not in chat:          # c_hat for positions s .. s+bs-2
                                    hid = block_hidden(dm, target, feats, x, [s], bs)[0]
                                    chat[dname] = predict_context(heads[dname], dm, target, hid, x, s, bs)
                                ctx = torch.cat([c_real[:, :s], chat[dname][:k][None].to(c_real.dtype)], 1)
                                sl = draft_logits_ctx(dm, target, ctx, x, s + k, bs)
                            else:
                                sl = draft_logits_lagged(dm, target, feats, x, s + k, k, bs)
                            logit_cache[(k, dname)] = (sl, torch.log_softmax(sl.float(), -1).cpu())
                        sl, sq = logit_cache[(k, dname)]
                        sub = build_ddtree(sl, max(budget, 1))
                        cache[key] = accepted(set(tree_paths(sub, sq)), x[s + k + 1 : s + k + 2 * bs].tolist())
                    sub_acc = cache[key]
                    need = a - k                                      # rest of round t + its bonus
                    if sub_acc - 1 < need:                            # matched tokens must cover them
                        return None
                    return sub_acc - need                             # progress of round t+1

                for dname in drafters:
                    for m in args.ms:
                        budget = B // m
                        for method, nodes in cand.items():
                            if method == "oracle" and m != 1:
                                continue
                            best = None
                            for pth in nodes[:m]:
                                r = outcome(pth, dname, budget)
                                if r is not None and (best is None or r > best):
                                    best = r
                            rec[(method, dname, m)].append((best is not None, best or 0))
        print(f"[{d}] done", flush=True)

    tau = mean(taus)
    dd, v = args.d_ms, args.v_ms
    rows = []
    for (method, dname, m), rs in sorted(rec.items()):
        q = mean(float(ok) for ok, _ in rs)
        prog = mean(pg for ok, pg in rs if ok) if q > 0 else 0.0
        tok_r = q * prog + (1 - q) * tau
        t_r = q * (dd + v / 2) + (1 - q) * (dd + v)
        rows.append({"hyp": method, "drafter": dname, "m": m, "sub_budget": B // m, "q": q,
                     "prog_ok": prog, "tau_fresh": tau,
                     "x_vs_seq": (tok_r / t_r) / (tau / (dd + v)),
                     "x_if_q1": (prog / (dd + v / 2)) / (tau / (dd + v)) if q > 0 else float("nan")})
    print(f"{len(taus)} anchors, sequential DDTree-{B} tau {tau:.2f}", flush=True)
    print_table(rows, ["hyp", "drafter", "m", "sub_budget", "q", "prog_ok", "tau_fresh", "x_vs_seq", "x_if_q1"],
                f"suspected-prefix pipelining (2-GPU halves model, d={dd} ms, v={v} ms)")
    save_json("exp25_suspect", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
