"""Experiment 62: aligned future features from the context (retrieval refresh), offline, no training.

The copy refresh gives the drafter COMPUTED stand-in features for the drafted positions (a 4-bit copy runs over them).
When the output repeats earlier text, the target has already computed those features: if the anchor token x_s
matches an earlier position i of the context and the text continues as it did then, the target's features at
i, i+1, ..., i+14 are stand-ins for its features at s, s+1, ..., s+14, and x[i+1 .. i+15] are the tokens. So the
round can (1) propose the looked-up tokens and (2) restart the drafter after them with the looked-up features as
context for those positions, without running any copy: a LOOKED-UP refresh. Slot k of the restart's context is
aligned with the retrieved position i+k.

Match position i (anchor = position s, token x_s):
  ngram   latest earlier end position i < s-15 whose last n tokens equal the last n tokens before and at s
          (prompt-lookup rule, longest n in 8..2)
  attn    the position the target's query at s attends to most (all layers and KV heads, each layer normalised,
          sinks and the last 64 positions excluded), shifted by one so that x_i = x_s if possible (induction heads
          attend to the token after the earlier occurrence)

Per anchor, with the drafter's own context = one of --bases (exp61 notation, e.g. recent-1024-orig,
attn-1024-compact, full):
  block1        the DFlash chain y
  copy          exp61's refresh with the KV-sparse 4-bit copy (pc + c1 + c2)            [needs --copy]
  lookup        y + the looked-up tokens r = x[i+1 .. i+15]                              (prompt lookup on top)
  aligned       y + r + restarts after r[:8] and after r[:15], each drafted from looked-up features
                feats[i .. i+L-1] placed at positions s .. s+L-1                          (the method)
  aligned+copy  union of aligned and copy                                                [needs --copy]
  oracle        as aligned, but the restarts get the target's REAL features at s .. s+L-1 (upper bound for the
                feature part; same tokens r)
tau = longest chain prefix matching the trajectory + 1 (tree union). Also: share of anchors with a match, how far r
matches the trajectory, cosine of the looked-up features to the real ones where r is right, and a projected
speed vs DFlash (bytes model of exp60; lookup search assumed free, attention scores from the verify pass).
Every anchor is saved with the prompt's copy rate, so results can be split by how much the output really copies.

  python scripts/exp62_aligned.py --ctx 8192 16384 --settings recall history --match ngram attn
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp60_sparsekv import build_history, greedy_long  # noqa: E402
from exp61_sparsedraft import build_recall, copy_rate, matched  # noqa: E402
from hspec.sparsedraft import context_for, draft_block, score_positions, select_positions  # noqa: E402
from hspec.sparsekv import attention_scores, prefill_features, select_indices, sparse_cache  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

VTAB = [(1, 12.7), (16, 13.3), (32, 14.1), (64, 14.2), (128, 15.8), (256, 19.0)]


def v(q):
    for (a, ta), (b, tb) in zip(VTAB, VTAB[1:]):
        if q <= b:
            return ta + (tb - ta) * (q - a) / (b - a)
    return VTAB[-1][1]


def tree_nodes(chains):
    nodes = set()
    for c in chains:
        for k in range(1, len(c) + 1):
            nodes.add(tuple(c[:k]))
    return len(nodes)


def ngram_match(arr: np.ndarray, s: int, D: int, nmax: int = 8, nmin: int = 2):
    """Latest end position i <= s-D-1 with arr[i-n+1 .. i] == arr[s-n+1 .. s] for the longest n in nmax..nmin."""
    hi = s - D - 1                                     # i + D must stay strictly before s (continuation is known)
    for n in range(nmax, nmin - 1, -1):
        if hi - n + 1 < 0 or s - n + 1 < 0:
            continue
        pat = arr[s - n + 1 : s + 1]
        L = hi - n + 2                                 # candidate starts 0 .. hi-n+1
        m = np.ones(L, dtype=bool)
        for t in range(n):
            m &= arr[t : t + L] == pat[t]
        idx = np.flatnonzero(m)
        if len(idx):
            return int(idx[-1] + n - 1), n
    return None, 0


def attn_match(layer_scores_last, x_list, s: int, D: int, sinks: int = 4, local: int = 64):
    """Position the query at s attends to most (outside sinks and the last `local` positions), aligned so that
    x[i] == x[s] when one of the two neighbours allows it. Returns i or None."""
    tot = score_positions(layer_scores_last)          # [s]
    hi = s - max(local, D + 1)
    if hi <= sinks + 1:
        return None
    j = int(tot[sinks:hi].argmax()) + sinks
    for i in (j - 1, j):
        if 0 <= i and i + D < s and x_list[i] == x_list[s]:
            return i
    i = j - 1
    return i if 0 <= i and i + D < s else None


@torch.inference_mode()
def eval_anchor(draft, target, copy, feats, tcache, x, x_np, s, bs, args):
    """Returns {(base, match): {variant: (tau, nodes)}, ...} plus match diagnostics."""
    from dflash.model import extract_context_feature

    D = bs - 1
    xl = x.tolist()
    truth = xl[s + 1 : s + 2 * bs + D]
    need_attn = any(b.startswith("attn") for b in args.bases) or "attn" in args.match or copy is not None
    layer_scores = attention_scores(target, tcache, x, s, args.probe) if need_attn else None
    last_q = attention_scores(target, tcache, x, s, 1) if "attn" in args.match else None
    score = score_positions(layer_scores) if layer_scores is not None else None

    matches = {}
    for m in args.match:
        if m == "ngram":
            i, n = ngram_match(x_np, s, D)
        else:
            i, n = attn_match(last_q, xl, s, D), 0
        matches[m] = i

    out, diag = {}, {}
    for m, i in matches.items():
        if i is None:
            diag[m] = None
            continue
        r = xl[i + 1 : i + 1 + D]
        ok = matched(r, truth)
        real = feats[0, s : s + D].float()
        look = feats[0, i : i + D].float()
        good = min(ok + 1, D)                          # features at s+k are comparable while tokens up to s+k agree
        cos = float(torch.nn.functional.cosine_similarity(look[:good], real[:good], dim=-1).mean())
        diag[m] = {"r_match": ok, "cos": cos, "n": n if m == "ngram" else None, "dist": s - i}

    for base in args.bases:
        if base == "full":
            idx = torch.arange(s, device=x.device)
            mode = "orig"
        else:
            kind, B, mode = base.split("-")
            idx = select_positions(score, s, int(B), kind, 4, args.window)
        ctx, cpos, ap = context_for(feats, idx, None, s, mode)
        L1 = draft_block(draft, target, ctx, cpos, x[s], ap, bs).float()
        conf, y = torch.softmax(L1, -1).max(-1)
        y, conf = y.tolist(), conf.tolist()
        res = {"block1": [y]}

        if copy is not None:
            cidx = select_indices(layer_scores, s, 4, 128, args.copy_budget)
            pos = torch.arange(s, s + bs, device=x.device)[None]
            z = torch.cat([x[: s + 1], torch.tensor(y, device=x.device)])
            po = copy(z[None, s:], past_key_values=sparse_cache(tcache, cidx, s), position_ids=pos, use_cache=True,
                      output_hidden_states=True, logits_to_keep=bs)
            pz = extract_context_feature(po.hidden_states, draft.target_layer_ids).to(feats.dtype)
            cn = po.logits[0].argmax(-1).tolist()
            del po
            cchains = [y]
            dis = [k for k in range(D) if cn[k] != y[k]]
            if dis:
                j1 = dis[0]
                c2, p2, a2 = context_for(feats, idx, pz[:, : j1 + 1], s, mode)
                cchains.append(y[:j1] + [cn[j1]] + draft_block(draft, target, c2, p2, cn[j1], a2, bs).argmax(-1).tolist())
            post, run = {}, 1.0
            for k in range(1, D + 1):
                run *= conf[k - 1]
                post[k] = run * (1.0 - conf[k]) if k < D else run
            for k in sorted(post, key=lambda k: -post[k])[:2]:
                c2, p2, a2 = context_for(feats, idx, pz[:, :k], s, mode)
                cchains.append(y[:k] + draft_block(draft, target, c2, p2, y[k - 1], a2, bs).argmax(-1).tolist())
            res["copy"] = cchains

        for m, i in matches.items():
            tag = f"{base}|{m}"
            if i is None:
                out[tag] = {k: (matched(c[0], truth) + 1 if k == "block1" else max(matched(cc, truth) for cc in c) + 1,
                                tree_nodes(c)) for k, c in res.items()}
                for k in ("lookup", "aligned", "oracle"):
                    out[tag][k] = out[tag]["block1"]
                if "copy" in res:
                    out[tag]["aligned+copy"] = out[tag]["copy"]
                out[tag]["_hit"] = 0
                continue
            r = xl[i + 1 : i + 1 + D]
            al, orc = [y, r], [y, r]
            for Lr in args.restart_at:
                look = feats[:, i : i + Lr]
                real = feats[:, s : s + Lr]
                for src, dst in ((look, al), (real, orc)):
                    c2, p2, a2 = context_for(feats, idx, src, s, mode)
                    dst.append(r[:Lr] + draft_block(draft, target, c2, p2, r[Lr - 1], a2, bs).argmax(-1).tolist())
            chains = dict(res)
            chains["lookup"] = [y, r]
            chains["aligned"] = al
            chains["oracle"] = orc
            if "copy" in res:
                chains["aligned+copy"] = al + res["copy"][1:]
            out[tag] = {k: (max(matched(cc, truth) for cc in c) + 1, tree_nodes(c)) for k, c in chains.items()}
            out[tag]["_hit"] = 1
    return out, diag


@torch.inference_mode()
def main():
    from hspec.data import load_prompts, stop_ids
    from hspec.models import load_draft, load_mid, load_target, load_tokenizer
    from hspec.pipeline import two_stage_generate

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B", help="'' to skip the copy refresh reference")
    ap.add_argument("--ctx", nargs="+", type=int, default=[8192, 16384])
    ap.add_argument("--settings", nargs="+", default=["recall", "history"])
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--pool-new", type=int, default=320)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--anchors", type=int, default=16)
    ap.add_argument("--bases", nargs="+", default=["recent-1024-orig", "attn-1024-compact"])
    ap.add_argument("--match", nargs="+", default=["ngram", "attn"])
    ap.add_argument("--restart-at", nargs="+", type=int, default=[8, 15])
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--probe", type=int, default=16)
    ap.add_argument("--copy-budget", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--drafter-ms", type=float, default=2.6)
    ap.add_argument("--restart-ms", type=float, default=1.0)
    ap.add_argument("--copy-ms", type=float, default=3.0)
    ap.add_argument("--kv-ms-per-ktoken", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp62_aligned")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    copy = load_mid(args.copy) if args.copy else None
    stops = stop_ids(target, tok)
    bs = draft.block_size
    kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}

    pools, evalq = {}, {}
    for d in args.datasets:
        qs = load_prompts(d, args.pool + args.n)
        pool = []
        for q in qs[: args.pool]:
            ids = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                              add_generation_prompt=True, **kw),
                      return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
            r = two_stage_generate(draft, target, target, target, ids, args.pool_new, stops)
            pool.append((q, tok.decode(r.generated, skip_special_tokens=True)))
        pools[d], evalq[d] = pool, qs[args.pool :]
        print(f"[{d}] history pool: {len(pool)} answers", flush=True)

    records = []                     # one per anchor
    rows = []
    for ctx in sorted(args.ctx):
        for setting in args.settings:
            for d in args.datasets:
                for pi, q in enumerate(evalq[d]):
                    if setting == "history":
                        text, nturns = build_history(tok, pools[d], q, ctx, kw)
                        k = None
                    else:
                        text, nturns, k = build_recall(tok, pools[d], ctx, kw, rng)
                    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
                    n0 = ids.shape[1]
                    feats = tcache = None
                    try:
                        x = greedy_long(target, ids[0], args.max_new, stops, args.chunk)
                        cr = None
                        if k is not None:
                            ref = tok(pools[d][k][1], add_special_tokens=False)["input_ids"]
                            cr = copy_rate(x[n0:].tolist(), ref)
                        feats, tcache = prefill_features(target, x, draft.target_layer_ids, args.chunk)
                        x_np = x.cpu().numpy()
                        anchors = list(range(n0, len(x) - 3 * bs, args.stride))[: args.anchors]
                        for s in anchors:
                            res, diag = eval_anchor(draft, target, copy, feats, tcache, x, x_np, s, bs, args)
                            records.append({"setting": setting, "ctx": ctx, "dataset": d, "prompt": pi,
                                            "copy_rate": cr, "s": s, "res": res, "diag": diag})
                        print(f"[{setting} {d} ctx={ctx}] prompt {n0} tokens, {len(anchors)} anchors"
                              + (f", copy rate {cr:.2f}" if cr is not None else "")
                              + f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                    except torch.OutOfMemoryError:
                        print(f"[{setting} {d} ctx={ctx}] prompt {n0} tokens: OUT OF MEMORY, skipped", flush=True)
                    del feats, tcache
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            block = summarize([r for r in records if r["setting"] == setting and r["ctx"] == ctx], args, ctx)
            rows += block
            if block:
                print_table(block, list(block[0]), f"{setting}, ctx {ctx}: aligned looked-up features (tau = accepted "
                                                   f"per verify; x = projected speed vs block1 of the same base)")
            hi = [r for r in records if r["setting"] == setting and r["ctx"] == ctx and (r["copy_rate"] or 0) >= 0.5]
            if setting == "recall" and hi:
                b2 = summarize(hi, args, ctx)
                print_table(b2, list(b2[0]), f"{setting}, ctx {ctx}, prompts with copy rate >= 0.5 only")
            save_json(args.out, {"args": vars(args), "rows": rows, "records": records})


def summarize(recs, args, ctx):
    if not recs:
        return []
    rows = []
    kv = args.kv_ms_per_ktoken * ctx / 1000
    for tag in recs[0]["res"]:
        base, m = tag.split("|")
        vals = defaultdict(list)
        for r in recs:
            for var, val in r["res"][tag].items():
                vals[var].append(val)
        hit = mean(vals.pop("_hit"))
        b1 = mean(t for t, _ in vals["block1"])
        b1_ms = args.drafter_ms + v(16) + kv
        diag = [r["diag"][m] for r in recs if r["diag"].get(m)]
        for var, vs in vals.items():
            tau = mean(t for t, _ in vs)
            nodes = mean(n for _, n in vs)
            extra = {"block1": 0.0, "lookup": 0.0, "aligned": args.restart_ms, "oracle": args.restart_ms,
                     "copy": args.copy_ms + args.restart_ms, "aligned+copy": args.copy_ms + args.restart_ms}[var]
            ms = args.drafter_ms + extra + v(nodes + 1) + kv
            rows.append({"ctx": ctx, "base": base, "match": m, "variant": var, "tau": tau, "nodes": nodes,
                         "x": (tau / ms) / (b1 / b1_ms), "hit": hit,
                         "r_match": mean(d_["r_match"] for d_ in diag), "cos": mean(d_["cos"] for d_ in diag),
                         "anchors": len(vs)})
    return rows


if __name__ == "__main__":
    main()
