"""Experiment 61: can a sparse feature context undo DFlash's long-context decay without training? (offline, T=0)

exp60: DFlash's accepted length falls from 8.8 at 2K context to 2.4 at 32K (public Qwen3-8B drafter, trained at
~3K), and the copy refresh only multiplies it by a constant ~1.9x. A copy that reads 1% of the cache gives features
as good as the full one, so the features carry what the drafter needs; the loss should come from the drafter's own
attention over a context ~10x longer than any it was trained on. Here the drafter sees only a selected set of
context positions (hspec/sparsedraft.py):
  full                       all positions (plain DFlash)
  recent-B-{orig,compact}    4 sinks + the last B-4 positions      (truncation; `compact` = exactly DFlash on a
                                                                   B-token context)
  attn-B-{orig,compact}      4 sinks + last 512 + the top positions by the target's attention mass from the 16
                             queries before the anchor
  orig keeps the true positions, compact renumbers the context 0 .. S-1 so it falls inside the trained range.

Two settings, both built from the target's own answers to earlier questions of the same dataset (multi-turn chat):
  history   the long context is unrelated earlier turns, then a new question (exp60's setting; easy for sparsity)
  recall    the final turn asks the model to repeat, word for word, an answer it gave in the first third of the
            history, so the output depends on text 10K-30K tokens back (hard for truncation, the stress test)
            `copy_rate` = share of generated 8-grams that occur in the recalled answer: how much the output really
            depends on the distant text.

Per anchor and drafter variant: block1 tau (DFlash chain), and the refresh with a KV-sparse 4-bit copy (exp60
sparse1024 view, which exp60 found as good as the full cache) and pc + c1 + c2 restarts drafted with the SAME
sparse drafter context: does the sparse drafter compose with the method? `ctx_read` = share of the context the
drafter attends to.

  python scripts/exp61_sparsedraft.py --ctx 8192 16384 32768 --settings history recall
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp60_sparsekv import build_history, greedy_long  # noqa: E402
from hspec.sparsedraft import context_for, draft_block, score_positions, select_positions  # noqa: E402
from hspec.sparsekv import attention_scores, prefill_features, select_indices, sparse_cache  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


def variants(budgets):
    out = [("full", None, None, "orig")]
    for B in budgets:
        for kind in ("recent", "attn"):
            for mode in ("orig", "compact"):
                out.append((f"{kind}-{B}-{mode}", kind, B, mode))
    return out


@torch.inference_mode()
def eval_anchor(draft, target, copy, feats, tcache, x, s, bs, vars_, args):
    """dict variant -> (block1 tau, refresh tau, ctx_read)."""
    from dflash.model import extract_context_feature

    D = bs - 1
    truth = x[s + 1 : s + 2 * bs].tolist()
    layer_scores = attention_scores(target, tcache, x, s, args.probe)
    score = score_positions(layer_scores)
    cidx = select_indices(layer_scores, s, args.copy_sinks, args.copy_window, args.copy_budget)
    pos = torch.arange(s, s + bs, device=x.device)[None]
    out = {}
    for name, kind, B, mode in vars_:
        idx = torch.arange(s, device=x.device) if kind is None else select_positions(score, s, B, kind, 4, args.window)
        ctx, cpos, ap = context_for(feats, idx, None, s, mode)
        L1 = draft_block(draft, target, ctx, cpos, x[s], ap, bs).float()
        conf, y = torch.softmax(L1, -1).max(-1)
        y, conf = y.tolist(), conf.tolist()
        chains = [y]
        if not args.no_refresh:
            z = torch.cat([x[: s + 1], torch.tensor(y, device=x.device)])
            po = copy(z[None, s:], past_key_values=sparse_cache(tcache, cidx, s), position_ids=pos, use_cache=True,
                      output_hidden_states=True, logits_to_keep=bs)
            pz = extract_context_feature(po.hidden_states, draft.target_layer_ids).to(feats.dtype)    # [1, bs, W]
            cn = po.logits[0].argmax(-1).tolist()
            del po
            dis = [k for k in range(D) if cn[k] != y[k]]
            if dis:
                j1 = dis[0]
                c2, p2, a2 = context_for(feats, idx, pz[:, : j1 + 1], s, mode)
                Lr = draft_block(draft, target, c2, p2, cn[j1], a2, bs)
                chains.append(y[:j1] + [cn[j1]] + Lr.argmax(-1).tolist())
            post, run = {}, 1.0
            for i in range(1, D + 1):
                run *= conf[i - 1]
                post[i] = run * (1.0 - conf[i]) if i < D else run
            for i in sorted(post, key=lambda i: -post[i])[:2]:
                c2, p2, a2 = context_for(feats, idx, pz[:, :i], s, mode)
                Lr = draft_block(draft, target, c2, p2, y[i - 1], a2, bs)
                chains.append(y[:i] + Lr.argmax(-1).tolist())
        out[name] = (matched(y, truth) + 1, max(matched(c, truth) for c in chains) + 1, len(idx) / max(s, 1))
    return out


def build_recall(tok, pool, ctx, kw, rng):
    """History of (q, a) pairs under ctx tokens, then 'repeat your answer to <q_k> word for word' for a k in the
    first third of the history. Returns (text, turns, k)."""
    msgs = []
    longest_q = max((q for q, _ in pool), key=len)
    tail = 'Repeat your earlier answer to the question "' + longest_q + '" word for word.'
    for q, a in pool:
        trial = msgs + [{"role": "user", "content": q}, {"role": "assistant", "content": a},
                        {"role": "user", "content": tail}]
        text = tok.apply_chat_template(trial, tokenize=False, add_generation_prompt=True, **kw)
        if len(tok(text, add_special_tokens=False)["input_ids"]) > ctx:
            break
        msgs = trial[:-1]
    n = len(msgs) // 2
    k = rng.randrange(max(1, n // 3))
    ask = 'Repeat your earlier answer to the question "' + pool[k][0] + '" word for word.'
    text = tok.apply_chat_template(msgs + [{"role": "user", "content": ask}], tokenize=False,
                                   add_generation_prompt=True, **kw)
    return text, n, k


def copy_rate(gen_ids, ref_ids, n=8):
    ref = {tuple(ref_ids[i : i + n]) for i in range(len(ref_ids) - n + 1)}
    grams = [tuple(gen_ids[i : i + n]) for i in range(len(gen_ids) - n + 1)]
    return sum(g in ref for g in grams) / len(grams) if grams else float("nan")


@torch.inference_mode()
def main():
    from hspec.data import load_prompts, stop_ids
    from hspec.models import load_draft, load_mid, load_target, load_tokenizer
    from hspec.pipeline import two_stage_generate

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--ctx", nargs="+", type=int, default=[8192, 16384, 32768])
    ap.add_argument("--settings", nargs="+", default=["history", "recall"])
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500"])
    ap.add_argument("--n", type=int, default=6, help="evaluation prompts per dataset, setting and context length")
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--pool-new", type=int, default=320)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--anchors", type=int, default=16)
    ap.add_argument("--budgets", nargs="+", type=int, default=[1024, 3072], help="drafter context positions")
    ap.add_argument("--window", type=int, default=512, help="recent window inside the attn selection")
    ap.add_argument("--probe", type=int, default=16)
    ap.add_argument("--copy-sinks", type=int, default=4)
    ap.add_argument("--copy-window", type=int, default=128)
    ap.add_argument("--copy-budget", type=int, default=1024)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp61_sparsedraft")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    copy = None if args.no_refresh else load_mid(args.copy)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
    vars_ = variants(args.budgets)

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

    rec = defaultdict(list)          # (setting, ctx, variant) -> [(b1, refresh, ctx_read)]
    crate = defaultdict(list)        # (setting, ctx) -> copy rates (recall only)
    rows = []
    for ctx in sorted(args.ctx):
        for setting in args.settings:
            oom = 0
            for d in args.datasets:
                for qi, q in enumerate(evalq[d]):
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
                        if k is not None:
                            ref = tok(pools[d][k][1], add_special_tokens=False)["input_ids"]
                            crate[(setting, ctx)].append(copy_rate(x[n0:].tolist(), ref))
                        feats, tcache = prefill_features(target, x, draft.target_layer_ids, args.chunk)
                        anchors = list(range(n0, len(x) - 2 * bs - 1, args.stride))[: args.anchors]
                        got = defaultdict(list)
                        for s in anchors:
                            for name, val in eval_anchor(draft, target, copy, feats, tcache, x, s, bs, vars_,
                                                         args).items():
                                got[name].append(val)
                        for name, vs in got.items():
                            rec[(setting, ctx, name)] += vs
                        extra = f", copy rate {crate[(setting, ctx)][-1]:.2f}" if k is not None else ""
                        print(f"[{setting} {d} ctx={ctx}] prompt {n0} tokens ({nturns} turns), {len(anchors)} "
                              f"anchors, {len(x) - n0} generated{extra}, peak "
                              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                    except torch.OutOfMemoryError:
                        oom += 1
                        print(f"[{setting} {d} ctx={ctx}] prompt {n0} tokens: OUT OF MEMORY, skipped", flush=True)
                    del feats, tcache
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            full = rec.get((setting, ctx, "full"), [])
            if not full:
                continue
            fb1, fr = mean(v[0] for v in full), mean(v[1] for v in full)
            block = []
            for name, *_ in vars_:
                vs = rec[(setting, ctx, name)]
                b1, rf = mean(v[0] for v in vs), mean(v[1] for v in vs)
                block.append({"setting": setting, "ctx": ctx, "drafter_ctx": name, "block1": b1,
                              "b1_vs_full": b1 / fb1, "refresh": rf, "ref_vs_full": rf / fr,
                              "ctx_read": mean(v[2] for v in vs), "anchors": len(vs)})
            rows += block
            cr = crate.get((setting, ctx))
            print_table(block, list(block[0]),
                        f"{setting}, ctx {ctx}: drafter over a selected context (copy: 4-bit over "
                        f"{args.copy_budget}+{args.copy_window}+{args.copy_sinks} cache positions; {oom} skipped"
                        + (f"; mean copy rate {mean(cr):.2f}" if cr else "") + ")")
            save_json(args.out, {"args": vars(args), "rows": rows,
                                 "copy_rate": {f"{a}@{b}": mean(v) for (a, b), v in crate.items()}})


if __name__ == "__main__":
    main()
