"""Experiment 60: a KV-sparse copy as the feature source at long context (offline, T=0).

The copies so far are cheap because they read fewer weight bytes. At long context the verify pass is dominated by
the KV cache instead (Qwen3-8B: ~147 KB/token, 4.7 GB at 32K vs 16 GB bf16 / 5 GB 4-bit weights), and our copy
attends to the target's FULL cache, so its relative cost roughly doubles there. The copy of this experiment is made
cheap by sparse attention over the shared cache (sinks + recent window + top-k positions by the target's own
attention mass, hspec/sparsekv.py), with full weights (the target itself) or 4-bit weights. MagicDec showed this
crossover for self-speculation; here the question is whether the DRAFTER can live with features computed over a
sparse context, i.e. whether what the drafter needs from the drafted tokens is mostly local.

Long contexts are built as multi-turn chat history: earlier (question, target answer) pairs from the same dataset
are prepended until the context reaches --ctx tokens, then the evaluation question follows. The trajectory is the
target's greedy answer (via DFlash, lossless at T=0). Per anchor s (stride --stride, at most --anchors per prompt):
  block1       DFlash block from real features (chain y)
  for every copy in --copies and every KV view (full, or sparse with a budget in --budgets):
    one copy pass over [x_s, y] on that view of the target's cache -> features for s .. s+15 and next tokens
    restarts pc (copy's first disagreement) + c1, c2 (drafter's likely stops), exp56's pc+c2 policy
    tau = longest chain prefix matching the trajectory + 1; nodes; copy-target agreement; feature cosine vs the
    target's real features on the same drafted tokens (per drafter layer, averaged)
Reported per context length, copy and view, plus a projected round time from a simple bytes model:
  pass_ms = weight_ms + kv_ms_per_ktoken * (positions read / 1000), target 16 GB bf16, copy 5 GB (4-bit) or 16 GB.

  python scripts/exp60_sparsekv.py --ctx 4096 16384 --copies target bnb4:Qwen/Qwen3-8B --budgets 256 1024
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hspec.sparsekv import attention_scores, prefill_features, prefix_cache, select_indices, sparse_cache  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

VTAB = [(1, 12.7), (16, 13.3), (32, 14.1), (64, 14.2), (128, 15.8), (256, 19.0)]


def v(q):
    for (a, ta), (b, tb) in zip(VTAB, VTAB[1:]):
        if q <= b:
            return ta + (tb - ta) * (q - a) / (b - a)
    return VTAB[-1][1]


def build_history(tok, pool, question, ctx, enable_thinking_kw):
    """Chat text with as many (q, a) pairs from pool as fit under ctx tokens, then `question`."""
    msgs = []
    for q, a in pool:
        trial = msgs + [{"role": "user", "content": q}, {"role": "assistant", "content": a},
                        {"role": "user", "content": question}]
        text = tok.apply_chat_template(trial, tokenize=False, add_generation_prompt=True, **enable_thinking_kw)
        if len(tok(text, add_special_tokens=False)["input_ids"]) > ctx:
            break
        msgs = trial[:-1]
    text = tok.apply_chat_template(msgs + [{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True, **enable_thinking_kw)
    return text, len(msgs) // 2


@torch.inference_mode()
def eval_anchor(draft, target, copies, feats, tcache, x, s, bs, views, probe):
    """One anchor: dict (copy name, view name) -> (tau, nodes, agree, cosine, positions read)."""
    from dflash.model import extract_context_feature

    from exp27_refresh import blocks_logits, matched, tree_nodes
    from hspec.lagtrain import draft_logits_lagged

    D = bs - 1
    truth = x[s + 1 : s + 2 * bs].tolist()
    pr = torch.softmax(draft_logits_lagged(draft, target, feats, x, s, 0, bs).float(), -1)
    conf, y = pr.max(-1)
    y, conf = y.tolist(), conf.tolist()
    z = torch.cat([x[: s + 1], torch.tensor(y, device=x.device)])
    pos = torch.arange(s, s + bs, device=x.device)[None]
    # the target's own features / next tokens on the drafted tokens (reference for cosine and agreement)
    tr = target(z[None, s:], past_key_values=prefix_cache(tcache, s), position_ids=pos, use_cache=True,
                output_hidden_states=True, logits_to_keep=bs)
    real = extract_context_feature(tr.hidden_states, draft.target_layer_ids)[0].to(feats.dtype)   # [bs, W]
    tnext = tr.logits[0].argmax(-1)
    del tr
    post, run = {}, 1.0
    for i in range(1, D + 1):
        run *= conf[i - 1]
        post[i] = run * (1.0 - conf[i]) if i < D else run
    cpos = sorted(post, key=lambda i: -post[i])[:2]
    W = real.shape[-1] // len(draft.target_layer_ids)

    scores = None
    out = {("block1", "-"): (matched(y, truth) + 1, D, float("nan"), float("nan"), 0)}
    for cname, copy in copies.items():
        for vname, spec in views.items():
            if spec is None:
                cache, nread = prefix_cache(tcache, s), s
            else:
                if scores is None:
                    scores = attention_scores(target, tcache, x, s, probe)
                idx = select_indices(scores, s, *spec)
                cache, nread = sparse_cache(tcache, idx, s), int(idx[0].shape[1])
            po = copy(z[None, s:], past_key_values=cache, position_ids=pos, use_cache=True,
                      output_hidden_states=True, logits_to_keep=bs)
            pz = extract_context_feature(po.hidden_states, draft.target_layer_ids)[0].to(feats.dtype)
            cnext = po.logits[0].argmax(-1)
            del po, cache
            agree = float((cnext == tnext).float().mean())
            cos = float(torch.nn.functional.cosine_similarity(
                pz.float().view(bs, -1, W), real.float().view(bs, -1, W), dim=-1).mean())
            full = torch.cat([feats[:, :s], pz[None]], 1)
            cn = cnext.tolist()
            dis = [k for k in range(D) if cn[k] != y[k]]
            chains = [y]
            if dis:
                j1 = dis[0]
                zc = torch.cat([z[: s + j1 + 1], torch.tensor([cn[j1]], device=z.device)])
                L = blocks_logits(draft, target, full, zc, [(s + j1 + 1, 0)], bs)
                chains.append(y[:j1] + [cn[j1]] + L[0].argmax(-1).tolist())
            mix = torch.cat([feats[:, :s], pz[None, :D]], 1)
            Lc = blocks_logits(draft, target, mix, z, [(s + i, 0) for i in cpos], bs)
            for r, i in enumerate(cpos):
                chains.append(y[:i] + Lc[r].argmax(-1).tolist())
            out[(cname, vname)] = (max(matched(c, truth) for c in chains) + 1, tree_nodes(chains), agree, cos, nread)
    return out


@torch.inference_mode()
def main():
    from dflash.model import extract_context_feature  # noqa: F401  (import check)

    from hspec.data import load_prompts, stop_ids
    from hspec.models import load_draft, load_mid, load_target, load_tokenizer
    from hspec.pipeline import two_stage_generate

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copies", nargs="+", default=["target", "bnb4:Qwen/Qwen3-8B"],
                    help="'target' = the target's own weights; else a load_mid spec with the target's architecture")
    ap.add_argument("--ctx", nargs="+", type=int, default=[2048, 8192, 16384, 32768])
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500"])
    ap.add_argument("--n", type=int, default=6, help="evaluation prompts per dataset and context length")
    ap.add_argument("--pool", type=int, default=120, help="(question, answer) pairs generated for the history")
    ap.add_argument("--pool-new", type=int, default=320, help="max answer tokens in the history")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--anchors", type=int, default=16, help="anchors per prompt")
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--budgets", nargs="+", type=int, default=[256, 1024, 4096], help="top-k positions per head")
    ap.add_argument("--probe", type=int, default=16, help="query positions before the anchor that score the prefix")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--drafter-ms", type=float, default=2.6)
    ap.add_argument("--restart-ms", type=float, default=1.0)
    ap.add_argument("--kv-ms-per-ktoken", type=float, default=0.1, help="147 KB/token at ~1.5 TB/s")
    ap.add_argument("--copy-weight-ms", type=float, default=3.0, help="4-bit copy, vLLM Marlin")
    ap.add_argument("--target-weight-ms", type=float, default=12.7)
    ap.add_argument("--out", default="exp60_sparsekv")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    copies = {}
    for c in args.copies:
        copies[c] = target if c == "target" else load_mid(c)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
    views = {"full": None}
    for b in args.budgets:
        views[f"sparse{b}"] = (args.sinks, args.window, b)

    rec = defaultdict(list)
    for d in args.datasets:
        qs = load_prompts(d, args.pool + args.n)
        pool = []
        for q in qs[: args.pool]:
            ids = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                              add_generation_prompt=True, **kw),
                      return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
            r = two_stage_generate(draft, target, target, target, ids, args.pool_new, stops)
            pool.append((q, tok.decode(r.generated, skip_special_tokens=True)))
        print(f"[{d}] history pool: {len(pool)} answers", flush=True)
        for ctx in args.ctx:
            for q in qs[args.pool :]:
                text, nturns = build_history(tok, pool, q, ctx, kw)
                ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
                n0 = ids.shape[1]
                x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
                feats, tcache = prefill_features(target, x, draft.target_layer_ids, args.chunk)
                anchors = list(range(n0, len(x) - 2 * bs - 1, args.stride))[: args.anchors]
                for s in anchors:
                    for key, val in eval_anchor(draft, target, copies, feats, tcache, x, s, bs, views, args.probe).items():
                        rec[(ctx, d) + key].append(val)
                del feats, tcache
                torch.cuda.empty_cache()
                print(f"[{d} ctx={ctx}] prompt {n0} tokens ({nturns} history turns), {len(anchors)} anchors, "
                      f"{len(x) - n0} generated", flush=True)

    rows = []
    for ctx in args.ctx:
        keys = sorted({k[2:] for k in rec if k[0] == ctx}, key=lambda k: (k[0] != "block1", k[0], k[1]))
        for cname, vname in keys:
            vals = [v_ for k, vs in rec.items() if k[0] == ctx and k[2:] == (cname, vname) for v_ in vs]
            if not vals:
                continue
            tau, nodes = mean(x_[0] for x_ in vals), mean(x_[1] for x_ in vals)
            nread = mean(x_[4] for x_ in vals)
            kv = args.kv_ms_per_ktoken
            # bytes model: a pass = weights + width overhead (exp29 table, relative to 1 token) + KV read
            base_tau = mean(x_[0] for x_ in rec_all(rec, ctx, ("block1", "-")))
            base_ms = args.drafter_ms + v(16) + kv * ctx / 1000
            if cname == "block1":
                ms = base_ms
            else:
                w = args.target_weight_ms if cname == "target" else args.copy_weight_ms
                copy_ms = w + kv * (nread + bs) / 1000
                verify_ms = v(nodes + 1) + kv * ctx / 1000
                ms = args.drafter_ms + copy_ms + args.restart_ms + verify_ms
            rows.append({"ctx": ctx, "copy": cname, "view": vname, "tau": tau, "nodes": nodes,
                         "agree": mean(x_[2] for x_ in vals), "cos": mean(x_[3] for x_ in vals),
                         "kv_read": nread / max(ctx, 1), "x_vs_dflash": (tau / ms) / (base_tau / base_ms),
                         "anchors": len(vals)})
    print_table(rows, ["ctx", "copy", "view", "tau", "nodes", "agree", "cos", "kv_read", "x_vs_dflash", "anchors"],
                f"KV-sparse copies at long context (sinks {args.sinks}, window {args.window}, probe {args.probe}; "
                f"speed from the bytes model, see header)")
    save_json(args.out, {"args": vars(args), "rows": rows})


def rec_all(rec, ctx, key):
    return [v_ for k, vs in rec.items() if k[0] == ctx and k[2:] == key for v_ in vs]


if __name__ == "__main__":
    main()
