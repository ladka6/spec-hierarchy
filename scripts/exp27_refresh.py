"""Experiment 27: feature-refreshed redrafting, the real algorithm offline.

exp26 measured fixes at the oracle stop with features of the TRUE prefix. Here the round only knows
its own draft:

  1. DFlash drafts block 1 at anchor s (real features up to s): chain y_1 .. y_15.
  2. A feature source runs once over the DRAFTED tokens x_s, y_1 .. y_14 and gives stand-in
     features for positions s .. s+14.
  3. One batched drafter pass re-anchors at s+i (i in a restart set) using those features and the
     drafted prefix y_1 .. y_i, giving a 15-token continuation per restart. i = 15 is a second block
     after block 1 (multi-block drafting).
  4. The verifier checks the union of block 1 and the chosen continuations (a tree). Accepted =
     longest path matching the target trajectory, + 1 bonus.

Feature sources (step 2):
  tokens   none: lag-trained drafter, drafted tokens as plain inputs (D2SD-like)
  <proxy>  a cheap target copy (load_mid spec, e.g. bnb4:Qwen/Qwen3-8B)
  fresh    the bf16 target itself (upper bound; costs a target pass)

Option 1 (pc*): the copy's own next-token predictions find the first likely error j in block 1 and
supply the corrected token (pc1 = its top-1, pc2 = its top-2); the drafter restarts after it.
Ablations: pc_tok (same correction, continuation drafted from tokens only by the lag drafter, no
copy features), pc_none (the correction alone, no continuation: a quantized intermediate verifier).
Option 2 (|gateG): all extra work only on rounds with P(stop within 8 tokens) >= G; "ran" = share of
rounds that run it.

Restart policies: all (i = 1 .. 15), conf-K (top-K positions by the drafter's stop posterior
prod_{k<=i} c_k * (1 - c_{i+1}), c = max prob of block 1; i = 15 uses prod of all c), end (i = 15
only: plain second block), oracle (the true stop; upper bound for one restart).

  python scripts/exp27_refresh.py --lag-draft .../drafter_lag16 --proxies bnb4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import _output_head, extract_context_feature  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged, pack  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def blocks_logits(dm, head_src, feats, ids, blocks, bs):
    """Logits [n, bs-1, V] for packed blocks (anchor, lag) over feats [1, T, W] and ids."""
    noise, pos, mask, starts = pack(dm, head_src, feats, ids, blocks, bs)
    hid = dm(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
             past_key_values=None, use_cache=False)[0]
    return dm.compute_logits(torch.stack([hid[st : st + bs - 1] for st in starts]), _output_head(head_src))


def matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


def tree_nodes(chains):
    nodes = set()
    for c in chains:
        for k in range(1, len(c) + 1):
            nodes.add(tuple(c[:k]))
    return len(nodes)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--lag-draft", default=None)
    ap.add_argument("--proxies", nargs="*", default=["bnb4:Qwen/Qwen3-8B"])
    ap.add_argument("--no-fresh", action="store_true")
    ap.add_argument("--no-tokens", action="store_true", help="skip the tokens-only source (saves the lag drafter)")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--gates", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5],
                    help="option 2: run the refresh only if P(stop within the first 8 tokens) >= gate")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp27_refresh")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    lagd = load_draft(args.lag_draft) if args.lag_draft else draft
    proxies = {spec.split(":", 1)[0]: load_mid(spec) for spec in args.proxies}
    stops = stop_ids(target, tok)
    bs = draft.block_size
    D = bs - 1                                       # drafted tokens per block
    sources = ([] if args.no_tokens else ["tokens"]) + list(proxies) + ([] if args.no_fresh else ["fresh"])
    policies = ["all", "end", "oracle"] + [f"conf{k}" for k in args.ks]

    rec = defaultdict(list)                          # (source, policy) -> [(tau, nodes, beyond_block)]
    base = []
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            if args.temp > 0:
                x = target.generate(ids, max_new_tokens=args.max_new, do_sample=True, temperature=args.temp,
                                    top_p=1.0, top_k=0, eos_token_id=stops)[0]
            else:
                x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            for s in range(n0, len(x) - 2 * bs - 1, args.stride):
                truth = x[s + 1 : s + 2 * bs].tolist()       # 31 tokens: enough for restarts at s+15
                lg = draft_logits_lagged(draft, target, feats, x, s, 0, bs)
                pr = torch.softmax(lg.float(), -1)
                conf, y = pr.max(-1)
                y, conf = y.tolist(), conf.tolist()
                m0 = matched(y, truth)
                base.append(m0 + 1)
                z = torch.cat([x[: s + 1], torch.tensor(y, device=x.device)])     # drafted sequence, len s+16
                # stop posterior per restart length i (keep y_1..y_i, redraft from s+i)
                post, run = {}, 1.0
                for i in range(1, D + 1):
                    run *= conf[i - 1]
                    post[i] = run * (1.0 - conf[i]) if i < D else run
                restarts = {"all": list(range(1, D + 1)), "end": [D],
                            "oracle": [min(m0, D)] if m0 >= 1 else []}
                for k in args.ks:
                    restarts[f"conf{k}"] = sorted(sorted(post, key=lambda i: -post[i])[:k])
                need = sorted({i for v in restarts.values() for i in v})
                if not need:
                    need = [1]
                for src in sources:
                    if src == "tokens":
                        L = blocks_logits(lagd, target, feats[:, :s], z, [(s + i, i) for i in need], bs)
                    else:
                        fm = target if src == "fresh" else proxies[src]
                        po = fm(z[None], output_hidden_states=True, logits_to_keep=bs)
                        pz = extract_context_feature(po.hidden_states, draft.target_layer_ids).to(feats.dtype)
                        top2 = po.logits[0].float().topk(2, -1).indices.tolist()   # [bs][2]: predictions for s+1 .. s+16
                        del po
                        mix = torch.cat([feats[:, :s], pz[:, s : s + D]], 1)
                        L = blocks_logits(draft, target, mix, z, [(s + i, 0) for i in need], bs)
                    cont = {i: y[:i] + L[j].argmax(-1).tolist() for j, i in enumerate(need)}
                    extra = {}
                    if src != "tokens":
                        # option 1: the copy's own predictions locate the first likely error j and supply the
                        # corrected token; restart there (anchor = corrected token, copy features before it)
                        j = next((k for k in range(D) if top2[k][0] != y[k]), D)        # 0-based; D = no error
                        full = torch.cat([feats[:, :s], pz[:, s : s + bs]], 1)           # features through s+15
                        for name, alt in (("pc1", 0), ("pc2", 1)):
                            tokj = top2[j][alt]
                            zc = torch.cat([z[: s + j + 1], torch.tensor([tokj], device=z.device)])
                            Lc = blocks_logits(draft, target, full, zc, [(s + j + 1, 0)], bs)
                            extra[name] = y[:j] + [tokj] + Lc[0].argmax(-1).tolist()
                            if alt == 0:
                                # ablations: same correction, continuation from tokens only (lag drafter,
                                # no copy features), and the correction alone (no continuation)
                                Lt = blocks_logits(lagd, target, feats[:, :s], zc, [(s + j + 1, j + 1)], bs)
                                extra["pc_tok"] = y[:j] + [tokj] + Lt[0].argmax(-1).tolist()
                                extra["pc_none"] = y[:j] + [tokj]
                    pols = {pol: [cont[i] for i in rs] for pol, rs in restarts.items()}
                    if extra:
                        pols["pc"] = [extra["pc1"]]
                        pols["pc_fork"] = [extra["pc1"], extra["pc2"]]
                        pols["pc_tok"] = [extra["pc_tok"]]
                        pols["pc_none"] = [extra["pc_none"]]
                        for k in args.ks:
                            pols[f"pc+conf{k}"] = [extra["pc1"]] + [cont[i] for i in restarts[f"conf{k}"]]
                    # option 2: skip all extra work when block 1 looks safe
                    p8 = 1.0
                    for c in conf[:8]:
                        p8 *= c
                    gated = {}
                    for g in args.gates:
                        run = (1.0 - p8) >= g
                        for pol in ("conf2", "pc+conf2", "pc"):
                            if pol in pols:
                                gated[f"{pol}|gate{g}"] = (pols[pol] if run else [], run)
                    for pol, ch in list(pols.items()):
                        gated[pol] = (ch, True)
                    for pol, (ch, ran) in gated.items():
                        chains = [y] + ch
                        best = max(matched(c, truth) for c in chains)
                        rec[(src, pol)].append((best + 1, tree_nodes(chains), float(best > D), float(ran)))
            print(f"[{d}] {len(base)} anchors", flush=True)

    rows = [{"source": "-", "policy": "block1 only", "tau": mean(base), "nodes": D, "beyond_block1": 0.0,
             "ran": 0.0}]
    for (src, pol), v in sorted(rec.items()):
        rows.append({"source": src, "policy": pol, "tau": mean(r[0] for r in v), "nodes": mean(r[1] for r in v),
                     "beyond_block1": mean(r[2] for r in v), "ran": mean(r[3] for r in v)})
    print_table(rows, ["source", "policy", "tau", "nodes", "beyond_block1", "ran"],
                f"refreshed redrafting (T={args.temp}): accepted per verify pass, tree nodes, share of rounds "
                "accepting past block 1")
    save_json(args.out, {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
