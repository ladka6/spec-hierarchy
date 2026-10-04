"""Experiment 26: where does a block drafter lose acceptance? (offline diagnostic)

All our dependency fixes (block trees, re-drafting at the uncertain spot, dependency head) land at
+5-8% over DDTree, and DDTree itself barely grows from 64 to 512 nodes (7.71 -> 8.10). This asks why,
by decomposing every stop of the DFlash chain / DDTree on target trajectories:

  cap       the whole block was accepted (block length is the limit)
  coverage  the target's token at the stop position is outside the drafter's top-K marginal there
            (no tree or dependency fix built on these marginals can reach it)
  path      the token is inside the top-K, but the tree did not include that path

and, for every stop, what would have fixed it:

  tokens    redraft conditioned on the (correct) prefix tokens, no new target features
            (= what in-block dependency fixes can give: D2SD, exp18, dependency heads)
  fresh     redraft with real target features up to the stop (upper bound: more target information)

  depth     redraft with features from only the target's first --depth-exit layers on the prefix
            (how much target computation buys back the gap between tokens and fresh)

  faker     redraft with context vectors emulated by a trained faker (hspec/faker.py)
  <proxy>   redraft with features from a cheap copy of the target (e.g. 4-bit) on the prefix

If "tokens" fixes few stops and "fresh" many, the drafter is limited by missing target information,
not by in-block independence. Also reports, per distance after a position, how much conditioning on
the true token there helps, binned by the target's entropy at that position (branch-point hypothesis:
the gain should concentrate after high-entropy positions).

Trajectories: target greedy (default) or sampled at --temp T (target's own samples).

  python scripts/exp26_lossdiag.py --lag-draft /scratch-shared/$USER/hspec_lagft/drafter_lag16
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

from dflash.model import _output_head, extract_context_feature  # noqa: E402
from exp15_blocktree import accepted, tree_paths  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.faker import block_logits_ctx, fake_context, load_faker  # noqa: E402
from hspec.feathead import context_vectors  # noqa: E402
from hspec.lagtrain import deep_columns, draft_logits_depth, draft_logits_lagged, pack  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.tree import build_ddtree  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

TOPK = 16                                   # DDTree candidates per depth
EBINS = [0.1, 0.5, 1.5]                     # target entropy bins (nats): <0.1, <0.5, <1.5, >=1.5


def ebin(h: float) -> str:
    for i, b in enumerate(EBINS):
        if h < b:
            return f"<{b}" if i == 0 else f"{EBINS[i - 1]}-{b}"
    return f">={EBINS[-1]}"


def ranks(logits: torch.Tensor, truth: torch.Tensor):
    """Rank (1 = top) and log-prob of truth[r] under logits[r]."""
    lp = torch.log_softmax(logits.float(), -1)
    t = lp.gather(-1, truth[:, None])[:, 0]
    return ((lp > t[:, None]).sum(-1) + 1).cpu().tolist(), t.cpu().tolist()


def cond_logits(dm, head_src, feats, x, s, bs, mode):
    """For i = 1 .. bs-2: logits of the block re-anchored at s+i (true prefix x[s+1..s+i]).
    mode tokens: features only up to s, prefix tokens as inputs (lag i).
    mode fresh: real features up to s+i (lag 0). Returns list of [bs-1, V] (index i-1)."""
    if mode == "tokens":
        blocks, f = [(s + i, i) for i in range(1, bs - 1)], feats[:, :s]
    else:
        blocks, f = [(s + i, 0) for i in range(1, bs - 1)], feats[:, : s + bs - 1]
    noise, pos, mask, starts = pack(dm, head_src, f, x, blocks, bs)
    hid = dm(target_hidden=f, noise_embedding=noise, position_ids=pos, attention_mask=mask,
             past_key_values=None, use_cache=False)[0]
    rows = torch.stack([hid[st : st + bs - 1] for st in starts])
    return dm.compute_logits(rows, _output_head(head_src))


def depth_logits(dm, head_src, feats, x, s, bs, deep):
    """Like cond_logits, mode depth: full features up to s, then features from only the target's
    lower layers (deep slices zeroed) for the prefix positions s .. s+i-1."""
    return torch.stack([draft_logits_depth(dm, head_src, feats, x, s + i, i, bs, deep) for i in range(1, bs - 1)])


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--lag-draft", default=None, help="lag-trained drafter for the tokens mode (else --draft)")
    ap.add_argument("--depth-draft", default=None, help="depth-lag drafter (adds mode depth)")
    ap.add_argument("--depth-exit", type=int, default=18, help="target layers run on the prefix (depth mode)")
    ap.add_argument("--proxies", nargs="*", default=[],
                    help="cheap target copies (load_mid specs, e.g. bnb4:Qwen/Qwen3-8B): mode <kind> redrafts "
                         "with real features up to s and the proxy's features on the prefix s .. s+i-1")
    ap.add_argument("--faker", default=None, help="trained faker (adds mode faker)")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--budgets", nargs="+", type=int, default=[64, 512])
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy trajectories, else sampled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp26_lossdiag")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    lagd = load_draft(args.lag_draft) if args.lag_draft else draft
    depd = load_draft(args.depth_draft) if args.depth_draft else None
    deep = None
    proxies = {spec.split(":", 1)[0]: load_mid(spec) for spec in args.proxies}
    faker = load_faker(args.faker, "cuda") if args.faker else None
    modes = ["tokens"] + (["depth"] if depd else []) + list(proxies) + (["faker"] if faker else []) + ["fresh"]
    stops = stop_ids(target, tok)
    bs = draft.block_size
    D = bs - 1

    stop_rec = defaultdict(list)     # (who) -> list of dicts, one per stop
    gain = defaultdict(list)         # (mode, ebin, dist) -> [(dlogp, dtop1)]
    rank_pos = defaultdict(list)     # offset -> unconditioned truth rank
    accs = defaultdict(list)
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            if args.temp > 0:
                x = target.generate(ids, max_new_tokens=args.max_new, do_sample=True, temperature=args.temp,
                                    top_p=1.0, top_k=0, eos_token_id=stops)[0]
            else:
                x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            out = target(x[None], output_hidden_states=True)
            feats = extract_context_feature(out.hidden_states, draft.target_layer_ids)
            lpt = torch.log_softmax(out.logits[0].float(), -1)
            ent = (-(lpt.exp() * lpt).nan_to_num(0.0).sum(-1)).cpu().tolist()       # ent[p-1] = entropy of the dist. of x[p]
            del out, lpt
            pfeats = {}
            for name, pm in proxies.items():
                po = pm(x[None], output_hidden_states=True, logits_to_keep=1)
                pfeats[name] = extract_context_feature(po.hidden_states, draft.target_layer_ids).to(feats.dtype)
                del po
            for s in range(n0, len(x) - bs - 1, args.stride):
                truth = x[s + 1 : s + bs]
                lg = draft_logits_lagged(draft, target, feats, x, s, 0, bs)
                r_u, l_u = ranks(lg, truth)
                for k in range(D):
                    rank_pos[k + 1].append(r_u[k])
                cl = {"tokens": cond_logits(lagd, target, feats, x, s, bs, "tokens"),
                      "fresh": cond_logits(draft, target, feats, x, s, bs, "fresh")}
                if faker is not None:
                    cvec = context_vectors(draft, feats)[0].float()
                    c_hat = fake_context(faker, draft, target, cvec, x, s, bs - 1)
                    ctx = torch.cat([cvec[:s], c_hat], 0)[None].to(feats.dtype)
                    cl["faker"] = block_logits_ctx(draft, target, ctx, x, [s + i for i in range(1, bs - 1)], bs)
                for name, pf in pfeats.items():
                    mix = torch.cat([feats[:, :s], pf[:, s : s + bs - 1]], 1)
                    cl[name] = cond_logits(draft, target, mix, x, s, bs, "fresh")
                if depd is not None:
                    if deep is None:
                        deep = deep_columns(draft, args.depth_exit, feats.shape[-1], feats.device)
                    cl["depth"] = depth_logits(depd, target, feats, x, s, bs, deep)
                cr = {}
                for m, L in cl.items():           # cr[m][i-1][k] = (rank, logp) of offset i+1+k
                    cr[m] = []
                    for i in range(1, bs - 1):
                        t = x[s + i + 1 : s + bs]
                        cr[m].append(ranks(L[i - 1][: len(t)], t))
                # conditioning gain by distance after position s+i, binned by target entropy there
                for m in cr:
                    for i in range(1, bs - 1):
                        eb = ebin(ent[s + i - 1])
                        rr, ll = cr[m][i - 1]
                        for dist in range(1, min(4, D - i) + 1):
                            k = i + dist                     # offset in the original block
                            gain[(m, eb, dist)].append((ll[dist - 1] - l_u[k - 1],
                                                        float(rr[dist - 1] == 1) - float(r_u[k - 1] == 1)))
                # stops: chain (top-1) and DDTree at each budget
                logq = torch.log_softmax(lg.float(), -1).cpu()
                chain = 0
                while chain < D and r_u[chain] == 1:
                    chain += 1
                who = {"chain": chain + 1}
                for B in args.budgets:
                    who[f"ddtree{B}"] = accepted(set(tree_paths(build_ddtree(lg, B), logq)), truth.tolist())
                for w, a in who.items():
                    accs[w].append(a)
                    k = a                                    # 1-based offset of the first miss
                    if k > D:
                        stop_rec[w].append({"cls": "cap"})
                        continue
                    rec = {"cls": "coverage" if r_u[k - 1] > TOPK else "path", "rank": r_u[k - 1],
                           "ent": ent[s + k - 1], "ent_prev": max(ent[s : s + k - 1], default=0.0)}
                    for m in cr:                             # redraft with the true prefix up to k-1
                        if k >= 2:
                            rr, _ = cr[m][k - 2]
                            rec[m] = rr[0]
                    stop_rec[w].append(rec)
            print(f"[{d}] {len(accs['chain'])} anchors", flush=True)

    rows = []
    for w, recs in stop_rec.items():
        n = len(recs)
        miss = [r for r in recs if r["cls"] != "cap"]
        row = {"who": w, "tau": mean(accs[w]), "stops": n,
               "cap": sum(r["cls"] == "cap" for r in recs) / n,
               "coverage": sum(r["cls"] == "coverage" for r in recs) / n,
               "path": sum(r["cls"] == "path" for r in recs) / n}
        for m in modes:
            ok = [r for r in miss if m in r]
            row[f"fix_{m}@1"] = mean(float(r[m] == 1) for r in ok) if ok else float("nan")
            row[f"fix_{m}@16"] = mean(float(r[m] <= TOPK) for r in ok) if ok else float("nan")
        row["ent_stop"] = mean(r["ent"] for r in miss) if miss else float("nan")
        rows.append(row)
    print_table(rows, ["who", "tau", "stops", "cap", "coverage", "path"] + [f"fix_{m}@1" for m in modes]
                + [f"fix_{m}@16" for m in modes] + ["ent_stop"],
                f"why acceptance stops (T={args.temp}); fix_* = redraft from the stop with true prefix: "
                "tokens only vs real features")
    # stops split by entropy at the stop and before it (chain)
    erows = []
    for w in ("chain", f"ddtree{args.budgets[0]}"):
        by = defaultdict(list)
        for r in stop_rec[w]:
            if r["cls"] != "cap":
                by[ebin(r["ent"])].append(r)
        for eb, rs in sorted(by.items()):
            erows.append({"who": w, "ent_at_stop": eb, "share": len(rs) / len(stop_rec[w]),
                          "coverage": mean(float(r["cls"] == "coverage") for r in rs),
                          **{f"fix_{m}@1": mean(float(r.get(m, 0) == 1) for r in rs) for m in modes}})
    print_table(erows, ["who", "ent_at_stop", "share", "coverage"] + [f"fix_{m}@1" for m in modes],
                "stops by target entropy at the stop position")
    grows = []
    for (m, eb, dist), v in sorted(gain.items()):
        grows.append({"mode": m, "ent_at_branch": eb, "dist": dist, "n": len(v),
                      "dlogp": mean(a for a, _ in v), "dtop1": mean(b for _, b in v)})
    print_table(grows, ["mode", "ent_at_branch", "dist", "n", "dlogp", "dtop1"],
                "gain from conditioning on the true token at a position, by its target entropy and distance")
    prow = [{"offset": k, "top1": mean(float(r == 1) for r in v), "in_top4": mean(float(r <= 4) for r in v),
             "in_top16": mean(float(r <= TOPK) for r in v)} for k, v in sorted(rank_pos.items())]
    print_table(prow, ["offset", "top1", "in_top4", "in_top16"], "unconditioned drafter accuracy by block offset")
    save_json(args.out, {"args": vars(args), "stops": rows, "by_entropy": erows, "gain": grows, "by_offset": prow})


if __name__ == "__main__":
    main()
