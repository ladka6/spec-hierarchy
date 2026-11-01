"""Experiment 93: selecting the path through DFlash's candidate lattice (offline, T=0, single block).

exp92: the target's token is in DFlash's top-16 at the right position 80-99% of the time; a perfect choice through
these position-locked candidates would accept 12.9 tokens per verification vs 7.99 for the argmax. Lightweight
selectors (DSpark, DFlash2, xPress) reach ~+16-30%. Here: how much does each information source recover, at what cost?

Lattice: top-16 tokens per block position from one DFlash pass. Selectors:
  argmax         plain DFlash
  width N        best-first tree over the drafter's marginals (top-8 children), N nodes, all verified by the target
                 (DDTree-style; the target itself selects)
  <scorer>       a model ranks only the 16 candidates per position, greedy left to right: rank / rank+draft (scorer
                 log-prob + drafter log-prob) / free (the scorer's own argmax over the full vocabulary, i.e. generating)
  oracle         best path through the lattice
For greedy T=0 a walk is right up to its first wrong pick, so the scorer columns are computed exactly from ONE
teacher-forced scorer pass over the target's text: accepted = 1 + first position where the pick is wrong.
Also per scorer: ranking agreement (pick right | target token in the lattice, true prefix) vs generation agreement
(full-vocab argmax right | true prefix): is choosing among 16 easier than generating?
Cost (ms): verify t_V(nodes) as exp89, drafter --t-draft, each scorer its --costs entry (one pass over 16 tokens).

  python scripts/exp93_selection.py --scorers hf:Qwen/Qwen3-0.6B rtn4:Qwen/Qwen3-8B --costs 1.0 4.5
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp89_hedge import t_verify  # noqa: E402
from exp90_treehier import best_first_tree  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def first_fail(ok):
    for k, v in enumerate(ok):
        if not v:
            return k
    return len(ok)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--scorers", nargs="+", default=["hf:Qwen/Qwen3-0.6B", "rtn4:Qwen/Qwen3-8B", "rtn3:Qwen/Qwen3-8B"])
    ap.add_argument("--costs", nargs="+", type=float, default=[1.0, 4.5, 3.6], help="ms per scorer pass")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--t-draft", type=float, default=2.6)
    ap.add_argument("--out", default="exp93_selection")
    args = ap.parse_args()
    assert len(args.costs) == len(args.scorers)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)

    # phase 1: target text, lattices
    seqs = []          # (x, [(s, cand [D][K], draft logp of cand [D][K], truth [D])])
    acc = {"argmax": [], "oracle": [], **{f"width {N}": [] for N in args.widths}}
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            xl, blocks = x.tolist(), []
            for s in range(n0, len(x) - bs - 1, args.stride):
                truth = xl[s + 1 : s + 1 + D]
                lp = torch.log_softmax(blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].float(), -1)
                v, i = lp.topk(args.topk, -1)
                cand, clp = i.tolist(), v.tolist()
                y = [c[0] for c in cand]
                acc["argmax"].append(matched(y, truth) + 1)
                acc["oracle"].append(first_fail([truth[k] in cand[k] for k in range(D)]) + 1)
                pr = lp.exp()
                for N in args.widths:
                    tree = best_first_tree(pr, N, k=8)
                    n = 0
                    while n < D and tuple(truth[: n + 1]) in tree:
                        n += 1
                    acc[f"width {N}"].append(n + 1)
                blocks.append((s, cand, clp, truth))
            seqs.append((x.cpu(), blocks))
        print(f"[{dname}] {sum(len(b) for _, b in seqs)} blocks", flush=True)
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()

    # phase 2: one teacher-forced pass per scorer per sequence
    agree = {}
    for spec in list(args.scorers):
        try:
            model = load_mid(spec)
        except Exception as e:                                    # e.g. a model that is not in the offline cache
            print(f"[{spec}] skipped: {e}", flush=True)
            i = args.scorers.index(spec)
            args.scorers.pop(i)
            args.costs.pop(i)
            continue
        dev = next(model.parameters()).device
        r_ok, r_tot, g_ok, g_tot = 0, 0, 0, 0
        for mode in ("rank", "rank+draft", "free"):
            acc[f"{spec} {mode}"] = []
        for x, blocks in seqs:
            lg = torch.log_softmax(model(x[None].to(dev), logits_to_keep=0).logits[0].float(), -1)   # [T, V]
            for s, cand, clp, truth in blocks:
                L = lg[s : s + D]                                  # predictions for s+1 .. s+D (true prefix)
                ci = torch.tensor(cand, device=dev)
                sc = L.gather(-1, ci)                              # [D, K] scorer log-probs of the candidates
                pick_r = ci.gather(-1, sc.argmax(-1, keepdim=True))[:, 0].tolist()
                pick_rd = ci.gather(-1, (sc + torch.tensor(clp, device=dev)).argmax(-1, keepdim=True))[:, 0].tolist()
                free = L.argmax(-1).tolist()
                acc[f"{spec} rank"].append(first_fail([pick_r[k] == truth[k] for k in range(D)]) + 1)
                acc[f"{spec} rank+draft"].append(first_fail([pick_rd[k] == truth[k] for k in range(D)]) + 1)
                acc[f"{spec} free"].append(first_fail([free[k] == truth[k] for k in range(D)]) + 1)
                for k in range(D):
                    g_tot += 1
                    g_ok += free[k] == truth[k]
                    if truth[k] in cand[k]:
                        r_tot += 1
                        r_ok += pick_r[k] == truth[k]
        agree[spec] = {"ranking agreement (among 16)": r_ok / max(r_tot, 1),
                       "generation agreement (full vocab)": g_ok / max(g_tot, 1)}
        print(f"[{spec}] {agree[spec]}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    t_base = t_verify(bs) + args.t_draft
    rate0 = (sum(acc["argmax"]) / len(acc["argmax"])) / t_base
    cost = {"argmax": t_base, "oracle": t_base}
    for N in args.widths:
        cost[f"width {N}"] = t_verify(N) + args.t_draft
    for spec, c in zip(args.scorers, args.costs):
        for mode in ("rank", "rank+draft", "free"):
            cost[f"{spec} {mode}"] = t_base + c
    rows = []
    for k, v in acc.items():
        a = sum(v) / len(v)
        rows.append({"selector": k, "accepted": a, "time ms": cost[k], "tokens/ms": a / cost[k],
                     "vs DFlash": a / cost[k] / rate0})
    print_table(rows, list(rows[0]), f"1. selection through the top-{args.topk} lattice ({len(acc['argmax'])} blocks)")
    rows2 = [{"scorer": s, **v} for s, v in agree.items()]
    print_table(rows2, list(rows2[0]), "2. choosing among 16 vs generating (teacher-forced, true prefix)")
    save_json(args.out, {"args": vars(args), "selection": rows, "agreement": rows2})


if __name__ == "__main__":
    main()
