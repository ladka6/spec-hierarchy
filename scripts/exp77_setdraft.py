"""Experiment 77: set-valued block drafting. One drafter pass proposes K whole blocks, each coherent on its own,
trained with a winner-takes-all loss so the K versions specialise on different continuations; the target verifies
all K as one trie (Colab-ready; --data for a large training set).

Why: DFlash predicts every block position on its own. Where the target's continuation can branch, its per-position
predictions mix the branches and the argmax block can stitch tokens of different continuations together. Deciding
the branch needs target information about the drafted tokens; here the verify pass (width is nearly free at batch 1)
decides it, among K versions the drafter proposes.

Model: DFlash (from its checkpoint, fp32 master weights under bf16 autocast) + K mode adapters on its last hidden
state: h_k = h + B_k(A_k h) (rank --rank; mode 0 starts as the identity, the others with small random weights so the
modes can separate). Each mode gives a full block through the target's LM head (as DFlash).

Loss (relaxed winner-takes-all): per block, L_k = position-weighted cross-entropy of mode k's block on the true
tokens (weights gamma^(i-1), as DFlash); the best mode gets weight 1 - eps, the others eps / (K - 1).

Evaluation on held-out anchors (tau out of 16; trie = the K chains merged, verified in one target pass):
  dflash_orig            the untouched checkpoint, one chain
  dflash_orig_branchK    the same drafter's best K chains WITHOUT dependence: chain j swaps in the 2nd choice at
                         its j-th least confident position (what one DFlash pass can offer for K chains)
  control                DFlash fine-tuned the same way with K = 1 (same data, steps, lr, seed)
  set_K                  the K-mode drafter: tau of the trie (best chain), tau of mode 0, trie size, how often each
                         mode is the best one (are the modes used?), token disagreement between modes

  python scripts/exp77_setdraft.py --cache /content/seqs.pt --K 3
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import _output_head, extract_context_feature  # noqa: E402

from hspec.lagtrain import pack  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


def trie_size(chains):
    nodes = set()
    for c in chains:
        for k in range(1, len(c) + 1):
            nodes.add(tuple(c[:k]))
    return len(nodes)


class ModeAdapters(nn.Module):
    def __init__(self, H, K, r, init_std=0.02):
        super().__init__()
        self.A = nn.Parameter(torch.randn(K, r, H) * init_std)
        self.B = nn.Parameter(torch.randn(K, H, r) * init_std)
        with torch.no_grad():
            self.B[0].zero_()                    # mode 0 = the drafter's own block at the start

    def forward(self, h):                        # [A, B, H] -> [K, A, B, H]
        hf = h.float()
        d = torch.einsum("abh,krh->kabr", hf, self.A)
        d = torch.einsum("kabr,khr->kabh", d, self.B)
        return hf[None] + d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences (eval; training unless --data)")
    ap.add_argument("--data", default="", help="directory of gen_traindata shards: train on these, eval on --cache")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.05, help="relaxed winner-takes-all: weight shared by the losers")
    ap.add_argument("--keep0", type=float, default=1.0,
                    help="weight of mode 0's own full loss on every block (mode 0 stays a full drafter; the WTA term "
                         "then trains the others to cover what it misses). 0 = plain winner-takes-all")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--blocks", type=int, default=8, help="anchors per sequence per step")
    ap.add_argument("--lr", type=float, default=1e-4, help="drafter")
    ap.add_argument("--lr-m", type=float, default=1e-3, help="mode adapters")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--anchors", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    args.save = args.save or f"exp77_setdraft_K{args.K}_keep{args.keep0:g}"
    dev = "cuda"
    t0 = time.time()

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    nv = int(len(seqs) * args.val_frac)
    if args.data:
        train = []
        for f in sorted(Path(args.data).glob("*.pt")):
            train += [(r["ids"].long(), int(r["n_prompt"])) for r in torch.load(f)]
        train = [(ids, lo) for ids, lo in train if len(ids) - lo > 20]
        val = seqs[: max(nv, 1)]
    else:
        val, train = seqs[:nv], seqs[nv:]
    print(f"[data] {len(train)} train / {len(val)} eval sequences", flush=True)

    target = load_target(args.target, dev)
    for p_ in target.parameters():
        p_.requires_grad_(False)
    head = _output_head(target)
    probe = load_draft(args.draft, dev)
    lids, bs = list(probe.target_layer_ids), probe.block_size
    Hd = probe.config.hidden_size
    B = bs - 1
    del probe
    torch.cuda.empty_cache()
    w = torch.tensor([args.gamma ** i for i in range(B)], device=dev)

    @torch.no_grad()
    def target_feats(x):
        out = target(x[None], output_hidden_states=True, use_cache=False)
        return extract_context_feature(out.hidden_states, lids)

    def drafter_hidden(draft, feats, x, anchors):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            noise, pos, mask, starts = pack(draft, target, feats, x, [(s, 0) for s in anchors], bs)
            hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                        past_key_values=None, use_cache=False)[0]
            return torch.stack([hid[st : st + B] for st in starts])                    # [A, B, Hd]

    def logits_of(draft, h):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return draft.compute_logits(h, head)

    def anchors_in(ids, lo, n, rng):
        hi = len(ids) - bs - 1
        if hi <= lo + 1:
            return []
        return sorted(rng.sample(range(lo + 1, hi), min(n, hi - lo - 1)))

    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 1, len(ids) - bs - 1)]
    random.Random(0).shuffle(cand)
    ev = {}
    for i, s in sorted(cand[: args.anchors]):
        ev.setdefault(i, []).append(s)

    @torch.no_grad()
    def evaluate(draft, modes, K):
        """modes = None: one chain + the no-dependence branch baseline with K chains."""
        draft.eval()
        r = {"tau1": 0.0, "tauK": 0.0, "nodes": 0.0, "wins": torch.zeros(K), "disagree": 0.0, "n": 0}
        for i, an in ev.items():
            x = val[i][0].to(dev)
            feats = target_feats(x)
            h = drafter_hidden(draft, feats, x, an)
            if modes is None:
                lg = logits_of(draft, h).float()                                          # [A, B, V]
                top2 = lg.topk(2, -1)
                margin = (top2.values[..., 0] - top2.values[..., 1])
                chains_all = []
                for a in range(len(an)):
                    base = top2.indices[a, :, 0].tolist()
                    order = margin[a].argsort().tolist()                                 # least confident first
                    chains = [base]
                    for j in range(K - 1):
                        c = list(base)
                        c[order[j]] = int(top2.indices[a, order[j], 1])
                        chains.append(c)
                    chains_all.append(chains)
            else:
                lg = logits_of(draft, modes(h)).float()                                  # [K, A, B, V]
                pred = lg.argmax(-1)
                chains_all = [[pred[k, a].tolist() for k in range(K)] for a in range(len(an))]
            for a, s in enumerate(an):
                truth = x[s + 1 : s + 1 + B].tolist()
                ms = [matched(c, truth) for c in chains_all[a]]
                r["tau1"] += ms[0] + 1
                r["tauK"] += max(ms) + 1
                r["nodes"] += trie_size(chains_all[a])
                r["wins"][int(torch.tensor(ms).argmax())] += 1
                if K > 1:
                    cs = chains_all[a]
                    r["disagree"] += sum(sum(u != v for u, v in zip(cs[p], cs[q])) / B
                                         for p in range(K) for q in range(p + 1, K)) / (K * (K - 1) / 2)
                r["n"] += 1
        n = r["n"]
        return {"tau_mode0": r["tau1"] / n, "tau_trie": r["tauK"] / n, "trie_nodes": r["nodes"] / n,
                "best_mode_share": [round(float(v) / n, 3) for v in r["wins"]], "mode_disagreement": r["disagree"] / n}

    rows = []
    d0 = load_draft(args.draft, dev)
    r = evaluate(d0, None, args.K)
    rows.append({"condition": "dflash_orig", "K": 1, "tau": r["tau_mode0"], "trie_nodes": B,
                 "best_mode_share": "-", "mode_disagreement": "-"})
    rows.append({"condition": f"dflash_orig, {args.K} chains w/o dependence (2nd choice at least confident pos.)",
                 "K": args.K, "tau": r["tau_trie"], "trie_nodes": r["trie_nodes"],
                 "best_mode_share": r["best_mode_share"], "mode_disagreement": r["mode_disagreement"]})
    print(f"[eval] dflash_orig: {r}", flush=True)
    del d0
    torch.cuda.empty_cache()

    def finetune(K):
        torch.manual_seed(0)
        rng = random.Random(2)
        draft = load_draft(args.draft, dev).float()      # fp32 master weights (bf16 + small lr rounds updates away)
        draft.train()
        modes = ModeAdapters(Hd, K, args.rank).to(dev) if K > 1 else None
        groups = [{"params": [p for p in draft.parameters() if p.requires_grad], "lr": args.lr}]
        if modes is not None:
            groups.append({"params": list(modes.parameters()), "lr": args.lr_m})
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))
        log = []
        for step in range(args.steps):
            ids, lo = train[rng.randrange(len(train))]
            x = ids.to(dev)
            an = anchors_in(ids, lo, args.blocks, rng)
            if not an:
                continue
            feats = target_feats(x)
            labels = torch.stack([x[s + 1 : s + 1 + B] for s in an])                       # [A, B]
            h = drafter_hidden(draft, feats, x, an)
            hk = modes(h) if modes is not None else h[None]                                 # [K, A, B, H]
            lg = logits_of(draft, hk)                                                       # [K, A, B, V]
            ce = F.cross_entropy(lg.float().flatten(0, 2), labels[None].expand(lg.shape[0], -1, -1).reshape(-1),
                                 reduction="none").view(lg.shape[0], len(an), B)
            per = (ce * w).sum(-1) / w.sum()                                                # [K, A] block losses
            if K > 1:
                best = per.detach().argmin(0)                                               # [A]
                wt = torch.full_like(per, args.eps / (K - 1))
                wt[best, torch.arange(len(an), device=dev)] = 1 - args.eps
                loss = (wt * per).sum(0).mean() + args.keep0 * per[0].mean()
                share = torch.bincount(best, minlength=K).float() / len(an)
            else:
                loss = per.mean()
                share = torch.ones(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for g in groups:
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            opt.step()
            sched.step()
            log.append((float(loss.detach()), share.cpu()))
            if step % 100 == 0 or step == args.steps - 1:
                tail = log[-100:]
                sh = torch.stack([b for _, b in tail]).mean(0).tolist()
                print(f"[train K={K}] step {step}: loss {sum(a for a, _ in tail)/len(tail):.4f}  "
                      f"best-mode share {[round(v, 2) for v in sh]}  {time.time()-t0:.0f}s", flush=True)
        return draft, modes

    if not args.skip_control:
        draft, _ = finetune(1)
        r = evaluate(draft, None, args.K)
        rows.append({"condition": "control: DFlash fine-tuned (K = 1)", "K": 1, "tau": r["tau_mode0"],
                     "trie_nodes": B, "best_mode_share": "-", "mode_disagreement": "-"})
        rows.append({"condition": f"control, {args.K} chains w/o dependence", "K": args.K, "tau": r["tau_trie"],
                     "trie_nodes": r["trie_nodes"], "best_mode_share": r["best_mode_share"],
                     "mode_disagreement": r["mode_disagreement"]})
        print(f"[eval] control: {r}", flush=True)
        del draft
        torch.cuda.empty_cache()

    draft, modes = finetune(args.K)
    r = evaluate(draft, modes, args.K)
    rows.append({"condition": f"set drafter: mode 0 alone", "K": 1, "tau": r["tau_mode0"], "trie_nodes": B,
                 "best_mode_share": "-", "mode_disagreement": "-"})
    rows.append({"condition": f"set drafter: {args.K} modes as a trie", "K": args.K, "tau": r["tau_trie"],
                 "trie_nodes": r["trie_nodes"], "best_mode_share": r["best_mode_share"],
                 "mode_disagreement": r["mode_disagreement"]})
    print(f"[eval] set: {r}", flush=True)
    torch.save({"modes": modes.state_dict()}, f"{args.save}_modes.pt")

    print_table(rows, list(rows[0]), f"set-valued block drafting (K = {args.K}, winner-takes-all); tau out of {B + 1}; "
                                     f"compare the trie rows at similar trie size")
    save_json(args.save, {"args": vars(args), "rows": rows})
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
