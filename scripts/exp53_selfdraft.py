"""Experiment 53 (foundation B, zero-shot, one GPU): is an MoE a good drafter of its own next block when it may only
use the experts the previous block already loaded?

In memory-bound decoding an MoE pays per expert read. If the pass that verifies block r could also draft block r+1
using only the experts already read for block r, drafting would cost almost no extra weights. Measured on OLMoE
(free routing, no training), at block starts p = b*n of real text:
  1. footprint: per layer, the experts the true block b uses that block b-1 did not (new experts), and |U_{b-1}|.
  2. teacher-forced: on the true block b, top-1 agreement and KL between the full model and the model restricted
     to U (each token takes its top-k inside U).
  3. drafting: from the true prefix, greedy continuation of n tokens by the full model and by the restricted model;
     accepted length = matching prefix (greedy verification of the restricted model's draft by the full model).
Sets U: union of the experts of the previous block (U1), of the previous two blocks (U2), and, as a control, a
random set of the same size as U1 (per layer).

  python scripts/exp53_selfdraft.py --data .../hspec_lagft/data
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exp49_routetrain as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-seqs", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--blocks-per-seq", type=int, default=8)
    args = ap.parse_args()
    n = args.n

    ttok = AutoTokenizer.from_pretrained(args.text_tokenizer)
    tok = AutoTokenizer.from_pretrained(args.model)
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
        if len(data) >= 4 * args.n_seqs:
            break
    random.Random(1).shuffle(data)
    seqs = []
    for d in data:
        ids = tok(ttok.decode(d["ids"], skip_special_tokens=True), return_tensors="pt").input_ids[0, : args.max_len]
        if len(ids) >= 6 * n:
            seqs.append(ids)
        if len(seqs) >= args.n_seqs:
            break

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)
    routers, _, _ = R.add_adapters(model, "router")              # routers untouched: the original model
    E, k, L = model.config.num_experts, model.config.num_experts_per_tok, len(routers)
    gen = torch.Generator().manual_seed(0)

    @torch.no_grad()
    def forward(ids, sets=None, cache=None):
        if sets is None:
            R.STATE.update(teacher=True, S=None)
        else:
            R.STATE.update(teacher=False, S=1, n=0, route="set", sets=sets, kin=0)
        out = model(ids[None], past_key_values=cache, use_cache=cache is not None)
        R.STATE.update(teacher=False, S=None)
        return out

    @torch.no_grad()
    def greedy(prefix, steps, sets):
        cache = DynamicCache()
        out = forward(prefix, None, cache)                          # prefix with free routing (already committed)
        nxt = out.logits[0, -1].argmax()
        toks = []
        for _ in range(steps):
            toks.append(nxt)
            out = forward(nxt.view(1), sets, cache)
            nxt = out.logits[0, -1].argmax()
        return torch.stack(toks)

    stats = {nm: {"agree": 0.0, "kl": 0.0, "cnt": 0, "acc": [], "size": 0.0} for nm in ("U1", "U2", "random")}
    new_exp = size1 = blocks = 0.0
    full_acc_self = []
    for si, s in enumerate(seqs):
        s = s.cuda()
        R.STATE["rec"] = []
        ref = forward(s).logits[0].float().log_softmax(-1)
        rec = R.STATE["rec"]                                        # per layer [T, E] bool of the experts used
        R.STATE["rec"] = None
        T = len(s)
        starts = list(range(2 * n, T - n, n))
        for p in random.Random(si).sample(starts, min(args.blocks_per_seq, len(starts))):
            prev1 = {l: rec[l][p - n : p].any(0) for l in range(L)}
            prev2 = {l: rec[l][p - 2 * n : p].any(0) for l in range(L)}
            cur = [rec[l][p : p + n].any(0) for l in range(L)]
            new_exp += sum(float((cur[l] & ~prev1[l]).sum()) for l in range(L)) / L
            size1 += sum(float(prev1[l].sum()) for l in range(L)) / L
            blocks += 1
            rand = {}
            for l in range(L):
                m = torch.zeros(E, dtype=torch.bool, device=s.device)
                m[torch.randperm(E, generator=gen)[: int(prev1[l].sum())].to(s.device)] = True
                rand[l] = m
            full = greedy(s[:p], n, None)
            for nm, sets in (("U1", prev1), ("U2", prev2), ("random", rand)):
                st = stats[nm]
                # teacher-forced on the true block: tokens before p routed freely, the block inside the set
                cache = DynamicCache()
                forward(s[:p], None, cache)
                lp = forward(s[p : p + n], sets, cache).logits[0].float().log_softmax(-1)
                r = ref[p : p + n]
                st["agree"] += float((lp.argmax(-1) == r.argmax(-1)).sum())
                st["kl"] += float((r.exp() * (r - lp)).sum())
                st["cnt"] += n
                st["size"] += sum(float(sets[l].sum()) for l in range(L)) / L
                draft = greedy(s[:p], n, sets)
                st["acc"].append(int((draft == full).long().cumprod(0).sum()))
            full_acc_self.append(int((full == s[p : p + n]).long().cumprod(0).sum()))
        print(f"{si + 1}/{len(seqs)} sequences", flush=True)

    print(f"\n{int(blocks)} blocks of {n} tokens, OLMoE ({E} experts, top-{k}, {L} layers)")
    print(f"experts per layer of the previous block: {size1 / blocks:.1f}; experts of the next block not in it: "
          f"{new_exp / blocks:.1f} per layer")
    print(f"(for reference: full-model greedy matches the true text for {sum(full_acc_self) / len(full_acc_self):.2f} "
          f"tokens on average)")
    print(f"\n{'set':8s} {'experts/layer':>13s} {'TF agree':>9s} {'TF KL':>7s} {'self-draft accepted (of ' + str(n) + ')':>28s}")
    for nm, st in stats.items():
        acc = torch.tensor(st["acc"]).float()
        print(f"{nm:8s} {st['size'] / len(st['acc']):13.1f} {st['agree'] / st['cnt']:9.3f} {st['kl'] / st['cnt']:7.3f} "
              f"{float(acc.mean()):16.2f} (median {float(acc.median()):.0f}, all {n}: {float((acc == n).float().mean()):.0%})")


if __name__ == "__main__":
    main()
