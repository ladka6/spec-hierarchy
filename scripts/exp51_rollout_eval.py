"""Experiment 51: is the 'cur' result (exp49: a block routed through its own top-S experts) real or leakage?

exp49 --route cur trains/evaluates with each block's expert set chosen from the block's OWN true tokens. The set at
position t then depends on token t+1, the token being predicted: possible leakage. Here the trained checkpoint is
evaluated with sets built WITHOUT the true block:
  rollout   for every block b (start p = b*n), the ORIGINAL model writes a greedy continuation of n-1 tokens from the
            true prefix s[:p+1] (anchor = s[p]); the student routes [anchor + continuation] and each layer keeps the
            top-S of that block (blocks before b use their own stored sets, as when they were verified). The final
            evaluation pass on the true text uses these fixed sets. The draft depends only on the prefix: no leak.
            Optimistic about the drafter (a self-greedy draft), not about the routing.
  cur       sets from the true block (the leaky number exp49 reported)
  block     sets from the previous block (causal design of exp47/49)
Reported per variant: KL, top-1 agreement and perplexity vs the original model, and expert union of 16-token windows.

  python scripts/exp51_rollout_eval.py --ckpt /scratch-shared/$USER/hspec_moe/routetrain_router_experts_S16_1000_routecur
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exp49_routetrain as R  # noqa: E402
from exp47_blockroute import weights_per_token  # noqa: E402


def load_val(args, tok):
    """Same selection as exp49 (seed 0 shuffle, same filters): the first val_seqs sequences are its validation set."""
    ttok = AutoTokenizer.from_pretrained(args.text_tokenizer)
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
        if len(data) >= 2 * (args.train_seqs + args.val_seqs):
            break
    random.Random(0).shuffle(data)
    seqs = []
    for d in data:
        ids = tok(ttok.decode(d["ids"], skip_special_tokens=True), return_tensors="pt").input_ids[0, : args.max_len]
        if len(ids) > 2 * args.n:
            seqs.append(ids)
        if len(seqs) >= args.val_seqs:
            break
    return seqs[: args.eval_seqs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="exp49 output dir with routers.pt")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--mode", default="router+experts")
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--train-seqs", type=int, default=3000)
    ap.add_argument("--val-seqs", type=int, default=60)
    ap.add_argument("--eval-seqs", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--variants", nargs="+", default=["cur", "block", "rollout"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    val = load_val(args, tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)
    routers, _, extra = R.add_adapters(model, args.mode)
    ck = torch.load(Path(args.ckpt) / "routers.pt")
    for r, w in zip(routers, ck["routers"]):
        r.weight.data.copy_(w.to(r.weight.device))
    for key, params in extra.items():
        for p, w in zip(params, ck[key]):
            p.data.copy_(w.to(p.device))
    cfg = model.config
    E, k = cfg.num_experts, cfg.num_experts_per_tok
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"loaded {args.ckpt}; {len(val)} sequences, S={args.S} n={args.n}", flush=True)

    @torch.no_grad()
    def run(s, route):
        R.STATE.update(teacher=True, S=None, rec=None)
        ref = F.log_softmax(model(s[None]).logits[0].float(), -1)
        R.STATE.update(teacher=False, S=args.S, n=args.n, route="fixed" if route == "rollout" else route, fixed={})
        if route == "rollout":
            T, n = len(s), args.n
            for b in range(1, (T + n - 1) // n):
                p = b * n
                m = min(n - 1, T - p - 1)
                prefix = s[: p + 1]
                if m > 0:
                    R.STATE.update(teacher=True)
                    out = model.generate(prefix[None], max_new_tokens=m, min_new_tokens=m, do_sample=False,
                                         pad_token_id=pad)[0]
                    R.STATE.update(teacher=False)
                else:
                    out = prefix
                model(out[None])                   # stores block b's sets (from anchor + draft) for every layer
        R.STATE["rec"] = []
        lp = F.log_softmax(model(s[None]).logits[0].float(), -1)
        rec, R.STATE["rec"] = R.STATE["rec"], None
        return ref, lp, rec

    for route in args.variants:
        t0 = time.time()
        kl = ag = nc = nr = cnt = 0.0
        un = [0.0] * 4
        for i, s in enumerate(val):
            s = s.cuda()
            ref, lp, rec = run(s, route)
            m = slice(args.n, len(s) - 1)
            tgt = s[args.n + 1 :]
            kl += float((ref[m].exp() * (ref[m] - lp[m])).sum())
            ag += float((ref[m].argmax(-1) == lp[m].argmax(-1)).sum())
            nc -= float(lp[m].gather(-1, tgt[:, None]).sum())
            nr -= float(ref[m].gather(-1, tgt[:, None]).sum())
            cnt += len(tgt)
            for j, v in enumerate(R.window_unions(rec, E, args.n)):
                un[j] += v
            if route == "rollout" and (i + 1) % 10 == 0:
                print(f"  [rollout] {i + 1}/{len(val)} sequences, {time.time() - t0:.0f}s", flush=True)
        ua, uu = un[0] / max(un[1], 1), un[2] / max(un[3], 1)
        c = weights_per_token(cfg, uu) / weights_per_token(cfg, k)
        print(f"[{route:8s}] KL {kl / cnt:.4f} agree {ag / cnt:.4f} ppl {math.exp(nc / cnt):.3f} "
              f"(original {math.exp(nr / cnt):.3f}) | experts/layer aligned {ua:.1f}, any offset {uu:.1f} ({c:.2f}x)",
              flush=True)


if __name__ == "__main__":
    main()
