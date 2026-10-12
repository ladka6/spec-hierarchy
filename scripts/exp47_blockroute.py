"""Experiment 47: what does block-consistent MoE routing cost in quality? (no training, single GPU)

On MoE targets, verifying a speculative block costs the union of the experts its tokens route to
(exp45/46). A speculation-friendly MoE would let every token of a block draw from one shared set of S
experts, so verification costs at most S experts per layer. To stay a proper autoregressive model the set
must be causal: block b (tokens b*n .. b*n+n-1) uses the S experts with the highest summed router
probability over the PREVIOUS block's tokens (block 0 routes freely). Each token then takes its top-k
inside that set (same weighting rule as the model).

Measured here on models never trained for it, teacher-forced on real text: KL(original || constrained)
and top-1 agreement of the next-token distributions, and the perplexity of the text, for S in --sets and
block length n in --blocks. Also the natural expert union of n consecutive tokens (unconstrained), and the
verification-cost ceiling each S implies (weights read for a block / for one token).

  python scripts/exp47_blockroute.py --models allenai/OLMoE-1B-7B-0924 Qwen/Qwen1.5-MoE-A2.7B
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.utils import print_table, save_json  # noqa: E402

STATE = {"S": None, "n": None, "union": None}


def routed_forward(self, hidden_states):
    """Drop-in for the Qwen2/Qwen3/OLMoE TopKRouter forward: optional causal block-shared expert sets."""
    h = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(h, self.weight)
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    T, E = probs.shape
    n = STATE["n"]
    if STATE["union"] is not None:                             # record natural block unions (unconstrained run)
        idx = probs.topk(self.top_k, -1).indices
        for nn, acc in STATE["union"].items():
            for b in range(T // nn):
                acc.append(len(torch.unique(idx[b * nn : (b + 1) * nn])))
    if STATE["S"] is not None:
        S = STATE["S"]
        allowed = torch.ones_like(probs, dtype=torch.bool)
        for b in range(1, (T + n - 1) // n):
            prev = probs[(b - 1) * n : b * n].sum(0)
            keep = torch.zeros(E, dtype=torch.bool, device=probs.device)
            keep[prev.topk(S).indices] = True
            allowed[b * n : (b + 1) * n] = keep
        probs = probs.masked_fill(~allowed, 0.0)
    top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        top_v = top_v / top_v.sum(-1, keepdim=True).clamp_min(1e-9)
    return logits, top_v.to(logits.dtype), top_i


def patch(model):
    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__.endswith("TopKRouter"):
            mod.forward = types.MethodType(routed_forward, mod)
            n += 1
    return n


def weights_per_token(cfg, experts_per_layer):
    H, L = cfg.hidden_size, cfg.num_hidden_layers
    hd = getattr(cfg, "head_dim", None) or H // cfg.num_attention_heads
    attn = H * hd * (2 * cfg.num_attention_heads + 2 * cfg.num_key_value_heads)
    inter = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
    shared = getattr(cfg, "shared_expert_intermediate_size", 0) or 0
    non_expert = L * (attn + H * cfg.num_experts + 3 * H * shared) + cfg.vocab_size * H
    return non_expert + L * experts_per_layer * 3 * H * inter


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["allenai/OLMoE-1B-7B-0924", "Qwen/Qwen1.5-MoE-A2.7B"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-tokenizer", default="Qwen/Qwen3-8B", help="tokenizer of the data shards")
    ap.add_argument("--n-seqs", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--blocks", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--sets", nargs="+", type=int, default=[0, 8, 12, 16, 24, 32], help="0 = top-k only")
    ap.add_argument("--out", default="exp47_blockroute")
    args = ap.parse_args()

    ttok = AutoTokenizer.from_pretrained(args.text_tokenizer)
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
        if len(data) >= 4 * args.n_seqs:
            break
    random.Random(0).shuffle(data)
    texts = [ttok.decode(d["ids"], skip_special_tokens=True) for d in data[: args.n_seqs]]

    rows, union_rows = [], []
    for mid in args.models:
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
        cfg = model.config
        k = cfg.num_experts_per_tok
        print(f"[{mid}] patched {patch(model)} routers; {cfg.num_experts} experts, top-{k}", flush=True)
        seqs = [tok(t, return_tensors="pt").input_ids[0, : args.max_len].cuda() for t in texts]
        seqs = [s for s in seqs if len(s) > 2 * max(args.blocks)]
        unions = {n: [] for n in args.blocks}
        acc = {(n, S): [0.0, 0.0, 0.0, 0.0, 0.0] for n in args.blocks for S in args.sets}   # kl, agree, nll_c, nll_r, cnt
        for i, s in enumerate(seqs):
            STATE.update(S=None, n=None, union=unions)
            ref = F.log_softmax(model(s[None]).logits[0].float(), -1)
            STATE["union"] = None
            for n in args.blocks:
                for S in args.sets:
                    STATE.update(S=(S or k), n=n)
                    lp = F.log_softmax(model(s[None]).logits[0].float(), -1)
                    m = slice(n, len(s) - 1)                    # positions inside constrained blocks
                    tgt = s[n + 1 :]
                    a_ = acc[(n, S)]
                    a_[0] += float((ref[m].exp() * (ref[m] - lp[m])).sum())
                    a_[1] += float((ref[m].argmax(-1) == lp[m].argmax(-1)).sum())
                    a_[2] -= float(lp[m].gather(-1, tgt[:, None]).sum())
                    a_[3] -= float(ref[m].gather(-1, tgt[:, None]).sum())
                    a_[4] += len(tgt)
            if (i + 1) % 20 == 0:
                print(f"[{mid}] {i + 1}/{len(seqs)} sequences", flush=True)
        for n in args.blocks:
            nat = sum(unions[n]) / len(unions[n])
            union_rows.append({"model": mid.split("/")[-1], "n": n, "natural_union": nat, "top_k": k,
                               "cost_vs_1tok": weights_per_token(cfg, nat) / weights_per_token(cfg, k)})
            for S in args.sets:
                kl, ag, nc, nr, cnt = acc[(n, S)]
                Sx = S or k
                rows.append({"model": mid.split("/")[-1], "n": n, "S": Sx, "KL": kl / cnt, "top1_agree": ag / cnt,
                             "ppl": float(torch.exp(torch.tensor(nc / cnt))),
                             "ppl_orig": float(torch.exp(torch.tensor(nr / cnt))),
                             "verify_cost_cap": weights_per_token(cfg, Sx) / weights_per_token(cfg, k)})
        STATE.update(S=None, n=None)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print_table(union_rows, list(union_rows[0]), "natural expert union of n consecutive tokens (per layer) and its "
                                                 "verification cost relative to one token (weights read)")
    print_table(rows, list(rows[0]), "causal block-shared expert sets (S experts per block, chosen from the previous "
                                     "block): quality vs the original routing, and the verification-cost cap")
    save_json(args.out, {"args": vars(args), "natural": union_rows, "constrained": rows})


if __name__ == "__main__":
    main()
