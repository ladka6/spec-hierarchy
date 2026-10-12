"""Experiment 49: can a small adapter teach an MoE causal block-shared routing? (training pilot, one GPU)

exp47: forcing block-shared expert sets on an MoE never trained for it hurts quality badly (OLMoE, n=16:
S=16 -> perplexity 9.2 vs 3.2; S=32 -> 4.5). With block-shared routing, verifying any part of a block
costs at most S experts per layer (OLMoE n=16: natural union 38 experts -> S=16 halves verification).

Here the model is fine-tuned to work under the constraint, with distillation to the ORIGINAL model
(teacher = same weights with the original routers and free routing, so no second copy in memory):
  router       only the router weights train (2M params for OLMoE): the "locality adapter"
  router+attn  routers + LoRA on the attention projections (hidden states can adapt to the restricted
               experts; the LoRA is switched off for the teacher pass)
Loss: KL(teacher || student) on positions inside constrained blocks. Reported on held-out text, before
and during training: KL, top-1 agreement with the original, perplexity vs the original's.

  python scripts/exp49_routetrain.py --mode router --S 16 --n 16 --data ...
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STATE = {"S": None, "n": None, "teacher": False}


def routed_forward(self, hidden_states):
    h = hidden_states.reshape(-1, self.hidden_dim)
    W = self.orig_weight if STATE["teacher"] else self.weight
    logits = F.linear(h, W.to(h.dtype))
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    T, E = probs.shape
    if STATE["S"] is not None and not STATE["teacher"]:
        n, S = STATE["n"], STATE["S"]
        allowed = torch.ones_like(probs, dtype=torch.bool)
        with torch.no_grad():
            for b in range(1, (T + n - 1) // n):
                keep = torch.zeros(E, dtype=torch.bool, device=probs.device)
                keep[probs[(b - 1) * n : b * n].sum(0).topk(S).indices] = True
                allowed[b * n : (b + 1) * n] = keep
        probs = probs.masked_fill(~allowed, 0.0)
    top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        top_v = top_v / top_v.sum(-1, keepdim=True).clamp_min(1e-9)
    return logits, top_v.to(logits.dtype), top_i


class SwitchLoRA(torch.nn.Module):
    """base(x) + B(A(x)), switched off while computing the teacher (so the teacher stays the original model)."""

    def __init__(self, base, r=16, alpha=32.0):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        if not STATE["teacher"]:
            y = y + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(y.dtype)
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--mode", choices=["router", "router+attn"], default="router")
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--train-seqs", type=int, default=3000)
    ap.add_argument("--val-seqs", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = 1e-3 routers / 5e-4 attention LoRA")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)

    ttok = AutoTokenizer.from_pretrained(args.text_tokenizer)
    tok = AutoTokenizer.from_pretrained(args.model)
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
        if len(seqs) >= args.train_seqs + args.val_seqs:
            break
    val, train = seqs[: args.val_seqs], seqs[args.val_seqs :]
    print(f"{len(train)} train / {len(val)} val sequences", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.requires_grad_(False)
    routers = [m for m in model.modules() if type(m).__name__.endswith("TopKRouter")]
    for r in routers:
        r.register_buffer("orig_weight", r.weight.detach().clone())
        r.forward = types.MethodType(routed_forward, r)
        r.weight.data = r.weight.data.float()
        r.weight.requires_grad_(True)
    groups = [{"params": [r.weight for r in routers], "lr": args.lr or 1e-3}]
    if args.mode == "router+attn":                              # LoRA on attention, off for the teacher
        loras = []
        for layer in model.model.layers:
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lo = SwitchLoRA(getattr(layer.self_attn, nm))
                setattr(layer.self_attn, nm, lo)
                loras.append(lo)
        groups.append({"params": [p for lo in loras for p in (lo.A, lo.B)], "lr": args.lr or 5e-4})
    n_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"[{args.mode}] {len(routers)} routers, {n_params / 1e6:.1f} M trainable params, S={args.S} n={args.n}",
          flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    def teacher_lp(ids):
        STATE.update(teacher=True)
        with torch.no_grad():
            lp = F.log_softmax(model(ids[None]).logits[0].float(), -1)
        STATE.update(teacher=False)
        return lp

    history = []

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        STATE.update(S=args.S, n=args.n)
        kl = ag = nc = nr = cnt = 0.0
        for s in val:
            s = s.cuda()
            ref = teacher_lp(s)
            lp = F.log_softmax(model(s[None]).logits[0].float(), -1)
            m = slice(args.n, len(s) - 1)
            tgt = s[args.n + 1 :]
            kl += float((ref[m].exp() * (ref[m] - lp[m])).sum())
            ag += float((ref[m].argmax(-1) == lp[m].argmax(-1)).sum())
            nc -= float(lp[m].gather(-1, tgt[:, None]).sum())
            nr -= float(ref[m].gather(-1, tgt[:, None]).sum())
            cnt += len(tgt)
        rec = {"step": step, "KL": kl / cnt, "agree": ag / cnt, "ppl": math.exp(nc / cnt), "ppl_orig": math.exp(nr / cnt)}
        history.append(rec)
        print(f"[eval step {step}] KL {rec['KL']:.4f} agree {rec['agree']:.4f} ppl {rec['ppl']:.3f} "
              f"(original {rec['ppl_orig']:.3f})", flush=True)
        model.train()

    evaluate(0)
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for s in random.sample(train, args.batch):
            s = s.cuda()
            ref = teacher_lp(s)
            STATE.update(S=args.S, n=args.n)
            lp = F.log_softmax(model(s[None]).logits[0].float(), -1)
            m = slice(args.n, len(s) - 1)
            loss = (ref[m].exp() * (ref[m] - lp[m])).sum(-1).mean()
            (loss / args.batch).backward()
            tot += float(loss.detach())
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            print(f"[train {step}/{args.steps}] KL {tot / args.batch:.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    torch.save({"routers": [r.weight.detach().cpu() for r in routers]}, out / "routers.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
