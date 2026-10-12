"""Experiment 49: can a small adapter teach an MoE causal block-shared routing? (training pilot, one GPU)

exp47: forcing block-shared expert sets on an MoE never trained for it hurts quality badly (OLMoE, n=16:
S=16 -> perplexity 9.2 vs 3.2; S=32 -> 4.5). With block-shared routing, verifying any part of a block
costs at most S experts per layer (OLMoE n=16: natural union 38 experts -> S=16 halves verification).

Here the model is fine-tuned to work under the constraint, with distillation to the ORIGINAL model
(teacher = same weights with the original routers and free routing, so no second copy in memory):
  router       only the router weights train (2M params for OLMoE): the "locality adapter"
  router+attn  routers + LoRA on the attention projections (hidden states can adapt to the restricted
               experts; the LoRA is switched off for the teacher pass)
  router+experts         routers + one low-rank adapter per expert (gate_up and down; OLMoE r=8: ~59M params):
               each expert can widen what it covers, so a block's S experts can serve tokens that would
               have gone elsewhere (OLMoE-style experts: 3D gate_up_proj / down_proj tensors)
  router+attn+experts    all three
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp47_blockroute import weights_per_token  # noqa: E402

STATE = {"S": None, "n": None, "teacher": False, "rec": None, "route": "block", "k": None, "kin": 0}


def window_unions(rec, E, n, win=None):
    """rec: per-layer bool masks [T, E] of the experts each token uses (nonzero weight). Mean over layers of the distinct experts in n-token windows that start
    inside the constrained region (t >= n): (sum over aligned windows, count, sum over all windows, count)."""
    ua = na = uu = nu = 0.0
    for idx in rec:
        T = idx.shape[0]
        if T < n + (win or n):
            continue
        m = idx.float()                                                       # [T, E]
        cs = torch.cat([torch.zeros(1, E, device=m.device), m.cumsum(0)])
        win = win or n
        st = torch.arange(n, T - win + 1, device=m.device)
        u = ((cs[st + win] - cs[st]) > 0).sum(-1).float()                     # [windows]
        al = (st % n) <= n - win                                              # window inside one routing block
        ua += float(u[al].sum()); na += float(al.sum())
        uu += float(u.sum()); nu += len(u)
    L = len(rec)
    return ua / L, na / L, uu / L, nu / L


def routed_forward(self, hidden_states):
    h = hidden_states.reshape(-1, self.hidden_dim)
    W = self.orig_weight if STATE["teacher"] else self.weight
    logits = F.linear(h, W.to(h.dtype))
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    T, E = probs.shape
    if STATE["S"] is not None and not STATE["teacher"]:
        n, S, route = STATE["n"], STATE["S"], STATE["route"]
        allowed = torch.ones_like(probs, dtype=torch.bool)
        with torch.no_grad():
            if route == "block":                    # causal block-shared: block b uses the top-S of block b-1
                for b in range(1, (T + n - 1) // n):
                    keep = torch.zeros(E, dtype=torch.bool, device=probs.device)
                    keep[probs[(b - 1) * n : b * n].sum(0).topk(S).indices] = True
                    allowed[b * n : (b + 1) * n] = keep
            elif route == "cur":                    # upper bound for drafter-informed sets: block b uses its OWN top-S
                for b in range(1, (T + n - 1) // n):     # (not causal; inputs are true text = a perfect draft)
                    keep = torch.zeros(E, dtype=torch.bool, device=probs.device)
                    keep[probs[b * n : (b + 1) * n].sum(0).topk(S).indices] = True
                    allowed[b * n : (b + 1) * n] = keep
            elif route == "fixed":                  # exp51: sets given per layer and block in STATE["fixed"][layer]
                sets = STATE["fixed"].setdefault(self.layer_idx, {})
                for b in range(1, (T + n - 1) // n):     # a block with no stored set takes its own top-S and stores it
                    if b not in sets:
                        keep = torch.zeros(E, dtype=torch.bool, device=probs.device)
                        keep[probs[b * n : (b + 1) * n].sum(0).topk(S).indices] = True
                        sets[b] = keep
                    allowed[b * n : (b + 1) * n] = sets[b]
            elif route == "seq":                    # baseline: one set for the whole sequence, from block 0
                keep = torch.zeros(E, dtype=torch.bool, device=probs.device)
                keep[probs[:n].sum(0).topk(S).indices] = True
                allowed[n:] = keep
            elif route == "topk":                   # baseline: free routing with fewer experts per token
                kk = torch.zeros_like(allowed)
                kk.scatter_(1, probs.topk(STATE["k"], -1).indices, True)
                allowed[n:] = kk[n:]
            if STATE["kin"]:                        # fewer experts per token inside the allowed set
                kk = torch.zeros_like(allowed)
                kk.scatter_(1, probs.masked_fill(~allowed, 0.0).topk(STATE["kin"], -1).indices, True)
                allowed[n:] &= kk[n:]
        probs = probs.masked_fill(~allowed, 0.0)
    top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
    if STATE["rec"] is not None:
        STATE["rec"].append(torch.zeros_like(probs, dtype=torch.bool).scatter_(1, top_i, top_v > 0))
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


def lora_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """OlmoeExperts.forward (transformers 5.x, eager loop) + per-expert LoRA on gate_up and down, off for the teacher."""
    final = torch.zeros_like(hidden_states)
    on = not STATE["teacher"]
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for e in expert_hit:
        e = e[0]
        if e == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e])
        x = hidden_states[token_idx]
        gu = F.linear(x, self.gate_up_proj[e])
        if on:
            gu = gu + ((x.float() @ self.lA_gu[e].T) @ self.lB_gu[e].T * self.lscale).to(gu.dtype)
        gate, up = gu.chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        y = F.linear(h, self.down_proj[e])
        if on:
            y = y + ((h.float() @ self.lA_d[e].T) @ self.lB_d[e].T * self.lscale).to(y.dtype)
        y = y * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, y.to(final.dtype))
    return final


def lora_experts_forward_grouped(self, hidden_states, top_k_index, top_k_weights):
    """Same as lora_experts_forward, vectorised: tokens sorted by expert, one grouped matmul per projection for the
    base weights and for each LoRA factor (transformers' grouped_mm path), instead of a Python loop over experts."""
    from transformers.integrations.moe import _grouped_linear
    k = top_k_index.size(-1)
    T, H = hidden_states.shape
    eid, perm = torch.sort(top_k_index.reshape(-1))
    x = hidden_states[perm // k]
    w = top_k_weights.reshape(-1)[perm]
    hin = eid.float() if hidden_states.device.type == "cpu" else eid.int()
    offs = torch.cumsum(torch.histc(hin, bins=self.num_experts, min=0, max=self.num_experts - 1), 0, dtype=torch.int32)
    on = not STATE["teacher"]
    dt = self.gate_up_proj.dtype

    def lora(inp, A, B):
        return _grouped_linear(_grouped_linear(inp.to(dt), A.to(dt), offs), B.to(dt), offs) * self.lscale

    gu = _grouped_linear(x, self.gate_up_proj, offs)
    if on:
        gu = gu + lora(x, self.lA_gu, self.lB_gu)
    gate, up = gu.chunk(2, dim=-1)
    h = self.act_fn(gate) * up
    y = _grouped_linear(h, self.down_proj, offs)
    if on:
        y = y + lora(h, self.lA_d, self.lB_d)
    y = y * w.unsqueeze(-1).to(y.dtype)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return y[inv].view(T, k, H).float().sum(1).to(hidden_states.dtype)


def add_expert_lora(model, r=8, alpha=16.0, impl="grouped"):
    mods = [m for m in model.modules() if hasattr(m, "gate_up_proj") and hasattr(m, "down_proj")
            and m.gate_up_proj.dim() == 3]
    params = []
    for m in mods:
        E, I2, H = m.gate_up_proj.shape
        I = m.down_proj.shape[2]
        dev = m.gate_up_proj.device
        m.lA_gu = torch.nn.Parameter(torch.randn(E, r, H, device=dev) / math.sqrt(H))
        m.lB_gu = torch.nn.Parameter(torch.zeros(E, I2, r, device=dev))
        m.lA_d = torch.nn.Parameter(torch.randn(E, r, I, device=dev) / math.sqrt(I))
        m.lB_d = torch.nn.Parameter(torch.zeros(E, H, r, device=dev))
        m.lscale = alpha / r
        m.forward = types.MethodType(lora_experts_forward_grouped if impl == "grouped" else lora_experts_forward, m)
        params += [m.lA_gu, m.lB_gu, m.lA_d, m.lB_d]
    return mods, params


def add_adapters(model, mode, lr=0.0, rank=8, impl="grouped"):
    """Patch routers (block-shared, trainable) and add the LoRAs `mode` asks for. Returns (routers, groups, extra)."""
    routers = [m for m in model.modules() if type(m).__name__.endswith("TopKRouter")]
    for i, r in enumerate(routers):
        r.layer_idx = i
        r.register_buffer("orig_weight", r.weight.detach().clone())
        r.forward = types.MethodType(routed_forward, r)
        r.weight.data = r.weight.data.float()
        r.weight.requires_grad_(True)
    groups = [{"params": [r.weight for r in routers], "lr": lr or 1e-3}]
    extra = {}
    if "attn" in mode:                                          # LoRA on attention, off for the teacher
        loras = []
        for layer in model.model.layers:
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lo = SwitchLoRA(getattr(layer.self_attn, nm))
                setattr(layer.self_attn, nm, lo)
                loras.append(lo)
        p = [p for lo in loras for p in (lo.A, lo.B)]
        groups.append({"params": p, "lr": lr or 5e-4})
        extra["attn"] = p
    if "experts" in mode:                                       # one LoRA per expert, off for the teacher
        mods, p = add_expert_lora(model, r=rank, impl=impl)
        assert mods, "no 3D expert tensors found (expects OLMoE-style experts)"
        groups.append({"params": p, "lr": lr or 5e-4})
        extra["experts"] = p
    return routers, groups, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--mode", choices=["router", "router+attn", "router+experts", "router+attn+experts"], default="router")
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
    ap.add_argument("--route", choices=["block", "cur", "seq", "topk"], default="block",
                    help="block = causal block-shared sets (ours); cur = a block's own top-S (upper bound); seq = one set per sequence; topk = fewer experts/token")
    ap.add_argument("--k", type=int, default=2, help="experts per token for --route topk")
    ap.add_argument("--kin", type=int, default=0, help="experts per token inside the block set (0 = model top-k)")
    ap.add_argument("--win", type=int, default=16, help="verification window for the expert-union measurement")
    ap.add_argument("--expert-rank", type=int, default=8)
    ap.add_argument("--expert-impl", choices=["grouped", "loop"], default="grouped")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    STATE.update(route=args.route, k=args.k, kin=args.kin)

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
        if len(ids) > 2 * args.win:
            seqs.append(ids)
        if len(seqs) >= args.train_seqs + args.val_seqs:
            break
    val, train = seqs[: args.val_seqs], seqs[args.val_seqs :]
    print(f"{len(train)} train / {len(val)} val sequences", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.requires_grad_(False)
    routers, groups, extra = add_adapters(model, args.mode, args.lr, args.expert_rank, args.expert_impl)
    if "experts" in args.mode and args.expert_impl == "grouped":     # check the grouped kernels run here, else loop
        try:
            STATE.update(S=args.S, n=args.n)
            model(train[0][: 4 * args.n].cuda()[None]).logits.float().mean().backward()
            model.zero_grad(set_to_none=True)
            print("expert LoRA: grouped_mm path", flush=True)
        except RuntimeError as e:
            print(f"expert LoRA: grouped path failed ({str(e)[:120]}), using the loop", flush=True)
            for m in model.modules():
                if hasattr(m, "lA_gu"):
                    m.forward = types.MethodType(lora_experts_forward, m)
    n_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"[{args.mode}] {len(routers)} routers, {n_params / 1e6:.1f} M trainable params, route={args.route} S={args.S} k={args.k} kin={args.kin} n={args.n}",
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
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    E, k = cfg.num_experts, cfg.num_experts_per_tok

    def save():
        (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
        torch.save({"routers": [r.weight.detach().cpu() for r in routers],
                    **{kk: [p.detach().cpu() for p in v] for kk, v in extra.items()}}, out / "routers.pt")

    def cost(u):
        return weights_per_token(cfg, u) / weights_per_token(cfg, k)

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        STATE.update(S=args.S, n=args.n)
        kl = ag = nc = nr = cnt = 0.0
        un = {"free": [0.0] * 4, "block": [0.0] * 4}
        for s in val:
            s = s.cuda()
            STATE["rec"] = []
            ref = teacher_lp(s)
            for i, v in enumerate(window_unions(STATE["rec"], E, args.n, args.win)):
                un["free"][i] += v
            STATE["rec"] = []
            lp = F.log_softmax(model(s[None]).logits[0].float(), -1)
            for i, v in enumerate(window_unions(STATE["rec"], E, args.n, args.win)):
                un["block"][i] += v
            STATE["rec"] = None
            m = slice(args.n, len(s) - 1)
            tgt = s[args.n + 1 :]
            kl += float((ref[m].exp() * (ref[m] - lp[m])).sum())
            ag += float((ref[m].argmax(-1) == lp[m].argmax(-1)).sum())
            nc -= float(lp[m].gather(-1, tgt[:, None]).sum())
            nr -= float(ref[m].gather(-1, tgt[:, None]).sum())
            cnt += len(tgt)
        rec = {"step": step, "KL": kl / cnt, "agree": ag / cnt, "ppl": math.exp(nc / cnt), "ppl_orig": math.exp(nr / cnt)}
        for nm, (ua, na, uu, nu) in un.items():
            rec[f"union_{nm}_aligned"] = ua / max(na, 1)
            rec[f"union_{nm}_any"] = uu / max(nu, 1)
        history.append(rec)
        print(f"[eval step {step}] KL {rec['KL']:.4f} agree {rec['agree']:.4f} ppl {rec['ppl']:.3f} "
              f"(original {rec['ppl_orig']:.3f}, ratio {rec['ppl'] / rec['ppl_orig']:.3f})", flush=True)
        print(f"[eval step {step}] experts/layer in {args.win}-token windows (cost vs 1 token): "
              f"block-routed aligned {rec['union_block_aligned']:.1f} ({cost(rec['union_block_aligned']):.2f}x), "
              f"any offset {rec['union_block_any']:.1f} ({cost(rec['union_block_any']):.2f}x); "
              f"free routing {rec['union_free_any']:.1f} ({cost(rec['union_free_any']):.2f}x)", flush=True)
        save()
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
    print("done", flush=True)


if __name__ == "__main__":
    main()
