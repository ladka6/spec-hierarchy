"""Experiment 81: target-state estimator built from the target's own tap layers, DFlash and target frozen.

Same problem and evaluation as exp80 (estimate the verifier's L25 / L33 for the in-flight tokens of a chained round so
that the unmodified DFlash drafts block 2 as with real features). The estimator here starts from the target's
knowledge instead of random weights:

  body     the target's layers 1, 9, 17, 25, 33 (exactly the layers DFlash reads), applied in sequence to the in-flight
           tokens; the skipped layers are replaced by small zero-initialised bridges (residual MLPs) before each layer
  context  every kept layer attends, as in the target, to its OWN real KV cache over the verified context (positions
           < s) plus the in-flight tokens (causal). k/v projections and k-norm stay frozen, so the new tokens' keys
           and values are produced exactly like the cached ones.
  trained  LoRA (rank r) on q / o / gate / up / down of the kept layers, the bridges, and a zero-initialised low-rank
           lookahead input (embedding of x_{p+1}, the target's own next token if the prefix is accepted)
  outputs  L25 = residual stream after layer 25, L33 = after layer 33; shallow taps at their dataset mean
At initialisation the estimator is the plain "5 real layers, others skipped" stack.

Loss: Huber on standardised L25 / L33 + kl_weight * KL of the frozen DFlash's block 2 (anchor s+n), as exp80.

  python scripts/exp81_tap_layers.py --cache seqs.pt --data .../data --out est81.pt
  python scripts/exp81_tap_layers.py --cache seqs.pt --load est81.pt --steps 0
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
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

LORA_TARGETS = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def cache_kv(pkv, i):
    """(keys, values) [1, n_kv, T, hd] of layer i from a transformers cache (several versions)."""
    if hasattr(pkv, "layers"):
        return pkv.layers[i].keys, pkv.layers[i].values
    if hasattr(pkv, "key_cache"):
        return pkv.key_cache[i], pkv.value_cache[i]
    return pkv[i][0], pkv[i][1]


class LoRA(nn.Module):
    def __init__(self, d_in, d_out, r, alpha):
        super().__init__()
        self.a = nn.Parameter(torch.randn(r, d_in) / math.sqrt(d_in))
        self.b = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r

    def forward(self, x):
        return (x @ self.a.t().to(x.dtype)) @ self.b.t().to(x.dtype) * self.scale


class TapEstimator(nn.Module):
    """Trainable parts only; the kept layers are the target's (frozen, shared, not stored)."""

    def __init__(self, target, layer_ids, rank=64, bridge=1024, look_rank=256):
        super().__init__()
        cfg = target.config
        self.layer_ids = list(layer_ids)
        H = cfg.hidden_size
        self.cfg = dict(layer_ids=self.layer_ids, rank=rank, bridge=bridge, look_rank=look_rank)
        self.lora = nn.ModuleDict()
        for li in self.layer_ids:
            layer = target.model.layers[li]
            for name in LORA_TARGETS:
                mod = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name)
                self.lora[f"{li}_{name}"] = LoRA(mod.in_features, mod.out_features, rank, 2 * rank)
        self.bridges = nn.ModuleList()
        for _ in self.layer_ids:
            br = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, bridge), nn.GELU(), nn.Linear(bridge, H))
            nn.init.zeros_(br[-1].weight)
            nn.init.zeros_(br[-1].bias)
            self.bridges.append(br)
        self.look_a = nn.Linear(H, look_rank, bias=False)
        self.look_b = nn.Linear(look_rank, H, bias=False)
        nn.init.zeros_(self.look_b.weight)
        self.register_buffer("mu", torch.zeros(5, H))
        self.register_buffer("sd", torch.ones(5, H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--span", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--anchors-train", type=int, default=24)
    ap.add_argument("--kl-anchors", type=int, default=4)
    ap.add_argument("--kl-weight", type=float, default=0.3)
    ap.add_argument("--zclip", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--stats-seqs", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--eval-stops", type=int, default=150)
    ap.add_argument("--stops", type=int, default=600)
    ap.add_argument("--load", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--save", default="exp81_tap_layers")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    rng = random.Random(0)
    torch.manual_seed(0)

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    val = seqs[: max(int(len(seqs) * 0.2), 1)]
    train = []
    if args.data:
        for f in sorted(Path(args.data).glob("*.pt")):
            train += [(r["ids"].long(), int(r["n_prompt"])) for r in torch.load(f)]
    else:
        train = seqs[len(val) :]
    train = [(ids, lo) for ids, lo in train if len(ids) <= args.max_len and len(ids) - lo > 3 * 16 + 8]
    rng.shuffle(train)
    print(f"[data] {len(train)} train / {len(val)} eval sequences", flush=True)

    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H, K = target.config.hidden_size, len(lids)
    i25, i33 = lids.index(25), lids.index(33)
    S = args.span
    layers = target.model.layers
    n_h, n_kv = target.config.num_attention_heads, target.config.num_key_value_heads
    emb = target.model.embed_tokens

    @torch.no_grad()
    def target_pass(x):
        out = target(x[None], output_hidden_states=True, use_cache=True, logits_to_keep=1)
        f = extract_context_feature(out.hidden_states, lids)[0]                       # [T, K*H]
        kv = {li: cache_kv(out.past_key_values, li) for li in lids}
        return f, kv

    if args.load:
        d = torch.load(args.load, map_location="cpu")
        est = TapEstimator(target, **d["cfg"]).to(dev)
        est.load_state_dict(d["state"])
    else:
        est = TapEstimator(target, lids, args.rank).to(dev)
        acc_s = torch.zeros(K, H, dtype=torch.float64, device=dev)
        acc_q = torch.zeros(K, H, dtype=torch.float64, device=dev)
        n_pos = 0
        for ids, lo in train[: args.stats_seqs]:
            f, _ = target_pass(ids.to(dev))
            f = f[lo:].double().view(-1, K, H)
            acc_s += f.sum(0)
            acc_q += f.pow(2).sum(0)
            n_pos += f.shape[0]
        mu = acc_s / n_pos
        est.mu.copy_(mu.float())
        est.sd.copy_((acc_q / n_pos - mu * mu).clamp_min(1e-8).sqrt().float().clamp_min(1e-4))
    print(f"[estimator] layers {est.layer_ids}, {sum(p.numel() for p in est.parameters()) / 1e6:.1f}M trainable "
          f"parameters", flush=True)

    def lin(mod, key, x):
        return F.linear(x, mod.weight, mod.bias) + est.lora[key](x)

    def run_layer(li, h, pos, kv, anchors):
        """one kept target layer on the in-flight tokens h [A, n, H], attending to its real cache (< s) + causal."""
        layer = layers[li]
        at = layer.self_attn
        A, n, _ = h.shape
        hd = at.head_dim
        x = layer.input_layernorm(h)
        q = at.q_norm(lin(at.q_proj, f"{li}_q_proj", x).view(A, n, n_h, hd)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(A, n, n_kv, hd)).transpose(1, 2)
        v = at.v_proj(x).view(A, n, n_kv, hd).transpose(1, 2)
        cos, sin = target.model.rotary_emb(v, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        smax = max(anchors)
        ck, cv = kv[li]
        ck = ck[:, :, :smax].expand(A, -1, -1, -1).to(k.dtype)
        cv = cv[:, :, :smax].expand(A, -1, -1, -1).to(v.dtype)
        keys, vals = torch.cat([ck, k], 2), torch.cat([cv, v], 2)
        st = torch.tensor(anchors, device=dev)
        ctx_ok = torch.arange(smax, device=dev)[None] < st[:, None]                    # [A, smax]
        causal = torch.ones(n, n, dtype=torch.bool, device=dev).tril()
        mask = torch.cat([ctx_ok[:, None, :].expand(A, n, smax), causal[None].expand(A, n, n)], 2)[:, None]
        rep = n_h // n_kv
        keys, vals = keys.repeat_interleave(rep, 1), vals.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask)
        h = h + lin(at.o_proj, f"{li}_o_proj", o.transpose(1, 2).reshape(A, n, n_h * hd))
        x = layer.post_attention_layernorm(h)
        mlp = layer.mlp
        g = lin(mlp.gate_proj, f"{li}_gate_proj", x)
        u = lin(mlp.up_proj, f"{li}_up_proj", x)
        return h + lin(mlp.down_proj, f"{li}_down_proj", mlp.act_fn(g) * u)

    def estimate(kv, x, anchors, n):
        """raw (h25, h33) [A, n, H] for positions s .. s+n-1 of each anchor s"""
        toks = torch.stack([x[s : s + n] for s in anchors])
        nxt = torch.stack([x[s + 1 : s + n + 1] for s in anchors])
        pos = torch.stack([torch.arange(s, s + n, device=dev) for s in anchors])
        out = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = emb(toks) + est.look_b(est.look_a(emb(nxt)))
            for k_i, li in enumerate(est.layer_ids):
                h = h + est.bridges[k_i](h)
                h = run_layer(li, h, pos, kv, anchors)
                out[li] = h
        return out[25].float(), out[33].float()

    def zs(h, t):
        z = (h - est.mu[t]) / est.sd[t]
        return z.clamp(-args.zclip, args.zclip) if args.zclip > 0 else z

    def raw_part(h25, h33):
        n = h25.shape[0]
        out = est.mu[None].expand(n, K, H).clone()
        out[:, i25], out[:, i33] = h25, h33
        return out.view(n, K * H)

    def stop_eval(n_anchors, tag):
        est.eval()
        vr = random.Random(7)
        cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 8, len(ids) - 3 * bs)]
        vr.shuffle(cand)
        by = {}
        for i, s in cand[:n_anchors]:
            by.setdefault(i, []).append(s)
        recs = []
        with torch.inference_mode():
            for i, anchors in by.items():
                ids, lo = val[i]
                x = ids.to(dev)
                f, kv = target_pass(x)
                for s in anchors:
                    y1 = blocks_logits(draft, target, f[None, :s], x, [(s, 0)], bs)[0].argmax(-1).tolist()
                    n = matched(y1, x[s + 1 :].tolist())
                    r = {"n": n, "real": 0, "tokens": 0, "est": 0}
                    if n > 0:
                        a = s + n
                        truth = x[a + 1 :].tolist()
                        lg = blocks_logits(draft, target, f[None, :a], x, [(a, 0)], bs)[0]
                        r["real"] = matched(lg.argmax(-1).tolist(), truth)
                        lg = blocks_logits(draft, target, f[None, :s], x, [(a, n)], bs)[0]
                        r["tokens"] = matched(lg.argmax(-1).tolist(), truth)
                        h25, h33 = estimate(kv, x, [s], n)
                        fz = torch.cat([f[:s], raw_part(h25[0], h33[0]).to(f.dtype)], 0)[None]
                        lg = blocks_logits(draft, target, fz, x, [(a, 0)], bs)[0]
                        r["est"] = matched(lg.argmax(-1).tolist(), truth)
                    recs.append(r)
        est.train()
        k = len(recs)
        normal = sum(r["n"] + 1 for r in recs) / k
        rows = [{"block 2 from": "none (normal DFlash)", "first token right": "-", "mean m2": 0.0,
                 "tokens / verification": normal, "vs normal": 1.0}]
        for c in ("real", "tokens", "est"):
            t = sum(r["n"] + r[c] + 1 for r in recs) / k
            rows.append({"block 2 from": c, "first token right": sum(r[c] >= 1 for r in recs if r["n"] > 0)
                         / max(sum(r["n"] > 0 for r in recs), 1), "mean m2": sum(r[c] for r in recs) / k,
                         "tokens / verification": t, "vs normal": t / normal})
        print_table(rows, list(rows[0]), f"[{tag}] chained round, all {k} rounds, oracle cut, drafter {args.draft}")
        return rows

    hist = []
    if args.steps > 0:
        hist.append({"step": 0, "rows": stop_eval(args.eval_stops, "step 0 (5 real layers, others skipped)")})
        opt = torch.optim.AdamW(est.parameters(), lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 200) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(s, args.steps)
                                                                                   / args.steps))))
        it, step, tl = 0, 0, time.time()
        run = {"reg": 0.0, "kl": 0.0, "z25": 0.0, "z33": 0.0, "n": 0}
        while step < args.steps:
            opt.zero_grad(set_to_none=True)
            for _ in range(args.batch):
                ids, lo = train[it % len(train)]
                it += 1
                x = ids.to(dev)
                T = len(ids)
                n = rng.randint(1, S)
                hi = T - n - bs - 2
                if hi <= lo + 1:
                    continue
                anc = list(range(lo + 1 + rng.randrange(S), hi, S))
                if len(anc) > args.anchors_train:
                    anc = sorted(rng.sample(anc, args.anchors_train))
                if not anc:
                    continue
                f, kv = target_pass(x)
                h25, h33 = estimate(kv, x, anc, n)
                fr = f.float().view(-1, K, H)
                r25 = torch.stack([zs(fr[s : s + n, i25], i25) for s in anc])
                r33 = torch.stack([zs(fr[s : s + n, i33], i33) for s in anc])
                y25, y33 = zs(h25, i25), zs(h33, i33)
                reg = F.smooth_l1_loss(y25, r25) + F.smooth_l1_loss(y33, r33)
                kl_tot = torch.zeros((), device=dev)
                if args.kl_weight > 0:
                    kl_idx = rng.sample(range(len(anc)), min(args.kl_anchors, len(anc)))
                    for a_i in kl_idx:
                        s = anc[a_i]
                        a = s + n
                        mix = torch.cat([f[:s], raw_part(h25[a_i], h33[a_i]).to(f.dtype)], 0)[None]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lg = blocks_logits(draft, target, mix, x, [(a, 0)], bs).float()
                        with torch.no_grad():
                            ref = blocks_logits(draft, target, f[None, :a], x, [(a, 0)], bs).float()
                        kl_tot = kl_tot + F.kl_div(torch.log_softmax(lg, -1), torch.log_softmax(ref, -1),
                                                   log_target=True, reduction="none").sum(-1).mean() / len(kl_idx)
                loss = reg + args.kl_weight * kl_tot
                (loss / args.batch).backward()
                run["reg"] += float(reg.detach())
                run["kl"] += float(kl_tot.detach())
                run["z25"] += float(F.cosine_similarity(y25.detach(), r25, dim=-1).mean())
                run["z33"] += float(F.cosine_similarity(y33.detach(), r33, dim=-1).mean())
                run["n"] += 1
                del kv
            torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 and run["n"]:
                m = run["n"]
                print(f"[step {step}/{args.steps}] huber {run['reg'] / m:.4f} kl {run['kl'] / m:.4f} "
                      f"zcos25 {run['z25'] / m:.3f} zcos33 {run['z33'] / m:.3f}  {(time.time() - tl) / step:.2f}s/step",
                      flush=True)
                run = {k_: 0.0 for k_ in run}
            if step % args.eval_every == 0 or step == args.steps:
                if args.out:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state": est.state_dict(), "cfg": est.cfg}, args.out)
                if step < args.steps:
                    hist.append({"step": step, "rows": stop_eval(args.eval_stops, f"step {step}")})

    rows = stop_eval(args.stops, "final")
    save_json(args.save + f"_{Path(args.draft).name}", {"args": vars(args), "history": hist, "final": rows})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
