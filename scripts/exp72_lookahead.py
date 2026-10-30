"""Experiment 72: free lookahead from the verify pass (offline, Colab-ready)

Idea: append k placeholder ("future") positions after the tokens the target verifies. Real tokens never attend to them
(they come last, causal mask), so verification is unchanged and lossless. The target's states at the placeholders are
its own guess of its states at the next positions, at almost no extra verify cost. A chained second drafting stage
(DLoop-style) needs exactly these: features for positions the target has not verified yet.

Two levels (same script):
  default   target FROZEN, only k placeholder input embeddings are learned (prompt-tuning style)
  --lora R  plus LoRA adapters (rank R) on every attention / MLP projection that are ACTIVE ONLY AT PLACEHOLDER
            POSITIONS (gate mask = 0 on real tokens), so the target's outputs on real text stay bit-identical
            (checked and printed: max |logit difference| on real positions must be 0)
Loss at the placeholders: per-tap (1 - cosine) + normalized MSE ||f - r||^2 / ||r||^2 (also matches the norm, so the
features look like real ones to DFlash), weighted toward the deep taps, + cross-entropy of the target's LM head on the
true tokens. A small gate head on each placeholder's final state predicts whether its LM-head guess is right (BCE).

Note on offsets: placeholder j (1-indexed) sits at position p + j - 1. The token AT position p is the one the target
produces in this same verify pass, so the placeholders never see it; placeholder j's LM head predicts x[p + j].

Evaluation on held-out sequences:
  1. information, no drafter involved (per offset j, m = max m):
       cosine of the placeholder state to the real state at the two deepest taps
       top-1 accuracy of the target's LM head at placeholder j on x[p + j]
       DFlash's top-1 accuracy on the same token from anchor p with real features (it even knows x[p])
       gate AUROC for flagging wrong placeholder guesses
  2. tau, same protocol as exp71: anchor s, real context features up to p = s - m, and for the last m positions
       real / removed / la_trained (placeholder states) / la_repeat (untrained: last token repeated)
     exp71 reference: real 8.44; removed 3.60 / 2.18 / 1.60; copy 8.42 / 8.38 / 8.30 for m = 4 / 8 / 15.
     DFlash was never trained on placeholder features, so tau here is a lower bound; part 1 is the shift-free signal.

  python scripts/exp72_lookahead.py --cache /content/seqs.pt                 # placeholders only
  python scripts/exp72_lookahead.py --cache /content/seqs.pt --lora 32       # + gated LoRA
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

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

GATE = {"mask": None}          # [1, T, 1] float: 1 at placeholder positions; None = adapters off


class GatedLoRA(nn.Module):
    """base(x) + mask * B(A(x)) * scale; exactly base(x) wherever mask = 0 (and everywhere when mask is None)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        m = GATE["mask"]
        if m is None:
            return y
        d = F.linear(F.linear(x.float(), self.A), self.B) * self.scale
        return y + (d * m).to(y.dtype)


def add_gated_lora(model, r, alpha):
    params = []
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for n in names:
                mod = GatedLoRA(getattr(parent, n), r, alpha)
                setattr(parent, n, mod)
                params += [mod.A, mod.B]
    return params


def auroc(score, label):
    """P(score of a positive > score of a negative)."""
    order = torch.argsort(score)
    ranks = torch.empty(len(score), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences, e.g. from exp69")
    ap.add_argument("--k", type=int, default=16, help="placeholders learned (eval uses the first m)")
    ap.add_argument("--m", nargs="+", type=int, default=[4, 8, 15])
    ap.add_argument("--lora", type=int, default=0, help="rank of placeholder-only LoRA (0 = placeholders only)")
    ap.add_argument("--lora-alpha", type=float, default=0.0, help="default = 2 * rank")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--per-step", type=int, default=4, help="prefixes per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-3, help="placeholder embeddings")
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--tap-w", nargs="+", type=float, default=[0.25, 0.25, 0.25, 1.0, 1.0])
    ap.add_argument("--nmse-w", type=float, default=1.0)
    ap.add_argument("--ce-w", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--anchors", type=int, default=600, help="eval anchors per m")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    args.save = args.save or ("exp72_lookahead" + (f"_lora{args.lora}" if args.lora else ""))
    dev = "cuda"
    t0 = time.time()
    torch.manual_seed(0)

    seqs = [(ids, lo) for ids, lo in torch.load(args.cache) if len(ids) - lo >= args.k + 40]
    random.Random(0).shuffle(seqs)
    nv = int(len(seqs) * args.val_frac)
    val, train = seqs[:nv], seqs[nv:]
    print(f"[data] {len(train)} train / {len(val)} val sequences", flush=True)

    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p_ in target.parameters():
        p_.requires_grad_(False)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    nT, H, K = len(lids), target.config.hidden_size, args.k
    emb = target.get_input_embeddings()
    tap_w = torch.tensor(args.tap_w, device=dev)

    with torch.no_grad():
        mu = emb.weight.float().mean(0)
        sd = emb.weight.float().std()
    P = nn.Parameter(mu[None].repeat(K, 1) + 0.02 * sd * torch.randn(K, H, device=dev))
    gate_head = nn.Linear(H, 1).to(dev)
    groups = [{"params": [P], "lr": args.lr}, {"params": list(gate_head.parameters()), "lr": 1e-3}]
    lora_params = []
    if args.lora:
        lora_params = add_gated_lora(target, args.lora, args.lora_alpha or 2.0 * args.lora)
        groups.append({"params": lora_params, "lr": args.lora_lr})
        print(f"[lora] rank {args.lora}: {sum(p.numel() for p in lora_params)/1e6:.1f}M params, placeholder-only",
              flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))

    def lookahead(x, p, ph, keep_all=False, adapters=True):
        """Target pass over real tokens x[:p] followed by placeholder embeddings ph [k, H] (adapters on at the
        placeholders only). Returns tap features [k, nT*H], LM logits [k, V], final states [k, H] at the placeholders
        (logits for all positions if keep_all)."""
        k = ph.shape[0]
        e = torch.cat([emb(x[None, :p]), ph[None].to(emb.weight.dtype)], 1)
        mask = torch.zeros(1, p + k, 1, device=dev)
        mask[:, p:] = 1.0
        GATE["mask"] = mask if (args.lora and adapters) else None
        out = target(inputs_embeds=e, output_hidden_states=True, use_cache=False, logits_to_keep=0 if keep_all else k)
        f = extract_context_feature(out.hidden_states, lids)[0, p:]
        return f, out.logits[0].float(), out.hidden_states[-1][0, p:]

    @torch.no_grad()
    def real_feats(x):
        GATE["mask"] = None
        out = target(x[None], output_hidden_states=True, use_cache=False)
        return extract_context_feature(out.hidden_states, lids)

    def tap_loss(f, r):
        f, r = f.float().view(-1, nT, H), r.float().view(-1, nT, H)
        cos = F.cosine_similarity(f, r, dim=-1)
        nmse = ((f - r) ** 2).sum(-1) / (r ** 2).sum(-1).clamp(min=1e-6)
        per = (1 - cos) + args.nmse_w * nmse
        return (per * tap_w).sum(-1).mean() / tap_w.sum(), cos

    # ---------------- train (target weights frozen) ----------------
    target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    target.train()                                    # checkpointing only runs in train mode; Qwen3 has no dropout
    rng = random.Random(1)
    log = []
    for step in range(args.steps):
        ids, lo = train[rng.randrange(len(train))]
        x = ids.to(dev)
        rf = real_feats(x)[0]
        opt.zero_grad(set_to_none=True)
        tot = {"loss": 0.0, "cos_deep": 0.0, "acc": 0.0, "acc1-4": 0.0}
        for _ in range(args.per_step):
            p = rng.randrange(lo + 8, len(x) - K - 1)
            f, lg, hfin = lookahead(x, p, P)
            l_tap, cos = tap_loss(f, rf[p : p + K])
            tgt = x[p + 1 : p + K + 1]
            l_ce = F.cross_entropy(lg, tgt)
            right = (lg.argmax(-1) == tgt)
            l_gate = F.binary_cross_entropy_with_logits(gate_head(hfin.detach().float())[:, 0], right.float())
            loss = (l_tap + args.ce_w * l_ce + 0.1 * l_gate) / args.per_step
            loss.backward()
            tot["loss"] += float(loss.detach())
            tot["cos_deep"] += float(cos[:, -2:].mean()) / args.per_step
            tot["acc"] += float(right.float().mean()) / args.per_step
            tot["acc1-4"] += float(right[:4].float().mean()) / args.per_step
        torch.nn.utils.clip_grad_norm_([P] + lora_params, 1.0)
        opt.step()
        sched.step()
        log.append(tot)
        if step % 50 == 0 or step == args.steps - 1:
            w = log[-50:]
            avg = {k_: sum(r[k_] for r in w) / len(w) for k_ in tot}
            print(f"[train] step {step}: loss {avg['loss']:.4f}  cos(deep taps) {avg['cos_deep']:.3f}  "
                  f"lm-head acc all {avg['acc']:.3f} / offsets 1-4 {avg['acc1-4']:.3f}  {time.time()-t0:.0f}s",
                  flush=True)
    target.gradient_checkpointing_disable()
    target.eval()
    torch.save({"P": P.detach().cpu(), "gate": gate_head.state_dict(),
                "lora": [p.detach().cpu() for p in lora_params]}, f"{args.save}_weights.pt")

    # ---------------- evaluate on held-out sequences ----------------
    M = max(args.m)
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + M + 8, len(ids) - bs)]
    random.Random(0).shuffle(cand)
    by_seq = {}
    for i, s in sorted(cand[: args.anchors]):
        by_seq.setdefault(i, []).append(s)
    srcs = ("real", "removed", "la_trained", "la_repeat")
    tau = {(src, m): [] for src in srcs for m in args.m}
    cos_j = {src: torch.zeros(M, nT, device=dev) for src in ("la_trained", "la_repeat")}
    acc_j = {src: torch.zeros(M, device=dev) for src in ("la_trained", "la_repeat", "dflash")}
    g_score, g_right = [], []
    n_j, lossless_diff, kernel_noise, argmax_agree = 0, 0.0, 0.0, 1.0
    with torch.inference_mode():
        for n_done, (i, anchors) in enumerate(by_seq.items()):
            ids, lo = val[i]
            x = ids.to(dev)
            rf = real_feats(x)
            if n_done < 3:      # lossless checks on real positions
                p0 = anchors[0] - M
                _, lg_on, _ = lookahead(x, p0, P.detach()[:M], keep_all=True)
                _, lg_off, _ = lookahead(x, p0, P.detach()[:M], keep_all=True, adapters=False)
                GATE["mask"] = None
                ref = target(x[None, :p0], use_cache=False).logits[0].float()
                lossless_diff = max(lossless_diff, float((lg_on[:p0] - lg_off[:p0]).abs().max()))
                kernel_noise = max(kernel_noise, float((lg_off[:p0] - ref).abs().max()))
                agree = float((lg_off[:p0].argmax(-1) == ref.argmax(-1)).float().mean())
                argmax_agree = min(argmax_agree, agree)
            for s in anchors:
                truth = x[s + 1 :].tolist()
                for m in args.m:
                    p = s - m
                    la = {}
                    la["la_trained"], lg_t, h_t = lookahead(x, p, P.detach()[:m])
                    rep = emb.weight[x[p - 1]].float()[None].repeat(m, 1)
                    la["la_repeat"], lg_r, _ = lookahead(x, p, rep, adapters=False)   # untrained: no adapters
                    GATE["mask"] = None
                    for src in srcs:
                        if src == "removed":
                            L = blocks_logits(draft, target, rf[:, :p], x, [(s, m)], bs)[0]       # as exp71
                        elif src == "real":
                            L = blocks_logits(draft, target, rf[:, :s], x, [(s, 0)], bs)[0]
                        else:
                            fz = torch.cat([rf[:, :p], la[src][None].to(rf.dtype)], 1)
                            L = blocks_logits(draft, target, fz, x, [(s, 0)], bs)[0]
                        tau[(src, m)].append(matched(L.argmax(-1).tolist(), truth) + 1)
                    if m == M:
                        tgt = x[p + 1 : p + m + 1]
                        rm = rf[0, p : p + m].float().view(m, nT, H)
                        for src, lg in (("la_trained", lg_t), ("la_repeat", lg_r)):
                            cos_j[src] += F.cosine_similarity(la[src].float().view(m, nT, H), rm, dim=-1)
                            acc_j[src] += (lg.argmax(-1) == tgt).float()
                        Ld = blocks_logits(draft, target, rf[:, :p], x, [(p, 0)], bs)[0][:m]
                        acc_j["dflash"] += (Ld.argmax(-1) == tgt).float()
                        g_score.append(gate_head(h_t.float())[:, 0].cpu())
                        g_right.append((lg_t.argmax(-1) == tgt).cpu())
                        n_j += 1
            if n_done % 20 == 0:
                print(f"[eval] {n_done}/{len(by_seq)} sequences, {time.time()-t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    print(f"\n[lossless] real positions, adapters on vs off: max |logit diff| {lossless_diff:.2e} (must be 0); "
          f"with vs without the appended placeholders: max |logit diff| {kernel_noise:.2e} (bf16 kernel noise only), "
          f"greedy token agreement {argmax_agree:.4f}")
    jrows = []
    for j in range(M):
        r = {"offset": j + 1}
        for src in ("la_trained", "la_repeat"):
            c = (cos_j[src][j] / max(n_j, 1)).tolist()
            r[f"{src}_cos_L{lids[-2]}"] = c[-2]
            r[f"{src}_cos_L{lids[-1]}"] = c[-1]
            r[f"{src}_acc"] = float(acc_j[src][j] / max(n_j, 1))
        r["dflash_acc"] = float(acc_j["dflash"][j] / max(n_j, 1))
        jrows.append(r)
    print_table(jrows, list(jrows[0]), f"1. information, no drafter: per placeholder offset j, cosine to the real state "
                                       f"(deep taps) and top-1 accuracy on x[p+j]; dflash_acc = DFlash from anchor p "
                                       f"with real features (it also knows x[p])")
    GS, GR = torch.cat(g_score), torch.cat(g_right)
    au = auroc(-GS.float(), ~GR)          # low gate score should flag wrong guesses
    print(f"gate: AUROC for flagging wrong placeholder guesses {au:.3f}; placeholder accuracy {float(GR.float().mean()):.3f}")
    rows = [{"m": m, **{src: mean(tau[(src, m)]) for src in srcs}} for m in args.m]
    print_table(rows, ["m", *srcs], "2. tau (DFlash not trained on these features: lower bound) when the last m context "
                                    "positions get real / no / learned-placeholder / untrained-placeholder features")
    save_json(args.save, {"args": vars(args), "lossless_max_diff": lossless_diff, "kernel_noise": kernel_noise, "per_offset": jrows,
                          "gate_auroc": au, "tau": rows, "n_anchors": sum(len(v) for v in by_seq.values()),
                          "train_log_tail": log[-50:]})
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
