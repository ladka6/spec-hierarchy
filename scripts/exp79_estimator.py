"""Experiment 79: a learned target-state estimator for drafted tokens, DFlash frozen.

Idea: after a draft, assume a prefix of it is accepted, estimate the target's features for those tokens without
running the target, and let DFlash draft the next block from them (chained drafting, one verification for both).
This script measures the estimator part only.

DFlash reads target layer outputs L1, L9, L17, L25, L33 (concatenated, fc, hidden_norm). exp71: only L25 / L33
matter (threshold ~ cosine 0.95). So the estimator predicts only L25 and L33; the shallow taps get a cheap filler.

Estimator (one model, two layer-matched stages):
  stage A  causal transformer over [context states ; embeddings of the new tokens]   -> L25 of the new tokens
  stage B  causal transformer over [context states ; stage-A hidden + L25 estimate]   -> L33 (+ learned skip from L25)
  --ctx fused   both stages attend to the real fused context vectors c = hidden_norm(fc(5 taps)) (what the faker did)
  --ctx layer   stage A attends to the real L25 states of the context, stage B to the real L33 states
Targets are standardised per dimension (outlier dimensions). Loss: standardised MSE + (1 - cosine) on L25 / L33
+ KL of the frozen drafter's blocks re-anchored inside the span (estimated vs real features).

Evaluation on the held-out part of --cache (same split as exp77):
  cosine per offset for L25, L33 and for the fused vector c (comparable with the faker's 0.38)
  drafter top-1 agreement (estimated vs real features)
  tau when the features of the last m context positions are replaced (exp71 protocol):
    real | removed | fill (real deep taps, shallow filler: tests the filler) | est | est + real shallow taps

  E0 (no training, filler check):  python scripts/exp79_estimator.py --cache seqs.pt --steps 0
  train:                           python scripts/exp79_estimator.py --cache seqs.pt --data .../data --ctx layer
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

from hspec.feathead import token_emb  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


# ----------------------------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        B, S, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, D))
        return x + self.mlp(self.ln2(x))


class Stage(nn.Module):
    def __init__(self, ctx_dim, in_dim, out_dim, d, layers, heads, window, span, ctx_norm=False):
        super().__init__()
        self.ctx_ln = nn.LayerNorm(ctx_dim) if ctx_norm else nn.Identity()
        self.ctx_in, self.tok_in = nn.Linear(ctx_dim, d), nn.Linear(in_dim, d)
        self.pos, self.seg = nn.Embedding(window + span, d), nn.Embedding(2, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, out_dim)

    def forward(self, ctx, tok, mask):
        W, L = ctx.shape[1], tok.shape[1]
        typ = torch.cat([torch.zeros(W, dtype=torch.long, device=ctx.device),
                         torch.ones(L, dtype=torch.long, device=ctx.device)])
        h = torch.cat([self.ctx_in(self.ctx_ln(ctx)), self.tok_in(tok)], 1) + self.pos.weight[: W + L] + self.seg(typ)
        for b in self.blocks:
            h = b(h, mask)
        hn = self.norm(h[:, W:])
        return hn, self.out(hn)


class Estimator(nn.Module):
    def __init__(self, H, ctx_dim, d=1024, layers=3, heads=16, window=128, span=8, ctx_norm=False, prev=False,
                 lookahead=False):
        super().__init__()
        self.cfg = dict(H=H, ctx_dim=ctx_dim, d=d, layers=layers, heads=heads, window=window, span=span,
                        ctx_norm=ctx_norm, prev=prev, lookahead=lookahead)
        self.window, self.span, self.prev = window, span, prev
        extra = ctx_dim if prev else 0          # the last real context state, given to every new position
        nxt = H if lookahead else 0             # embedding of the next token (the target's output if accepted)
        self.a = Stage(ctx_dim, H + nxt + extra, H, d, layers, heads, window, span, ctx_norm)
        self.b = Stage(ctx_dim, d + H + extra, H, d, layers, heads, window, span, ctx_norm)
        self.prev_ln = nn.LayerNorm(ctx_dim) if prev else None
        self.skip = nn.Parameter(torch.zeros(H))
        for n in ("mu25", "sd25", "mu33", "sd33"):
            self.register_buffer(n, torch.zeros(H) if n.startswith("mu") else torch.ones(H))

    def forward(self, ctx_a, ctx_b, e_tok, mask):
        L = e_tok.shape[1]
        ta, tb = [e_tok], []
        if self.prev:                           # EAGLE-style: the real state right before the span
            ta.append(self.prev_ln(ctx_a[:, -1:]).expand(-1, L, -1))
            tb.append(self.prev_ln(ctx_b[:, -1:]).expand(-1, L, -1))
        ha, y25 = self.a(ctx_a, torch.cat(ta, -1), mask)
        _, y33 = self.b(ctx_b, torch.cat([ha, y25] + tb, -1), mask)
        return y25, y33 + self.skip * y25                       # standardised L25 / L33 of the new tokens


def make_mask(valid, L, bidir=False):
    """valid [B, W] bool (real context positions) -> [B, 1, S, S] attention mask (True = attend). Causal; with
    bidir the new tokens see each other in both directions (all of them are known at inference)."""
    B, W = valid.shape
    S = W + L
    dev = valid.device
    causal = torch.ones(S, S, dtype=torch.bool, device=dev).tril()
    if bidir:
        causal[W:, W:] = True
    keys = torch.cat([valid, torch.ones(B, L, dtype=torch.bool, device=dev)], 1)
    m = (causal[None] & keys[:, None, :]) | torch.eye(S, dtype=torch.bool, device=dev)[None]
    return m[:, None]


def gather(states, s, W):
    """states [T, X] -> ([W, X] for positions s-W .. s-1, left zero-padded), valid [W]."""
    lo = max(0, s - W)
    n = s - lo
    out = states.new_zeros(W, states.shape[1])
    valid = torch.zeros(W, dtype=torch.bool, device=states.device)
    if n:
        out[W - n :] = states[lo:s]
        valid[W - n :] = True
    return out, valid


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences; held-out part = eval")
    ap.add_argument("--data", default="", help="gen_traindata shards for training")
    ap.add_argument("--ctx", choices=["fused", "layer"], default="layer")
    ap.add_argument("--shallow", choices=["zero", "mean", "real"], default="mean",
                    help="filler for the shallow taps of estimated positions")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3, help="per stage")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--span", type=int, default=8, help="new tokens estimated per anchor")
    ap.add_argument("--steps", type=int, default=8000, help="0 = no training (filler check / --load eval)")
    ap.add_argument("--batch", type=int, default=4, help="sequences per step")
    ap.add_argument("--anchors-train", type=int, default=4, help="anchors per sequence per step")
    ap.add_argument("--dense", type=int, default=0,
                    help="regression loss on anchors tiling the response (stride = span), at most this many per "
                         "sequence; the KL term stays on --anchors-train of them. 0 = off (regression on the KL anchors)")
    ap.add_argument("--offsets", type=int, default=4, help="re-anchored drafter blocks per anchor in the KL term")
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse", help="regression loss in standardised space")
    ap.add_argument("--zclip", type=float, default=0.0, help="clip standardised states (targets and context) to +-z")
    ap.add_argument("--ctx-norm", action="store_true", help="LayerNorm on the context inputs")
    ap.add_argument("--prev", action="store_true", help="give every new position the last real context state")
    ap.add_argument("--bidir", action="store_true", help="new tokens attend to each other in both directions")
    ap.add_argument("--lookahead", action="store_true",
                    help="give position p the embedding of x_{p+1} (the target's greedy output if the prefix is "
                         "accepted; for the last position it is the next block's anchor)")
    ap.add_argument("--var-span", action="store_true", help="train on random span lengths 1..span (with --bidir)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--anchors", type=int, default=400, help="eval anchors for tau")
    ap.add_argument("--m", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--stats-seqs", type=int, default=64)
    ap.add_argument("--load", default="")
    ap.add_argument("--stops", type=int, default=0,
                    help="stop-position eval on this many anchors instead of training: cut at block 1's first error "
                         "(oracle n), draft block 2 from real / tokens-only / estimated features, compare with a "
                         "normal second round")
    ap.add_argument("--out", default="")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    args.save = args.save or (f"exp79_estimator_{args.ctx}_{args.shallow}_s{args.steps}"
                              + ("_v2" if args.loss == "huber" or args.prev or args.ctx_norm else "")
                              + ("_bidir" if args.bidir else "") + ("_look" if args.lookahead else ""))
    assert max(args.m) <= args.span
    dev = "cuda"
    t0 = time.time()
    rng = random.Random(0)
    torch.manual_seed(0)

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    val = seqs[: max(int(len(seqs) * args.val_frac), 1)]
    train = []
    if args.data:
        for f in sorted(Path(args.data).glob("*.pt")):
            train += [(r["ids"].long(), int(r["n_prompt"])) for r in torch.load(f)]
    else:
        train = seqs[len(val) :]
    train = [(ids, lo) for ids, lo in train if len(ids) <= args.max_len and len(ids) - lo > args.span + 40]
    rng.shuffle(train)
    print(f"[data] {len(train)} train / {len(val)} eval sequences", flush=True)

    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H, nT = target.config.hidden_size, len(lids)
    assert lids[-2:] == [25, 33], lids
    sl = {t: slice(i * H, (i + 1) * H) for i, t in enumerate(lids)}
    shallow_cols = slice(0, (nT - 2) * H)
    L, W = args.span, args.window

    @torch.no_grad()
    def feats_of(x):
        out = target(x[None], output_hidden_states=True, use_cache=False, logits_to_keep=1)
        return extract_context_feature(out.hidden_states, lids)[0]                     # [T, 5H] bf16

    @torch.no_grad()
    def fused(f):
        return draft.hidden_norm(draft.fc(f[None]))[0].float()                         # [T, H]

    # per-dimension statistics of L25 / L33 and the shallow mean (response positions of training sequences)
    acc = {k: torch.zeros(H if k != "sh" else (nT - 2) * H, dtype=torch.float64, device=dev)
           for k in ("s25", "q25", "s33", "q33", "sh")}
    n_pos = 0
    for ids, lo in train[: args.stats_seqs]:
        f = feats_of(ids.to(dev))[lo:].double()
        for t in (25, 33):
            acc[f"s{t}"] += f[:, sl[t]].sum(0)
            acc[f"q{t}"] += f[:, sl[t]].pow(2).sum(0)
        acc["sh"] += f[:, shallow_cols].sum(0)
        n_pos += f.shape[0]
    stats = {}
    for t in (25, 33):
        mu = acc[f"s{t}"] / n_pos
        stats[f"mu{t}"] = mu.float()
        stats[f"sd{t}"] = (acc[f"q{t}"] / n_pos - mu * mu).clamp_min(1e-8).sqrt().float().clamp_min(1e-4)
    sh_mean = (acc["sh"] / n_pos).float()
    mean_feat = torch.cat([sh_mean, stats["mu25"], stats["mu33"]])          # all taps at their dataset mean
    print(f"[stats] {n_pos} positions; sd L25 median {stats['sd25'].median():.3f} max {stats['sd25'].max():.1f}; "
          f"sd L33 median {stats['sd33'].median():.3f} max {stats['sd33'].max():.1f}", flush=True)

    est = None
    if args.steps > 0 or args.load:
        est = Estimator(H, H, args.dim, args.layers, args.heads, W, L, args.ctx_norm, args.prev,
                        args.lookahead).to(dev)
        if args.load:
            d = torch.load(args.load, map_location="cpu")
            est = Estimator(**d["cfg"]).to(dev)
            est.load_state_dict(d["state"])
        else:
            for k, v in stats.items():
                getattr(est, k).copy_(v)
        print(f"[estimator] {sum(p.numel() for p in est.parameters()) / 1e6:.1f}M parameters, ctx={args.ctx}",
              flush=True)

    def std(f, t):
        z = (f[:, sl[t]].float() - getattr(est, f"mu{t}")) / getattr(est, f"sd{t}")
        return z.clamp(-args.zclip, args.zclip) if args.zclip > 0 else z

    def shallow_fill(f, s, n):
        if args.shallow == "zero":
            return torch.zeros(n, (nT - 2) * H, device=dev)
        if args.shallow == "mean":
            return sh_mean[None].expand(n, -1)
        return f[s : s + n, shallow_cols].float()

    def estimate(f, x, anchors, n=None):
        """standardised (y25, y33) [A, n, H] for positions s .. s+n-1 of each anchor s (n = span by default)."""
        n = n or L
        if args.ctx == "layer":
            sa, sb = std(f, 25), std(f, 33)
        else:
            sa = sb = fused(f)
        ca, cb, vv = [], [], []
        for s in anchors:
            a_, v = gather(sa, s, W)
            b_, _ = gather(sb, s, W)
            ca.append(a_)
            cb.append(b_)
            vv.append(v)
        toks = torch.stack([x[s : s + n] for s in anchors])
        e = token_emb(draft, target, toks).float()
        if est.cfg.get("lookahead"):
            nx = torch.stack([x[s + 1 : s + n + 1] for s in anchors])
            e = torch.cat([e, token_emb(draft, target, nx).float()], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y25, y33 = est(torch.stack(ca), torch.stack(cb), e, make_mask(torch.stack(vv), n, args.bidir))
        return y25.float(), y33.float()

    def raw_part(f, s, y25, y33, n, real_shallow=False):
        """[n, 5H] features for positions s .. s+n-1: shallow filler + de-standardised estimates."""
        sh = f[s : s + n, shallow_cols].float() if real_shallow else shallow_fill(f, s, n)
        r25 = y25[:n] * est.sd25 + est.mu25
        r33 = y33[:n] * est.sd33 + est.mu33
        return torch.cat([sh, r25, r33], -1)

    # ------------------------------------------------------------------------------------- evaluation
    def evaluate(n_seqs=None, with_tau=True, tag="final"):
        if est is not None:
            est.eval()
        vr = random.Random(123)
        vs = val if n_seqs is None else val[:n_seqs]
        cos = {k: torch.zeros(L, device=dev) for k in ("L25", "L33", "zL25", "zL33", "c")}
        cnt = torch.zeros(L, device=dev)
        agree, n_ag = 0.0, 0
        cand = [(i, s) for i, (ids, lo) in enumerate(vs) for s in range(lo + L + 8, len(ids) - bs - 1)]
        vr.shuffle(cand)
        tau_by = {}
        for i, s in cand[: args.anchors]:
            tau_by.setdefault(i, []).append(s)
        conds = [("real", 0)] + [(k, m) for m in args.m for k in ("removed", "mean", "fill")]
        if est is not None:
            conds += [(k, m) for m in args.m for k in ("est", "est_realshallow")]
        taus = {c: [] for c in conds}
        with torch.inference_mode():
            for i, (ids, lo) in enumerate(vs):
                if not with_tau and est is None:
                    break
                x = ids.to(dev)
                f = feats_of(x)
                k_ = max(0, min(4, len(ids) - L - lo - 2))
                if est is not None and k_ > 0:
                    anc = vr.sample(range(lo + 1, len(ids) - L - 1), k_)
                    fz_all = fused(f)
                    y25, y33 = estimate(f, x, anc)
                    for a, s in enumerate(anc):
                        r = f[s : s + L].float()
                        e5 = raw_part(f, s, y25[a], y33[a], L, real_shallow=True)
                        cos["L25"] += F.cosine_similarity(e5[:, sl[25]], r[:, sl[25]], dim=-1)
                        cos["L33"] += F.cosine_similarity(e5[:, sl[33]], r[:, sl[33]], dim=-1)
                        cos["zL25"] += F.cosine_similarity(y25[a], std(f, 25)[s : s + L], dim=-1)
                        cos["zL33"] += F.cosine_similarity(y33[a], std(f, 33)[s : s + L], dim=-1)
                        e5f = raw_part(f, s, y25[a], y33[a], L)
                        cf = draft.hidden_norm(draft.fc(e5f.to(f.dtype)[None]))[0].float()
                        cos["c"] += F.cosine_similarity(cf, fz_all[s : s + L], dim=-1)
                        cnt += 1
                        offs = [j for j in range(1, L + 1) if s + j + bs < len(ids)]
                        if offs:
                            blocks = [(s + j, 0) for j in offs]
                            mix = torch.cat([f[:s], e5f.to(f.dtype)], 0)[None]
                            lg = blocks_logits(draft, target, mix, x, blocks, bs).argmax(-1)
                            ref = blocks_logits(draft, target, f[None, : s + L], x, blocks, bs).argmax(-1)
                            agree += float((lg == ref).float().sum())
                            n_ag += lg.numel()
                if not with_tau or i not in tau_by:
                    continue
                for s in tau_by[i]:
                    truth = x[s + 1 :].tolist()
                    for kind, m in conds:
                        if kind == "real":
                            fz = f[None, :s]
                            blocks = [(s, 0)]
                        elif kind == "removed":
                            fz = f[None, : s - m]
                            blocks = [(s, m)]
                        else:
                            fz = f[None, :s].clone()
                            if kind == "fill":
                                fz[0, s - m : s, shallow_cols] = shallow_fill(f, s - m, m).to(f.dtype)
                            elif kind == "mean":
                                fz[0, s - m : s] = mean_feat[None].to(f.dtype)
                            else:
                                y25, y33 = estimate(f, x, [s - m], m)
                                fz[0, s - m : s] = raw_part(f, s - m, y25[0], y33[0], m,
                                                            real_shallow=kind == "est_realshallow").to(f.dtype)
                            blocks = [(s, 0)]
                        Lg = blocks_logits(draft, target, fz, x, blocks, bs)[0]
                        taus[(kind, m)].append(matched(Lg.argmax(-1).tolist(), truth) + 1)
        if est is not None:
            est.train()
        res = {}
        if est is not None and cnt.sum() > 0:
            for k in cos:
                res[f"cos_{k}"] = (cos[k] / cnt.clamp_min(1)).tolist()
            res["agree"] = agree / max(n_ag, 1)
            print(f"[{tag}] cos by offset  L25: {' '.join(f'{v:.3f}' for v in res['cos_L25'])}", flush=True)
            print(f"[{tag}] cos by offset  L33: {' '.join(f'{v:.3f}' for v in res['cos_L33'])}", flush=True)
            print(f"[{tag}] standardised  zL25: {' '.join(f'{v:.3f}' for v in res['cos_zL25'])}", flush=True)
            print(f"[{tag}] standardised  zL33: {' '.join(f'{v:.3f}' for v in res['cos_zL33'])}", flush=True)
            print(f"[{tag}] cos by offset    c: {' '.join(f'{v:.3f}' for v in res['cos_c'])}  "
                  f"drafter top-1 agreement {res['agree']:.3f}", flush=True)
        if with_tau:
            res["tau"] = {f"{k}|{m}": sum(v) / max(len(v), 1) for (k, m), v in taus.items()}
        return res

    # ------------------------------------------------------------------ stop-position eval (the chained round)
    def stop_eval():
        """Block 1 at anchor s (real features) matches n tokens. Assume the n are accepted (oracle cut) and draft
        block 2 at anchor a = s+n, which needs features for positions s .. a-1 (n of them):
          real    the verifier's features (ceiling; what a copy gives)
          tokens  none: the n tokens as plain inputs (lagged block; a lag-trained drafter is the fair version)
          est     the estimator's features (n <= span)
        Block 2's first token re-predicts position a+1, where block 1 failed (no bonus token yet).
        One verification of block 1 + block 2 gives n + m2 + 1 tokens. Normal DFlash needs a second verification:
        anchor at the bonus a+1 with real features, m_next matched -> ((n+1) + (m_next+1)) / 2 per verification."""
        vr = random.Random(7)
        cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 8, len(ids) - 3 * bs)]
        vr.shuffle(cand)
        by = {}
        for i, s in cand[: args.stops]:
            by.setdefault(i, []).append(s)
        nmax = min(bs - 2, L) if est is not None else bs - 2
        recs, n_all, n_dist = [], 0, {}
        with torch.inference_mode():
            for i, anchors in by.items():
                ids, lo = val[i]
                x = ids.to(dev)
                f = feats_of(x)
                for s in anchors:
                    y1 = blocks_logits(draft, target, f[None, :s], x, [(s, 0)], bs)[0].argmax(-1).tolist()
                    n = matched(y1, x[s + 1 :].tolist())
                    n_all += 1
                    n_dist[n] = n_dist.get(n, 0) + 1
                    if n < 1 or n > nmax:
                        continue
                    a = s + n
                    truth = x[a + 1 :].tolist()
                    r = {"n": n}
                    lg = blocks_logits(draft, target, f[None, :a], x, [(a, 0)], bs)[0]
                    r["real"] = matched(lg.argmax(-1).tolist(), truth)
                    lg = blocks_logits(draft, target, f[None, :s], x, [(a, n)], bs)[0]
                    r["tokens"] = matched(lg.argmax(-1).tolist(), truth)
                    if est is not None:
                        y25, y33 = estimate(f, x, [s], n)
                        fz = f[None, :a].clone()
                        fz[0, s:a] = raw_part(f, s, y25[0], y33[0], n).to(f.dtype)
                        lg = blocks_logits(draft, target, fz, x, [(a, 0)], bs)[0]
                        r["est"] = matched(lg.argmax(-1).tolist(), truth)
                    lg = blocks_logits(draft, target, f[None, : a + 1], x, [(a + 1, 0)], bs)[0]
                    r["m_next"] = matched(lg.argmax(-1).tolist(), x[a + 2 :].tolist())
                    recs.append(r)
        k = len(recs)
        normal = sum((r["n"] + 1 + r["m_next"] + 1) / 2 for r in recs) / max(k, 1)
        rows = []
        for c in ("real", "tokens", "est"):
            if not recs or c not in recs[0]:
                continue
            rows.append({"block 2 from": c,
                         "fixes failed position": sum(r[c] >= 1 for r in recs) / k,
                         "mean m2": sum(r[c] for r in recs) / k,
                         "tokens / verification (chained)": sum(r["n"] + r[c] + 1 for r in recs) / k,
                         "normal DFlash (2 verifications)": normal})
        print(f"[stops] {k} of {n_all} rounds with 1 <= n <= {nmax} (mean n {sum(r['n'] for r in recs) / max(k, 1):.2f}); "
              f"n distribution {dict(sorted(n_dist.items()))}", flush=True)
        print_table(rows, list(rows[0]), f"chained round at block 1's first error (oracle cut), drafter {args.draft}")
        save_json(args.save + "_stops", {"args": vars(args), "rows": rows, "records": recs, "n_dist": n_dist})

    if args.stops:
        stop_eval()
        print(f"[done] {time.time() - t0:.0f}s", flush=True)
        return

    hist = []
    if est is not None and args.steps > 0:
        opt = torch.optim.AdamW(est.parameters(), lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps)
                                                                                     / args.steps))))
        hist.append({"step": 0, **evaluate(32, with_tau=False, tag="step 0")})
        est.train()
        it, step, tl = 0, 0, time.time()
        run = {"mse": 0.0, "cos25": 0.0, "cos33": 0.0, "kl": 0.0, "n": 0}
        while step < args.steps:
            opt.zero_grad(set_to_none=True)
            for _ in range(args.batch):
                ids, lo = train[it % len(train)]
                it += 1
                x = ids.to(dev)
                f = feats_of(x)
                T = len(ids)
                hi = T - L - bs - 1
                if hi <= lo + 1:
                    continue
                if args.dense:
                    start = lo + 1 + rng.randrange(L)
                    anc = list(range(start, hi, L))
                    if len(anc) > args.dense:
                        anc = sorted(rng.sample(anc, args.dense))
                    if not anc:
                        continue
                else:
                    anc = rng.sample(range(lo + 1, hi), min(args.anchors_train, hi - lo - 1))
                kl_idx = sorted(rng.sample(range(len(anc)), min(args.anchors_train, len(anc))))
                Lb = rng.randint(1, L) if args.var_span else L
                y25, y33 = estimate(f, x, anc, Lb)
                r25 = torch.stack([std(f, 25)[s : s + Lb] for s in anc])
                r33 = torch.stack([std(f, 33)[s : s + Lb] for s in anc])
                if args.loss == "huber":
                    mse = F.smooth_l1_loss(y25, r25) + F.smooth_l1_loss(y33, r33)
                else:
                    mse = (y25 - r25).pow(2).mean() + (y33 - r33).pow(2).mean()
                raw = {a: raw_part(f, anc[a], y25[a], y33[a], Lb) for a in kl_idx}
                c_idx = kl_idx
                rr = {a: f[anc[a] : anc[a] + Lb].float() for a in c_idx}
                c25 = torch.stack([F.cosine_similarity(raw[a][:, sl[25]], rr[a][:, sl[25]], dim=-1) for a in c_idx])
                c33 = torch.stack([F.cosine_similarity(raw[a][:, sl[33]], rr[a][:, sl[33]], dim=-1) for a in c_idx])
                loss = mse + args.cos_weight * ((1 - c25).mean() + (1 - c33).mean())
                kl_tot = 0.0
                if args.kl_weight > 0:
                    for a in kl_idx:
                        s = anc[a]
                        offs = sorted(rng.sample(range(1, Lb + 1), min(args.offsets, Lb)))
                        blocks = [(s + j, 0) for j in offs]
                        mix = torch.cat([f[:s], raw[a].to(f.dtype)], 0)[None]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lg = blocks_logits(draft, target, mix, x, blocks, bs).float()
                        with torch.no_grad():
                            ref = blocks_logits(draft, target, f[None, : s + Lb], x, blocks, bs).float()
                        kl = F.kl_div(torch.log_softmax(lg, -1), torch.log_softmax(ref, -1), log_target=True,
                                      reduction="none").sum(-1).mean()
                        kl_tot = kl_tot + kl / len(kl_idx)
                    loss = loss + args.kl_weight * kl_tot
                (loss / args.batch).backward()
                run["mse"] += float(mse.detach())
                run["cos25"] += float(c25.mean())
                run["cos33"] += float(c33.mean())
                run["kl"] += float(kl_tot)
                run["n"] += 1
            torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 and run["n"]:
                n = run["n"]
                print(f"[step {step}/{args.steps}] {args.loss} {run['mse'] / n:.4f} cos25 {run['cos25'] / n:.3f} "
                      f"cos33 {run['cos33'] / n:.3f} kl {run['kl'] / n:.4f}  {(time.time() - tl) / step:.2f}s/step",
                      flush=True)
                run = {k: 0.0 for k in run} | {"n": 0}
            if step % args.eval_every == 0 or step == args.steps:
                if step < args.steps:
                    hist.append({"step": step, **evaluate(32, with_tau=False, tag=f"step {step}")})
                if args.out:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state": est.state_dict(), "cfg": est.cfg}, args.out)

    res = evaluate(tag="final")
    ref = {"real": res["tau"]["real|0"]}
    rows = []
    for m in args.m:
        row = {"m": m, "real": ref["real"], "removed": res["tau"][f"removed|{m}"], "mean feats": res["tau"][f"mean|{m}"],
               f"fill ({args.shallow}, real deep)": res["tau"][f"fill|{m}"]}
        if est is not None:
            row["est"] = res["tau"][f"est|{m}"]
            row["est + real shallow"] = res["tau"][f"est_realshallow|{m}"]
        rows.append(row)
    print_table(rows, list(rows[0]), f"tau when the features of the last m context positions are replaced "
                                     f"(ctx={args.ctx}, shallow filler={args.shallow}, {args.anchors} anchors)")
    save_json(args.save, {"args": vars(args), "history": hist, "final": res})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
