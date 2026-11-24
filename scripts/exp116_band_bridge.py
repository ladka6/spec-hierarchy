"""Experiment 116: the 3/4-depth state for in-flight tokens from the BAND that creates it. Which bridge, which slice?

exp111 / exp115 (Qwen3-8B, LLaMA-3.1-8B, gemma4-12B): for the newest context positions the drafter lives on ONE
tap, at about 3/4 of the depth; the predictive content of the target's states is flat before about half depth and
rises between L/2 and 3L/4. Hypothesis: for the in-flight tokens only the band L/2 .. 3L/4 has to run; what comes
before it can be bridged, because the band layers read the REAL cache of the verified prefix at their own depth
(the context does not pass through the bridge) and the local content at the band's entry is already present in
the shallow states.

Sub-stack (frozen target layers, draft mode only, the verify pass is untouched): a set of kept layers below the
tap layer c; every removed span is replaced by a bridge; every kept layer attends to its own original key / value
cache of the verified prefix and to the in-flight tokens. Output = the state after layer c = the drafter's tap.

Shapes at a budget of n layers (c = the drafter's tap nearest to 3/4 depth, L = number of layers):
  band:a     the first a layers, one bridge, then the n - a layers that END at c      (a = 0: bridge from embeddings)
  first:0    layers 0 .. n-1, then one bridge up to the tap                           (early-exit style)
  skip:0     n layers spread evenly over 0 .. c, bridges in between
  plateau:a  the first a layers, then the n - a layers that end at L/2, bridges around (the slice before the band)
Bridge kinds (the closed-form affine start, ridge on real activations, is common to all but "none"):
  none   no bridge at all: the state is passed on unchanged
  lin    the affine map only, nothing trained                                         (ReplaceMe-style)
  mlp    + position-wise residual MLP
  attn   + causal self-attention over the in-flight tokens + MLP
  ctx    + attention from the in-flight tokens to the REAL states of the last --ctx-win verified positions at the
         bridge's OUTPUT depth (the verify pass has them: correct examples of the space the bridge writes into)
         and to the in-flight tokens themselves, + MLP
Trained parts: the bridges only, standardised MSE to the real tap state (+ --mid-w on every bridge's own output).

Evaluation as exp111 ("only this tap real"): the last m context positions get the dataset mean for every tap
except the 3/4 tap, which comes from the sub-stack run over those m positions as in-flight tokens.
  tau m      accepted length of one block (drafted + 1)
  of 1 tap   (tau - removed) / (real tap - removed): share of what this one tap can give at all
  cos        cosine between produced and real tap state, both centred on the dataset mean
Reference rows: all taps real, no features for the last m positions (removed), only the real tap.
Also printed: cosine between the states at the two ends of every block of --sim-block layers (the usual
"which block can be skipped" criterion), to see whether the plateau is also flat in that sense.

  python scripts/exp116_band_bridge.py --data /scratch-shared/$USER/hspec_lagft/data --budget 10 \
      --configs band:2:lin band:2:mlp band:2:ctx first:0:mlp skip:0:mlp
"""

from __future__ import annotations

import argparse
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
from exp107_lift_diag import load_seqs  # noqa: E402
from exp108b_nonlinear_lift import cosine_lr  # noqa: E402
from exp112_draft_branch import BF, rot  # noqa: E402

from hspec.bridges import Bridge, RMSNorm, spans_of  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

KINDS = ("none", "lin", "mlp", "attn", "ctx")
MAX_M = 64


class CtxBridge(nn.Module):
    """affine -> attention to the real states of the verified positions at the output depth (+ in-flight, causal)
    -> residual MLP. Attention output and MLP output start at zero: at initialisation this is the affine map."""

    def __init__(self, H, rank, win, heads=8, hd=64):
        super().__init__()
        self.win, self.h, self.d = win, heads, hd
        self.lin = nn.Linear(H, H, bias=True)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.nq, self.nm, self.n1 = RMSNorm(H), RMSNorm(H), RMSNorm(H)
        self.q = nn.Linear(H, heads * hd, bias=False)
        self.k = nn.Linear(H, heads * hd, bias=False)
        self.v = nn.Linear(H, heads * hd, bias=False)
        self.o = nn.Linear(heads * hd, H, bias=False)
        nn.init.zeros_(self.o.weight)
        self.pos = nn.Embedding(win + MAX_M, heads * hd)     # index win = the first in-flight position
        nn.init.normal_(self.pos.weight, std=0.02)
        self.gate = nn.Linear(H, rank, bias=False)
        self.up = nn.Linear(H, rank, bias=False)
        self.down = nn.Linear(rank, H, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x, mem):                               # x [1, m, H] in-flight, mem [1, w, H] real, w <= win
        dt = x.dtype
        x = x.float()
        x = x + self.lin(x)
        m, w = x.shape[1], mem.shape[1]
        xq = self.nq(x).float()
        allm = torch.cat([self.nm(mem.float()).float(), xq], 1)
        pe = self.pos(torch.arange(self.win - w, self.win + m, device=x.device))
        k = (self.k(allm) + pe).view(1, w + m, self.h, self.d).transpose(1, 2)
        v = self.v(allm).view(1, w + m, self.h, self.d).transpose(1, 2)
        q = (self.q(xq) + pe[w:]).view(1, m, self.h, self.d).transpose(1, 2)
        mask = torch.ones(m, w + m, dtype=torch.bool, device=x.device)
        mask[:, w:] = torch.tril(mask[:, w:])
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(1, m, self.h * self.d))
        y = self.n1(x).float()
        x = x + self.down(F.silu(self.gate(y)) * self.up(y))
        return x.to(dt)


class BandStack(nn.Module):
    """Kept target layers below `stop` over the in-flight positions; removed spans bridged; output = state after
    layer stop - 1 (or the last bridge's output when the top layers are removed)."""

    def __init__(self, target, kept, stop, kind, rank, win):
        super().__init__()
        cfg = target.config
        self.stop, self.kept, self.kset, self.kind = stop, sorted(kept), set(kept), kind
        assert self.kept and max(self.kept) < stop
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nh
        self._orig = list(target.model.layers)              # plain list: the target is not a submodule
        H, dev = cfg.hidden_size, target.lm_head.weight.device
        self.spans = spans_of(self.kept, stop)
        mods = {}
        if kind != "none":
            for _, b in self.spans:
                mods[str(b)] = CtxBridge(H, rank, win) if kind == "ctx" else Bridge(H, kind, rank)
        self.bridges = nn.ModuleDict(mods).to(dev)

    def cost(self):
        full = sum(p.numel() for p in self._orig[0].parameters())
        return (len(self.kept) * full + sum(p.numel() for p in self.parameters())) / full

    def _kv(self, attn, hn, cos, sin):
        B, T, _ = hn.shape
        kk = attn.k_proj(hn).view(B, T, -1, self.hd)
        if hasattr(attn, "k_norm"):
            kk = attn.k_norm(kk)
        kk = rot(kk.transpose(1, 2).to(BF), cos, sin)
        vv = attn.v_proj(hn).view(B, T, -1, self.hd).transpose(1, 2).to(BF)
        return kk, vv

    @torch.no_grad()
    def prefix_kv(self, hs, cos, sin):
        """the ORIGINAL post-rotary keys / values of every kept layer over the whole sequence (the verify's cache)"""
        return {g: self._kv(self._orig[g].self_attn, self._orig[g].input_layernorm(hs[g]), cos, sin) for g in self.kept}

    def forward(self, x0, a, kvp, cos, sin, hs):
        """x0 [1, m, H] embeddings of positions a .. a+m-1; hs = the real hidden states (only rows < a are read, by
        the ctx bridges). Returns the produced state [1, m, H] and every bridge's output {span end: [1, m, H]}."""
        m = x0.shape[1]
        mask = torch.ones(m, a + m, dtype=torch.bool, device=x0.device)
        mask[:, a:] = torch.tril(mask[:, a:])
        rep = self.nh // self.nkv
        x, outs = x0, {}
        for g in range(self.stop):
            if g not in self.kset:
                if str(g) in self.bridges:                   # last layer of a removed span
                    br = self.bridges[str(g)]
                    with torch.autocast("cuda", enabled=False):          # bridges run in fp32
                        if isinstance(br, CtxBridge):
                            x = br(x, hs[g + 1][:, max(0, a - br.win) : a])
                        else:
                            x = br(x)
                    outs[g] = x
                continue
            lay = self._orig[g]
            attn = lay.self_attn
            hn = lay.input_layernorm(x)
            q = attn.q_proj(hn).view(1, m, -1, self.hd)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            q = rot(q.transpose(1, 2).to(BF), cos, sin)
            kn, vn = self._kv(attn, hn, cos, sin)
            Kp, Vp = kvp[g]
            K = torch.cat([Kp[:, :, :a], kn], 2).repeat_interleave(rep, 1)
            V = torch.cat([Vp[:, :, :a], vn], 2).repeat_interleave(rep, 1)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            x = x + attn.o_proj(o.transpose(1, 2).reshape(1, m, -1))
            h2 = lay.post_attention_layernorm(x)
            mlp = lay.mlp
            x = x + mlp.down_proj(mlp.act_fn(mlp.gate_proj(h2)) * mlp.up_proj(h2))
        return x, outs


def spread(n, top):
    return sorted({round(j * (top - 1) / max(n - 1, 1)) for j in range(n)})


def layers_of(shape, a, n, c, L):
    if shape == "band":
        return list(range(a)) + list(range(c - (n - a) + 1, c + 1))
    if shape == "first":
        return list(range(n))
    if shape == "skip":
        return spread(n, c + 1)
    if shape == "plateau":
        return list(range(a)) + list(range(L // 2 - (n - a), L // 2))
    raise ValueError(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--configs", nargs="+", default=["band:2:lin", "band:2:mlp", "band:2:ctx"],
                    help="shape:a:kind, shape in band / first / skip / plateau, kind in " + " / ".join(KINDS))
    ap.add_argument("--budget", type=int, default=10, help="number of target layers run")
    ap.add_argument("--tap-depth", type=float, default=0.75, help="use the drafter's tap nearest to this depth")
    ap.add_argument("--rank", type=int, default=1024, help="width of the bridges' MLP")
    ap.add_argument("--ctx-win", type=int, default=64)
    ap.add_argument("--mid-w", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--blocks", type=int, default=8, help="anchors per sequence per step")
    ap.add_argument("--m-train", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-seqs", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--fit-seqs", type=int, default=150, help="sequences for the closed-form affine maps")
    ap.add_argument("--stat-seqs", type=int, default=60, help="sequences for means / stds / block similarity")
    ap.add_argument("--sim-block", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--ms", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--fit-tokens", type=int, default=600_000, help="as exp112: fixes the train / held-out split")
    ap.add_argument("--val-tokens", type=int, default=15_000)
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--save", default="exp116_band_bridge")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    cfgs = []
    for c in args.configs:
        shape, a, kind = c.split(":")
        assert kind in KINDS and max(args.ms + [args.m_train]) <= MAX_M
        cfgs.append((shape, int(a), kind))
    dev, t0 = "cuda", time.time()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    rng = random.Random(0)

    seqs = load_seqs(args)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H, L, nT = target.config.hidden_size, len(target.model.layers), len(lids)
    tix = min(range(nT), key=lambda t: abs((lids[t] + 1) / L - args.tap_depth))
    c = lids[tix]                                            # the tap = state after layer c = hidden_states[c + 1]
    stop = c + 1
    M = max(args.ms)
    print(f"[setup] {len(seqs)} sequences, {L} layers, taps {lids}, tap used: layer {c} "
          f"(depth {(c + 1) / L:.2f}), budget {args.budget} layers, {time.time() - t0:.0f}s", flush=True)

    # ---- the split of exp112 / exp113 (same train / validation sequences, same held-out anchors) ----
    def take(i, n_tokens, cap):
        recs, n = [], 0
        while n < n_tokens and i < cap:
            ids, lo = seqs[i]
            n += len(ids) - lo
            recs.append(i)
            i += 1
        return recs, i

    max_fit = int(0.6 * len(seqs))
    fit_seqs, i = take(0, args.fit_tokens, max_fit)
    val_recs, fit_end = take(i, args.val_tokens, max_fit + (len(seqs) - max_fit) // 4)
    assert val_recs, "no validation sequences: add data or lower --fit-tokens"
    val_seqs = val_recs[: args.val_seqs]
    rest = seqs[fit_end:]
    cand_a = [(si, s) for si, (ids, lo) in enumerate(rest) for s in range(max(lo, 18), len(ids) - bs)]
    random.Random(1).shuffle(cand_a)
    by_seq = {}
    for si, s in sorted(cand_a[: args.anchors]):
        by_seq.setdefault(si, []).append(s)

    @torch.no_grad()
    def hidden(ids):
        return target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states

    @torch.no_grad()
    def seq_ctx(ids, stack):
        hs = hidden(ids)
        pos = torch.arange(len(ids), device=dev)[None]
        cos, sin = target.model.rotary_emb(hs[0], pos)
        with torch.autocast("cuda", dtype=BF):
            kvp = stack.prefix_kv(hs, cos, sin)
        return hs, cos, sin, kvp

    def run(stack, ctx, a, b):
        """produced tap state and bridge outputs for the in-flight positions a .. b-1 (true tokens as input)"""
        hs, cos, sin, kvp = ctx
        with torch.autocast("cuda", dtype=BF):
            return stack(hs[0][:, a:b], a, kvp, cos[:, a:b], sin[:, a:b], hs)

    def block(feats, x, s, lag=0):
        with torch.autocast("cuda", dtype=BF):
            return blocks_logits(draft, target, feats, x, [(s, lag)], bs)[0].argmax(-1).tolist()

    def mean(v):
        return sum(v) / max(len(v), 1)

    # ---- statistics: per-layer mean / std (standardised losses), tap means (ablation), block similarity ----
    with torch.no_grad():
        s1 = torch.zeros(stop + 1, H, dtype=torch.float64, device=dev)
        s2 = torch.zeros(stop + 1, H, dtype=torch.float64, device=dev)
        fsum, n_tok = torch.zeros(nT * H, dtype=torch.float64, device=dev), 0
        nb = args.sim_block
        sim = {l: 0.0 for l in range(0, L - nb + 1)}
        for si in fit_seqs[: args.stat_seqs]:
            ids, lo = seqs[si]
            hs = hidden(ids)
            st = torch.stack([hs[l][0, lo:] for l in range(stop + 1)]).double()
            s1 += st.sum(1)
            s2 += (st * st).sum(1)
            fsum += extract_context_feature(hs, lids)[0, lo:].double().sum(0)
            for l in sim:
                sim[l] += F.cosine_similarity(hs[l][0, lo:].float(), hs[l + nb][0, lo:].float(), dim=-1).sum().item()
            n_tok += len(ids) - lo
            del hs, st
        mu = (s1 / n_tok).float()
        sd = (s2 / n_tok - (s1 / n_tok) ** 2).clamp_min(1e-6).sqrt().float()
        mean_vec = (fsum / n_tok).to(torch.bfloat16)
        sim_rows = [{"block": f"{l}..{l + nb - 1}", "from depth": l / L, "to depth": (l + nb) / L,
                     "cos(in, out)": v / n_tok} for l, v in sim.items()]
        print_table(sim_rows, list(sim_rows[0]),
                    f"similarity of the states before and after every block of {nb} layers (higher = easier to skip)")

    # ---- start-up check: all layers below the tap kept -> the real tap state ----
    with torch.no_grad():
        ids, _lo = seqs[0]
        full = BandStack(target, list(range(stop)), stop, "none", 8, args.ctx_win)
        ctx = seq_ctx(ids, full)
        a = len(ids) - 16
        z, _ = run(full, ctx, a, len(ids))
        cs = F.cosine_similarity(z[0].float(), ctx[0][stop][0, a:].float(), dim=-1).mean().item()
        print(f"[check] layers 0..{c} kept: cosine(produced, real tap) {cs:.4f}", flush=True)
        assert cs > 0.99, "the sub-stack forward does not reproduce the target's tap state"
        del full, ctx, z

    @torch.no_grad()
    def fit_affine(stack):
        """closed-form affine part of every bridge: ridge regression of the removed span's update on its input"""
        if not stack.bridges:
            return
        acc = {b: [torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev),
                   torch.zeros(H + 1, H, dtype=torch.float64, device=dev)] for _, b in stack.spans}
        for si in fit_seqs[: args.fit_seqs]:
            hs = hidden(seqs[si][0])
            for a, b in stack.spans:
                x = hs[a][0].double()
                d = hs[b + 1][0].double() - x
                w = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                xa = torch.cat([x, torch.ones_like(x[:, :1])], -1) * w
                acc[b][0] += xa.T @ xa
                acc[b][1] += xa.T @ (d * w)
            del hs
        for _, b in stack.spans:
            xtx, xtd = acc[b]
            sol = torch.linalg.solve(
                xtx + args.ridge * xtx.diagonal().mean() * torch.eye(H + 1, dtype=xtx.dtype, device=dev), xtd)
            stack.bridges[str(b)].lin.weight.copy_(sol[:H].T.float())
            stack.bridges[str(b)].lin.bias.copy_(sol[H].float())
        del acc

    def std_mse(pred, real, layer):
        return F.mse_loss((pred.float() - mu[layer]) / sd[layer], (real.float() - mu[layer]) / sd[layer])

    def loss_fn(stack, si, n_blocks, grad):
        ids, lo = seqs[si]
        T, m = len(ids), args.m_train
        if T - m - max(lo, 18) < 1:
            return None
        ctx = seq_ctx(ids, stack)
        hs = ctx[0]
        tap_t, mid_t = 0.0, 0.0
        with torch.set_grad_enabled(grad):
            for _ in range(n_blocks):
                s = rng.randrange(max(lo, 18), T - m)
                out, outs = run(stack, ctx, s, s + m)
                tap_t = tap_t + std_mse(out[0], hs[stop][0, s : s + m], stop)
                for b, o in outs.items():
                    if b + 1 != stop:
                        mid_t = mid_t + std_mse(o[0], hs[b + 1][0, s : s + m], b + 1)
        return tap_t / n_blocks, mid_t / n_blocks

    def train(tag, stack):
        params = list(stack.bridges.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

        def scorer():
            stack.eval()
            st = rng.getstate()
            rng.seed(123)
            tot, n = 0.0, 0
            with torch.no_grad():
                for si in val_seqs:
                    out = loss_fn(stack, si, 4, False)
                    if out is not None:
                        tot += float(out[0])
                        n += 1
            rng.setstate(st)
            stack.train()
            return -tot / max(n, 1)

        def snap():
            return {n: v.detach().cpu().clone() for n, v in stack.state_dict().items()}

        best_sc, best_state, bad = scorer(), snap(), 0
        print(f"[{tag}] start val {best_sc:.4f}, trainable {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
        for step in range(args.steps):
            for g in opt.param_groups:
                g["lr"] = args.lr * cosine_lr(step, args.steps, min(200, max(args.steps // 10, 1)))
            out = None
            for _ in range(10):
                out = loss_fn(stack, rng.choice(fit_seqs), args.blocks, True)
                if out is not None:
                    break
            tap, mid = out
            loss = tap + args.mid_w * mid
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                sc = scorer()
                print(f"[{tag}] step {step + 1} tap {float(tap):.4f} mid {float(mid):.4f} val {sc:.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
                if sc > best_sc:
                    best_sc, best_state, bad = sc, snap(), 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        print(f"[{tag}] early stop at step {step + 1}", flush=True)
                        break
        stack.load_state_dict(best_state)
        stack.eval()

    # ---- references per anchor (independent of the sub-stack): all taps real, removed, only the tap real ----
    ref_cache = {}

    def one_tap(feats, s, m, tap_rows):
        """features for the block at s: the last m rows hold the dataset mean except the tap, = tap_rows [m, H]"""
        fz = feats[:, :s].clone()
        blk = mean_vec.view(1, nT, H).repeat(m, 1, 1)
        blk[:, tix] = tap_rows.to(blk.dtype)
        fz[0, s - m : s] = blk.view(m, nT * H)
        return fz

    @torch.no_grad()
    def refs(si, s, x, feats, truth):
        if (si, s) not in ref_cache:
            r = {"real": matched(block(feats[:, :s], x, s), truth) + 1}
            for m in args.ms:
                r[f"rem{m}"] = matched(block(feats[:, : s - m], x, s, m), truth) + 1
                real_tap = feats[0, s - m : s].view(m, nT, H)[:, tix]
                r[f"tap{m}"] = matched(block(one_tap(feats, s, m, real_tap), x, s), truth) + 1
            ref_cache[(si, s)] = r
        return ref_cache[(si, s)]

    @torch.no_grad()
    def evaluate(row, stack):
        tau = {m: [] for m in args.ms}
        ref = {k: [] for k in ["real"] + [f"{p}{m}" for m in args.ms for p in ("rem", "tap")]}
        cs = []
        for n_done, (si, anchors) in enumerate(by_seq.items()):
            ids, _lo = rest[si]
            x = ids.to(dev)
            ctx = seq_ctx(ids, stack)
            hs = ctx[0]
            feats = extract_context_feature(hs, lids)
            truth_all = x.tolist()
            for s in anchors:
                truth = truth_all[s + 1 :]
                r = refs(si, s, x, feats, truth)
                for k, v in r.items():
                    ref[k].append(v)
                for m in args.ms:
                    out, _ = run(stack, ctx, s - m, s)
                    tau[m].append(matched(block(one_tap(feats, s, m, out[0]), x, s), truth) + 1)
                    if m == M:
                        cs.append(F.cosine_similarity(out[0].float() - mu[stop],
                                                      hs[stop][0, s - m : s].float() - mu[stop], dim=-1).mean().item())
            del hs, feats, ctx
            if n_done % 100 == 0:
                print(f"[eval {row['config']}] {n_done}/{len(by_seq)} sequences, {time.time() - t0:.0f}s", flush=True)
        out = dict(row)
        out["cos"] = mean(cs)
        for m in args.ms:
            rem, tap = mean(ref[f"rem{m}"]), mean(ref[f"tap{m}"])
            out[f"tau m{m}"] = mean(tau[m])
            out[f"of 1 tap m{m}"] = (mean(tau[m]) - rem) / max(tap - rem, 1e-9)
        refrow = {"all taps real": mean(ref["real"])}
        for m in args.ms:
            refrow[f"removed m{m}"] = mean(ref[f"rem{m}"])
            refrow[f"real tap m{m}"] = mean(ref[f"tap{m}"])
        print_table([out], list(out), "sub-stack")
        return out, refrow

    rows, refrow = [], {}
    for shape, a, kind in cfgs:
        torch.cuda.empty_cache()
        kept = layers_of(shape, a, args.budget, c, L)
        stack = BandStack(target, kept, stop, kind, args.rank, args.ctx_win)
        name = f"{shape}:{a}:{kind}"
        desc = ",".join(map(str, kept))
        row = {"config": name, "layers": desc, "bridged": str(stack.spans), "cost": stack.cost()}
        print(f"[{name}] layers {desc}, bridged spans {stack.spans}, cost {row['cost']:.2f} layers, "
              f"{time.time() - t0:.0f}s", flush=True)
        fit_affine(stack)
        stack.eval()
        if kind in ("mlp", "attn", "ctx") and stack.spans:
            stack.train()
            train(name, stack)
        r, refrow = evaluate(row, stack)
        rows.append(r)
        del stack

    print_table([refrow], list(refrow), "References (tau): all taps real / no features for the last m / only the 3/4 tap real")
    print_table(rows, list(rows[0]), f"Tap of layer {c} produced by a sub-stack of {args.budget} layers (held-out anchors)")
    save_json(args.save, {"args": vars(args), "tap_layer": c, "rows": rows, "refs": refrow, "similarity": sim_rows})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
