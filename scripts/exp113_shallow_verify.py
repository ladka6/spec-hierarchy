"""Experiment 113: what should a SHALLOW VERIFY look like? Same budget, different shapes, measured by f.

A chain of blocks before the one full verify needs, between two blocks, what a real verify gives: the position of
the drafter's first error (the cut) and the target's token there (the fix). With f = P(the chain survives a cut),
B blocks give about L (1 - f^B) / (1 - f) drafted tokens, so f decides how deep a chain pays. exp112 optimised the
FEATURES of the first k layers (share 0.67-0.73 at k = 4) while f stayed at 0.2-0.4. Here the shallow verify is
trained and scored on its DECISION, and the question is which part of the target it should run.

Shallow verify = a sub-stack of the frozen target's layers run over the in-flight tokens only. Every kept layer
attends to its own ORIGINAL key / value cache of the verified prefix (exact, as the last full verify wrote it) and
to the in-flight tokens. It ends in the target's own final norm and lm_head, so its output is a next-token
distribution in the target's output space; first disagreement with the draft = cut, its token there = fix.

Shapes at a budget of n layers (L = number of target layers):
  first     layers 0 .. n-1, the removed tail replaced by one bridge                    (early-exit like, Kangaroo)
  skip      n layers spread evenly from the first to the last, bridges in between        (Draft & Verify / SWIFT)
  sandwich  first n/2 and last n/2 layers, one bridge over the middle
  attn      attention sublayers only (they read the real prefix cache at every depth), each MLP replaced by a small
            low-rank surrogate; as many layers as the budget pays for, spread evenly (n = 8: about all of them)
Bridge = affine map fitted in closed form on real activations (ridge, hspec.bridges form) + residual MLP, trained.
--rank r > 0 adds LoRA (rank r) on the linears of the kept layers, draft mode only (the verify is untouched).

Methods: fit    closed-form bridges only, nothing trained (not for attn)
         train  bridges / surrogates / LoRA trained with KL(target || shallow) at the in-flight positions
                + --rank-w * listwise CE over the drafter's top --cands candidates (first-error positions x --fork-w)

Two readouts of the same pass:
  open  argmax over the whole vocabulary (the shallow verify alone)
  cand  argmax over the drafter's top --cands of lm_head(shallow)[c] + alpha * log q_drafter(c)   (it only has to
        choose among a few candidates; the target's token is among them at 80-99% of the first errors)

Table 1, decision (held-out anchors of exp112, the pass runs over the drafted block up to its first error):
  agree        open top-1 = the target's token, all in-flight positions before the first error
  fork top-1/3 open readout at the first error
  cand recall  target's token among the candidates at the first error;  2nd = it is the drafter's second choice
  pick@fork    cand readout right at the first error (location given)
  f            first disagreement exactly at the first error AND the token right       (per readout)
  false alarm  disagreement with a correct drafted token before the first error, all anchors   (per readout)
  wrong fix / missed   at the first error: other wrong token / agrees with the wrong draft    (cand readout)
Table 2, chain of two blocks in accepted drafted tokens (the redraft starts after the fixed token):
  one block    the drafter alone
  2 blk        false alarm -> tokens before it; resolved -> first error + 1 + redraft; otherwise one block.
               "real" = redraft with real features for the in-flight tokens (isolates the decision),
               "none" = redraft without features for them (tokens only)
  ideal        an exact verify between the blocks (f = 1), same two feature settings
Table 3, operating points of the cand readout: it disagrees with the drafted token only if its best candidate's
  score is more than --margins above the drafted token's (0 = plain argmax). Trades false alarms against missed
  errors; f, false alarm, missed and the two-block chain per margin (no selection, the whole curve is reported).

  python scripts/exp113_shallow_verify.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
      --data /scratch-shared/$USER/hspec_lagft/data --shapes first skip sandwich attn --budgets 4 8
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
from exp107_lift_diag import load_seqs  # noqa: E402
from exp108b_nonlinear_lift import cosine_lr  # noqa: E402
from exp112_draft_branch import ATTN_LIN, BF, MLP_LIN, rot  # noqa: E402

from hspec.bridges import Bridge, spans_of  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

SHAPES = ("first", "skip", "sandwich", "attn")


def spread(n, L):
    return sorted({round(j * (L - 1) / max(n - 1, 1)) for j in range(n)})


def layers_of(shape, n, L, attn_frac):
    """kept layers and the subset that runs its attention sublayer only"""
    if shape == "first":
        return list(range(n)), set()
    if shape == "skip":
        return spread(n, L), set()
    if shape == "sandwich":
        a = n // 2
        return list(range(a)) + list(range(L - (n - a), L)), set()
    kept = spread(min(L, max(2, round(n / attn_frac))), L)
    return kept, set(kept)


class Sur(nn.Module):
    """low-rank stand-in for one MLP sublayer (input: the layer's own post-attention norm), starts at zero"""

    def __init__(self, H, r):
        super().__init__()
        self.gate = nn.Linear(H, r, bias=False)
        self.up = nn.Linear(H, r, bias=False)
        self.down = nn.Linear(r, H, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, h):
        dt = h.dtype
        h = h.float()
        return self.down(F.silu(self.gate(h)) * self.up(h)).to(dt)


class SubStack(nn.Module):
    """Some of the target's decoder layers (frozen, shared) over the in-flight positions, bridges over the rest."""

    def __init__(self, target, kept, attn_only, bridge_rank, sur_rank, lora_rank, bridges=True):
        super().__init__()
        cfg = target.config
        self.L = len(target.model.layers)
        self.kept, self.kset, self.attn_only = sorted(kept), set(kept), set(attn_only)
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nh
        self._orig = list(target.model.layers)              # plain lists: the target is not a submodule
        self._norm = [target.model.norm]
        H, dev = cfg.hidden_size, target.lm_head.weight.device
        self.spans = spans_of(self.kept, self.L) if bridges else []
        self.tail = bool(self.spans) and self.spans[-1][1] == self.L - 1
        self.bridges = nn.ModuleDict({str(b): Bridge(H, "mlp", bridge_rank) for _, b in self.spans}).to(dev)
        self.sur = nn.ModuleDict({str(g): Sur(H, sur_rank) for g in sorted(self.attn_only)}).to(dev)
        self.lora = nn.ParameterDict()
        self.scale = 2.0
        if lora_rank:
            for g in self.kept:
                lay = self._orig[g]
                groups = [(lay.self_attn, ATTN_LIN)] + ([] if g in self.attn_only else [(lay.mlp, MLP_LIN)])
                for parent, names in groups:
                    for nm in names:
                        lin = getattr(parent, nm)
                        self.lora[f"A{g}_{nm}"] = nn.Parameter(
                            torch.randn(lora_rank, lin.in_features, device=dev) / math.sqrt(lin.in_features))
                        self.lora[f"B{g}_{nm}"] = nn.Parameter(torch.zeros(lin.out_features, lora_rank, device=dev))

    def cost(self):
        """weights read per pass, in units of one full target layer (trained parts counted at the same precision)"""
        full = sum(p.numel() for p in self._orig[0].parameters())
        attn = sum(p.numel() for p in self._orig[0].self_attn.parameters())
        n = sum(attn if g in self.attn_only else full for g in self.kept)
        return (n + sum(p.numel() for p in self.parameters())) / full

    def lin(self, g, parent, nm, x):
        y = getattr(parent, nm)(x)
        if f"A{g}_{nm}" in self.lora:
            d = (x.float() @ self.lora[f"A{g}_{nm}"].T) @ self.lora[f"B{g}_{nm}"].T * self.scale
            y = y + d.to(y.dtype)
        return y

    def _kv(self, attn, hn, cos, sin, lin):
        B, T, _ = hn.shape
        kk = lin("k_proj", hn).view(B, T, -1, self.hd)
        if hasattr(attn, "k_norm"):
            kk = attn.k_norm(kk)
        kk = rot(kk.transpose(1, 2).to(BF), cos, sin)
        vv = lin("v_proj", hn).view(B, T, -1, self.hd).transpose(1, 2).to(BF)
        return kk, vv

    @torch.no_grad()
    def prefix_kv(self, hs, cos, sin):
        """the ORIGINAL post-rotary keys / values of every kept layer over the whole sequence (the verify's cache)"""
        out = {}
        for g in self.kept:
            lay = self._orig[g]
            hn = lay.input_layernorm(hs[g])
            out[g] = self._kv(lay.self_attn, hn, cos, sin, lambda nm, x, a=lay.self_attn: getattr(a, nm)(x))
        return out

    def forward(self, x0, a, kvp, cos, sin):
        """x0 [1, m, H] embeddings of positions a .. a+m-1. Returns the final state (after the final norm) [1, m, H]."""
        m = x0.shape[1]
        mask = torch.ones(m, a + m, dtype=torch.bool, device=x0.device)
        mask[:, a:] = torch.tril(mask[:, a:])
        rep = self.nh // self.nkv
        x = x0
        for g in range(self.L):
            if g not in self.kset:
                if str(g) in self.bridges:                   # last layer of a removed span
                    x = self.bridges[str(g)](x)
                continue
            lay = self._orig[g]
            attn = lay.self_attn
            hn = lay.input_layernorm(x)
            q = self.lin(g, attn, "q_proj", hn).view(1, m, -1, self.hd)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            q = rot(q.transpose(1, 2).to(BF), cos, sin)
            kn, vn = self._kv(attn, hn, cos, sin, lambda nm, t, g=g, attn=attn: self.lin(g, attn, nm, t))
            Kp, Vp = kvp[g]
            K = torch.cat([Kp[:, :, :a], kn], 2).repeat_interleave(rep, 1)
            V = torch.cat([Vp[:, :, :a], vn], 2).repeat_interleave(rep, 1)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            o = o.transpose(1, 2).reshape(1, m, -1)
            x = x + self.lin(g, attn, "o_proj", o)
            h2 = lay.post_attention_layernorm(x)
            if g in self.attn_only:
                x = x + self.sur[str(g)](h2)
            else:
                mlp = lay.mlp
                x = x + self.lin(g, mlp, "down_proj",
                                 mlp.act_fn(self.lin(g, mlp, "gate_proj", h2)) * self.lin(g, mlp, "up_proj", h2))
        # a removed tail is bridged straight to the target's last hidden state, which already has the final norm
        return x if self.tail else self._norm[0](x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--budgets", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--methods", nargs="+", default=["fit", "train"])
    ap.add_argument("--bridge-rank", type=int, default=1024)
    ap.add_argument("--sur-rank", type=int, default=256)
    ap.add_argument("--rank", type=int, default=0, help="LoRA rank on the kept layers (0 = frozen layers)")
    ap.add_argument("--kl-w", type=float, default=1.0)
    ap.add_argument("--rank-w", type=float, default=1.0)
    ap.add_argument("--cands", type=int, default=16)
    ap.add_argument("--fork-w", type=float, default=3.0)
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
                    help="cand readout: disagree with the draft only if the best candidate beats it by this much")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--blocks", type=int, default=8, help="anchors per sequence per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=5e-5)
    ap.add_argument("--lr-alpha", type=float, default=1e-2)
    ap.add_argument("--val-seqs", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--fit-seqs", type=int, default=150, help="sequences for the closed-form bridges")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--fit-tokens", type=int, default=600_000, help="as exp112: fixes the train / held-out split")
    ap.add_argument("--val-tokens", type=int, default=15_000)
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--save", default="exp113_shallow_verify")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    assert all(s in SHAPES for s in args.shapes) and all(m in ("fit", "train") for m in args.methods)
    dev, t0 = "cuda", time.time()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    rng = random.Random(0)

    seqs = load_seqs(args)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    head = target.lm_head
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H, L = target.config.hidden_size, len(target.model.layers)
    lay0 = target.model.layers[0]
    attn_frac = sum(p.numel() for p in lay0.self_attn.parameters()) / sum(p.numel() for p in lay0.parameters())
    C = args.cands
    print(f"[setup] {len(seqs)} sequences, {L} layers, block {bs}, attention = {attn_frac:.2f} of a layer's weights",
          flush=True)

    # ---- the split of exp112 (same train / validation sequences, same held-out anchors) ----
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
    print(f"[data] train {len(fit_seqs)} seqs, val {len(val_seqs)}, held-out {len(rest)} "
          f"({sum(len(v) for v in by_seq.values())} anchors in {len(by_seq)} seqs)", flush=True)

    @torch.no_grad()
    def seq_ctx(ids, sub):
        hs = target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        pos = torch.arange(len(ids), device=dev)[None]
        cos, sin = target.model.rotary_emb(hs[0], pos)
        with torch.autocast("cuda", dtype=BF):
            kvp = sub.prefix_kv(hs, cos, sin)
        return hs, cos, sin, kvp

    def run(sub, ctx, a, b):
        """final states of the shallow verify at positions a .. b-1 (true tokens as input) [b-a, H]"""
        hs, cos, sin, kvp = ctx
        return sub(hs[0][:, a:b], a, kvp, cos[:, a:b], sin[:, a:b])[0]

    def block_lg(feats, x, s, lag=0):
        return blocks_logits(draft, target, feats, x, [(s, lag)], bs)[0]

    def mean(v):
        return sum(v) / max(len(v), 1)

    # ---- start-up check: all layers kept, nothing trained -> the target's own last hidden state ----
    with torch.no_grad(), torch.autocast("cuda", dtype=BF):
        ids, _lo = seqs[0]
        full = SubStack(target, list(range(L)), set(), 8, 8, 0)
        ctx = seq_ctx(ids, full)
        a = len(ids) - 16
        z = run(full, ctx, a, len(ids))
        ref = ctx[0][-1][0, a:]
        cs = F.cosine_similarity(z.float(), ref.float(), dim=-1).mean().item()
        ag = (head(z.to(ref.dtype)).argmax(-1) == head(ref).argmax(-1)).float().mean().item()
        print(f"[check] all {L} layers: cosine(final state) {cs:.4f}, same argmax {ag:.3f}", flush=True)
        assert cs > 0.99, "the sub-stack forward does not reproduce the target"
        del full, ctx, z, ref

    @torch.no_grad()
    def fit_affine(sub):
        """closed-form affine part of every bridge: ridge regression of the removed span's update on its input,
        rows weighted by 1 / mean(x^2) (hspec.bridges.fit_linear_, independent variant)"""
        if not sub.spans:
            return
        acc = {b: [torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev),
                   torch.zeros(H + 1, H, dtype=torch.float64, device=dev)] for _, b in sub.spans}
        for si in fit_seqs[: args.fit_seqs]:
            ids, _lo = seqs[si]
            hs = target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
            for a, b in sub.spans:
                x = hs[a][0].double()
                d = hs[b + 1][0].double() - x
                w = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                xa = torch.cat([x, torch.ones_like(x[:, :1])], -1) * w
                acc[b][0] += xa.T @ xa
                acc[b][1] += xa.T @ (d * w)
            del hs
        for _, b in sub.spans:
            xtx, xtd = acc[b]
            lam = args.ridge * xtx.diagonal().mean()
            sol = torch.linalg.solve(xtx + lam * torch.eye(H + 1, dtype=xtx.dtype, device=dev), xtd)
            m = sub.bridges[str(b)]
            m.lin.weight.copy_(sol[:H].T.float())
            m.lin.bias.copy_(sol[H].float())
        del acc

    def cand_scores(sub, lg, Lr):
        """scores [n, C] of the drafter's top-C candidates and the candidates [n, C] (drafter order)"""
        cand = Lr.topk(C, -1).indices
        return lg.gather(1, cand) + sub.alpha * F.log_softmax(Lr, -1).gather(1, cand), cand

    def loss_fn(sub, si, n_blocks, grad):
        ids, lo = seqs[si]
        T = len(ids)
        if T - bs - max(lo, 18) < 1:
            return None
        x = ids.to(dev)
        ctx = seq_ctx(ids, sub)
        hs = ctx[0]
        feats = extract_context_feature(hs, lids)
        kl_t, rk_t, n = 0.0, 0.0, 0
        with torch.autocast("cuda", dtype=BF), torch.set_grad_enabled(grad):
            for _ in range(n_blocks):
                s = rng.randrange(max(lo, 18), T - bs)
                lg = head(run(sub, ctx, s, s + bs).to(hs[-1].dtype)).float()
                with torch.no_grad():
                    tl = head(hs[-1][0, s : s + bs]).float()
                    Lr = block_lg(feats[:, :s], x, s).float()
                kl = F.kl_div(F.log_softmax(lg, -1), F.softmax(tl, -1), reduction="batchmean")
                sc, cand = cand_scores(sub, lg[: bs - 1], Lr)
                truth = x[s + 1 : s + bs]
                hit = cand == truth[:, None]
                ok = hit.any(1)
                rk = lg.new_zeros(())
                if ok.any():
                    w = 1.0 + (args.fork_w - 1.0) * (cand[:, 0] != truth).float()
                    ce = F.cross_entropy(sc[ok], hit[ok].float().argmax(1), reduction="none")
                    rk = (ce * w[ok]).sum() / w[ok].sum()
                kl_t, rk_t, n = kl_t + kl, rk_t + rk, n + 1
        return kl_t / n, rk_t / n

    def train(tag, sub):
        groups = [{"params": list(sub.bridges.parameters()) + list(sub.sur.parameters()), "base": args.lr},
                  {"params": list(sub.lora.parameters()), "base": args.lr_lora},
                  {"params": [sub.alpha], "base": args.lr_alpha}]
        groups = [g for g in groups if g["params"]]
        params = [p for g in groups for p in g["params"]]
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)

        def scorer():
            sub.eval()
            st = rng.getstate()
            rng.seed(123)
            tot, n = 0.0, 0
            with torch.no_grad():
                for si in val_seqs:
                    out = loss_fn(sub, si, 4, False)
                    if out is not None:
                        tot += args.kl_w * out[0].item() + args.rank_w * out[1].item()
                        n += 1
            rng.setstate(st)
            sub.train()
            return -tot / max(n, 1)

        def snap():
            return {n: v.detach().cpu().clone() for n, v in sub.state_dict().items()}

        best_sc, best_state, bad, log = scorer(), snap(), 0, []
        print(f"[{tag}] start val {best_sc:.4f}, trainable {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
        for step in range(args.steps):
            for g in opt.param_groups:
                g["lr"] = g["base"] * cosine_lr(step, args.steps, min(200, max(args.steps // 10, 1)))
            out = None
            for _ in range(10):
                out = loss_fn(sub, rng.choice(fit_seqs), args.blocks, True)
                if out is not None:
                    break
            kl, rk = out
            loss = args.kl_w * kl + args.rank_w * rk
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                sc = scorer()
                log.append((step + 1, kl.item(), rk.item(), sc))
                print(f"[{tag}] step {step + 1} kl {kl.item():.4f} rank {rk.item():.4f} val {sc:.4f} "
                      f"alpha {float(sub.alpha):.2f} {time.time() - t0:.0f}s", flush=True)
                if sc > best_sc:
                    best_sc, best_state, bad = sc, snap(), 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        print(f"[{tag}] early stop at step {step + 1}", flush=True)
                        break
        sub.load_state_dict(best_state)
        sub.eval()
        return log

    # ---- per anchor, independent of the shallow verify: the drafter's block, its candidates, the redrafts ----
    anchor_cache = {}

    @torch.no_grad()
    def anchor_info(si, s, x, feats, truth_all):
        if (si, s) not in anchor_cache:
            Lr = block_lg(feats[:, :s], x, s).float()
            chain = Lr.argmax(-1).tolist()
            e = matched(chain, truth_all[s + 1 :])
            last = min(e, len(chain) - 1)
            top = Lr[: last + 1].topk(C, -1)
            lq = F.log_softmax(Lr[: last + 1], -1).gather(1, top.indices)
            red = None
            if e < len(chain) and s + 2 * bs < len(truth_all):
                a2 = s + e + 1                               # the fixed token is the new anchor
                tr = truth_all[a2 + 1 :]
                red = (matched(block_lg(feats[:, :a2], x, a2).argmax(-1).tolist(), tr),
                       matched(block_lg(feats[:, :s], x, a2, e + 1).argmax(-1).tolist(), tr))
            anchor_cache[(si, s)] = (chain, e, top.indices.cpu(), lq.cpu(), red)
        return anchor_cache[(si, s)]

    @torch.no_grad()
    def evaluate(row, sub):
        agree, fk3 = [], [0, 0, 0]
        st = {r: {n: 0 for n in ("res", "fa", "wrong", "miss")} for r in ("open", "cand")}
        ch = {n: [] for n in ("one", "open_real", "open_none", "cand_real", "cand_none", "ideal_real", "ideal_none")}
        n_err = n_all = recall = second = pick_ok = 0
        op = {t: {"res": 0, "fa": 0, "miss": 0, "real": [], "none": []} for t in args.margins}
        for n_done, (si, anchors) in enumerate(by_seq.items()):
            ids, _lo = rest[si]
            x = ids.to(dev)
            ctx = seq_ctx(ids, sub)
            hs = ctx[0]
            feats = extract_context_feature(hs, lids)
            truth_all = x.tolist()
            with torch.autocast("cuda", dtype=BF):
                for s in anchors:
                    chain, e, cand, lq, red = anchor_info(si, s, x, feats, truth_all)
                    truth = truth_all[s + 1 :]
                    last = min(e, len(chain) - 1)
                    err = e < len(chain)
                    lg = head(run(sub, ctx, s, s + last + 1).to(hs[-1].dtype)).float()
                    cand = cand.to(dev)
                    sc = lg.gather(1, cand) + sub.alpha * lq.to(dev)
                    picks = {"open": lg.argmax(-1).tolist(),
                             "cand": cand.gather(1, sc.argmax(1, keepdim=True))[:, 0].tolist()}
                    n_ok = e if err else last + 1            # positions where the drafted token is right
                    agree.extend(picks["open"][i] == truth[i] for i in range(n_ok))
                    n_all += 1
                    two = {}
                    for r, pick in picks.items():
                        dis = [i for i in range(last + 1) if pick[i] != chain[i]]
                        j = dis[0] if dis else -1
                        fa = 0 <= j < e if err else j >= 0
                        res = err and j == e and pick[e] == truth[e]
                        st[r]["fa"] += fa
                        st[r]["res"] += res
                        st[r]["wrong"] += err and j == e and pick[e] != truth[e]
                        st[r]["miss"] += err and j == -1
                        two[r] = (fa, res, j)
                    gap = (sc.max(1).values - sc[:, 0]).tolist()        # candidate 0 = the drafted token
                    for t, o in op.items():
                        dis = [i for i in range(last + 1) if gap[i] > t]
                        j = dis[0] if dis else -1
                        fa = 0 <= j < e if err else j >= 0
                        res = err and j == e and picks["cand"][e] == truth[e]
                        o["fa"] += fa
                        o["res"] += res
                        o["miss"] += err and j == -1
                        if s + 2 * bs < len(ids):
                            o["real"].append(j if fa else e + 1 + red[0] if res else e)
                            o["none"].append(j if fa else e + 1 + red[1] if res else e)
                    if err:
                        n_err += 1
                        recall += truth[e] in cand[e].tolist()
                        second += int(cand[e, 1]) == truth[e]
                        pick_ok += picks["cand"][e] == truth[e]
                        t3 = lg[e].topk(3).indices.tolist()
                        fk3[0] += t3[0] == truth[e]
                        fk3[1] += truth[e] in t3
                        fk3[2] += 1
                    if s + 2 * bs < len(ids):
                        ch["one"].append(e)
                        for q, feat in enumerate(("real", "none")):
                            ch[f"ideal_{feat}"].append(e + 1 + red[q] if err else e)
                            for r in picks:
                                fa, res, j = two[r]
                                ch[f"{r}_{feat}"].append(j if fa else e + 1 + red[q] if res else e)
            del hs, feats, ctx
            if n_done % 50 == 0:
                print(f"[eval {row['shape']} n={row['budget']} {row['method']}] {n_done}/{len(by_seq)} sequences, "
                      f"{time.time() - t0:.0f}s", flush=True)
        ne = max(n_err, 1)
        dec = dict(row)
        dec.update({"agree": mean(agree), "fork top-1": fk3[0] / max(fk3[2], 1), "fork top-3": fk3[1] / max(fk3[2], 1),
                    "cand recall": recall / ne, "2nd": second / ne, "pick@fork": pick_ok / ne,
                    "f open": st["open"]["res"] / ne, "f cand": st["cand"]["res"] / ne,
                    "false alarm open": st["open"]["fa"] / max(n_all, 1),
                    "false alarm cand": st["cand"]["fa"] / max(n_all, 1),
                    "wrong fix": st["cand"]["wrong"] / ne, "missed": st["cand"]["miss"] / ne,
                    "alpha": float(sub.alpha), "errors": n_err})
        chn = dict(row)
        chn.update({"anchors": len(ch["one"]), "one block": mean(ch["one"]),
                    "2 blk open real": mean(ch["open_real"]), "2 blk cand real": mean(ch["cand_real"]),
                    "ideal real": mean(ch["ideal_real"]),
                    "2 blk open none": mean(ch["open_none"]), "2 blk cand none": mean(ch["cand_none"]),
                    "ideal none": mean(ch["ideal_none"])})
        ops = [{**row, "margin": t, "f": o["res"] / ne, "false alarm": o["fa"] / max(n_all, 1),
                "missed": o["miss"] / ne, "one block": mean(ch["one"]), "2 blk real": mean(o["real"]),
                "2 blk none": mean(o["none"]), "ideal real": mean(ch["ideal_real"])} for t, o in op.items()]
        print_table([dec], list(dec), "decision")
        print_table([chn], list(chn), "chain of two blocks (accepted drafted tokens)")
        print_table(ops, list(ops[0]), "cand readout by margin")
        return dec, chn, ops

    dec_rows, chain_rows, op_rows, train_log = [], [], [], {}
    for n in args.budgets:
        for shape in args.shapes:
            torch.cuda.empty_cache()
            kept, attn_only = layers_of(shape, n, L, attn_frac)
            sub = SubStack(target, kept, attn_only, args.bridge_rank, args.sur_rank, args.rank,
                           bridges=shape != "attn")
            sub.alpha = nn.Parameter(torch.ones((), device=dev))
            desc = ("attn of " if attn_only else "") + (
                f"{kept[0]}..{kept[-1]} ({len(kept)})" if len(kept) > 10 else ",".join(map(str, kept)))
            row = {"shape": shape, "budget": n, "layers": desc, "cost": sub.cost()}
            print(f"[{shape} n={n}] layers {desc}, bridged spans {sub.spans}, cost {row['cost']:.2f} layers, "
                  f"{time.time() - t0:.0f}s", flush=True)
            fit_affine(sub)
            sub.eval()
            if "fit" in args.methods and shape != "attn":
                d, c, o = evaluate({**row, "method": "fit"}, sub)
                dec_rows.append(d)
                chain_rows.append(c)
                op_rows.extend(o)
            if "train" in args.methods:
                tag = f"{shape} n={n}"
                sub.train()
                train_log[f"{shape}_n{n}"] = train(tag, sub)
                d, c, o = evaluate({**row, "method": "train" + (f"+lora{args.rank}" if args.rank else "")}, sub)
                dec_rows.append(d)
                chain_rows.append(c)
                op_rows.extend(o)
            del sub

    print_table(dec_rows, list(dec_rows[0]), "Shallow verify: decision at the drafter's first error (held-out anchors)")
    print_table(chain_rows, list(chain_rows[0]), "Shallow verify: chain of two blocks (accepted drafted tokens)")
    print_table(op_rows, list(op_rows[0]), "Shallow verify: cand readout by margin (false alarms against missed errors)")
    save_json(args.save, {"args": vars(args), "decision": dec_rows, "chain": chain_rows, "margins": op_rows,
                          "train_log": train_log})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
