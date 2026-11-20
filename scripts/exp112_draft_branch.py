"""Experiment 112: can a draft-only change INSIDE the first k layers beat reading the frozen h_k?

exp108b (Qwen3-8B, frozen drafter): readers of the frozen h_k reach a share of 0.40 (ridge) / 0.48 (mlp) /
0.60 (dthru) at k = 4 and 0.51 / 0.58 / 0.65 at k = 8. LayerSkip's exit training (weights changed, next-token
objective) gives 0.60 / 0.85 with the linear reader. Question: with the target's weights untouched for
verification, how far does a DRAFT-MODE version of the first k layers get when it is trained for the drafter?

Draft branch: the in-flight positions a .. s-1 are run through k layers that start as the target's first k layers.
Verified positions < a keep the target's ORIGINAL key / value cache (computed here from the real hidden states with
the original weights), so the verify pass and everything it wrote stay exact; only the in-flight rows differ.
Own forward for the decoder layers (Llama / Qwen3 layout: pre-norm, GQA, rotary, optional q_norm / k_norm), no
HF cache objects. A start-up check compares the untrained branch with the target's real h_k (must agree).

Methods (one shared protocol, each trained separately):
  ridge   frozen branch, closed-form map                       (reference, = exp108b ridge)
  mlp     frozen branch, MSE-trained residual MLP              (reference, = exp108b mlp)
  read    frozen branch, MLP trained through the drafter       (= exp108b dthru; the reader-only level)
  lora    LoRA (rank --rank) on all linears of the k branch layers + the MLP head, through the drafter
  copy    full trainable fp32 copy of the k layers + the MLP head, through the drafter
--kv-map: which ORIGINAL layer's verified-prefix cache branch layer j attends to
  own     layer j                (the real cache of the same layer)
  half    first k/2 layers own, the rest spread up to the last layer   (pulls deep verified context down)
  spread  spread over all layers from 0 to the last
Head: the drafter's missing taps (+ the final state, for the probe) = ridge(z_k) + MLP(z_k), z_k the branch output;
taps of layers <= k are read from the branch's own intermediate states.
Loss: --ce-w * drafter block CE (DFlash recipe, as exp108b dthru) + --mse-w * standardised MSE to the real states
      + --kl-w * KL(target's next-token distribution || lm_head(predicted final state)) at the in-flight positions
      (trains the fork resolver; 0 = off).
--rank-w > 0: fork reranker (docs/method_fork_reranker.md, on the draft branch instead of a copy slice). The final
  state only has to choose among the drafter's top --cands candidates: score = lm_head(final)[c] + alpha * log q(c),
  trained with a listwise CE over the candidates at the in-flight positions (positions where the drafter's first
  choice is wrong weighted by --fork-w; the others teach it not to raise false alarms).
Fork columns of the table (one partial pass over the drafted block, first disagreement with the draft = fork):
  cand recall   the target's token is among the drafter's candidates at its first error (ceiling)
  2nd           it is the drafter's second choice (what a rule without target computation gets, location given)
  rr top-1      reranker's choice is right at the first error (location given)
  resolved      the first disagreement is exactly at the first error AND the choice is right (= c of the cost model)
  false alarm   the reranker disagrees with a correct drafted token before the first error (all anchors)
--draft-rank r > 0: the drafter is no longer frozen for the trained methods: LoRA of rank r on all its linears,
  trained with the same loss (the drafter learns to use the provisional rows). The table then also gives
  "real adapted" = tau of the adapted drafter on REAL features; real / removed / share keep the original drafter
  as reference, so shares stay comparable with the frozen-drafter rows.

Reports per (k, method), exp107 / exp108 protocol on held-out sequences, frozen drafter:
  share   (tau - removed) / (real - removed) with the last m context positions from the branch
  probe   top-1 of lm_head(predicted final state) at in-flight positions vs the target's token
  forks   top-1 / top-3 at the drafter's first error (partial pass over the drafted block up to the error)

  python scripts/exp112_draft_branch.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
      --data /scratch-shared/$USER/hspec_lagft/data --methods read lora --exits 4 8
"""

from __future__ import annotations

import argparse
import copy
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
from exp107_lift_diag import Ridge, apply_map, load_seqs  # noqa: E402
from exp108b_nonlinear_lift import MLPReader, cosine_lr, pack_sub  # noqa: E402

from hspec.lagtrain import _weighted_ce  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

METHODS = ("ridge", "mlp", "read", "lora", "copy")
TRAINED = ("read", "lora", "copy")
ATTN_LIN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_LIN = ("gate_proj", "up_proj", "down_proj")
BF = torch.bfloat16


class DLoRA(nn.Module):
    """LoRA on one drafter linear, switched globally (off = the original drafter)."""

    on = False

    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.reset()

    def reset(self):
        nn.init.normal_(self.A, std=1 / math.sqrt(self.base.in_features))
        nn.init.zeros_(self.B)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        if DLoRA.on:
            y = y + ((x.float() @ self.A.T) @ self.B.T * 2.0).to(y.dtype)
        return y


def add_draft_lora(draft, rank, vocab):
    mods = []
    for parent in list(draft.modules()):
        for nm, ch in list(parent.named_children()):
            if isinstance(ch, nn.Linear) and ch.out_features < vocab:
                w = DLoRA(ch, rank)
                setattr(parent, nm, w)
                mods.append(w)
    return mods


def rot(x, cos, sin):
    """rotary on [B, heads, T, hd]; cos / sin [B, T, hd]"""
    cos, sin = cos[:, None].to(x.dtype), sin[:, None].to(x.dtype)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return x * cos + torch.cat((-x2, x1), -1) * sin


def kv_map(kind, k, L):
    if kind == "own":
        return list(range(k))
    if kind == "spread":
        return [round(j * (L - 1) / (k - 1)) for j in range(k)]
    h = k // 2
    return list(range(h)) + [round(h + (j - h) * (L - 1 - h) / (k - 1 - h)) for j in range(h, k)]


class Branch(nn.Module):
    """Draft-mode version of the target's first k decoder layers, run over in-flight positions only."""

    def __init__(self, target, k, method, rank, gmap):
        super().__init__()
        cfg = target.config
        self.k, self.method, self.gmap = k, method, gmap
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nh
        self._orig = list(target.model.layers)              # plain list: the target is not a submodule
        self.own = None
        self.lora = nn.ParameterDict()
        self.scale = 2.0
        if method == "copy":
            self.own = nn.ModuleList(copy.deepcopy(self._orig[j]).float() for j in range(k))
            for p in self.own.parameters():
                p.requires_grad_(True)
        elif method == "lora":
            for j in range(k):
                lay = self._orig[j]
                for parent, names in ((lay.self_attn, ATTN_LIN), (lay.mlp, MLP_LIN)):
                    for nm in names:
                        lin = getattr(parent, nm)
                        dev = lin.weight.device
                        self.lora[f"A{j}_{nm}"] = nn.Parameter(
                            torch.randn(rank, lin.in_features, device=dev) / math.sqrt(lin.in_features))
                        self.lora[f"B{j}_{nm}"] = nn.Parameter(torch.zeros(lin.out_features, rank, device=dev))

    def layer(self, j):
        return self.own[j] if self.own is not None else self._orig[j]

    def lin(self, j, parent, nm, x):
        y = getattr(parent, nm)(x)
        if f"A{j}_{nm}" in self.lora:
            d = (x.float() @ self.lora[f"A{j}_{nm}"].T) @ self.lora[f"B{j}_{nm}"].T * self.scale
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
        """the ORIGINAL post-rotary keys / values of every layer in gmap over the whole sequence (what the
        verify pass cached); sliced to the verified prefix per anchor."""
        out = {}
        for g in sorted(set(self.gmap)):
            lay = self._orig[g]
            hn = lay.input_layernorm(hs[g])
            out[g] = self._kv(lay.self_attn, hn, cos, sin, lambda nm, x, a=lay.self_attn: getattr(a, nm)(x))
        return out

    def forward(self, x0, a, kvp, cos, sin):
        """x0 [1, m, H] token embeddings of positions a .. a+m-1; cos / sin for those positions.
        Returns the residual stream after 0 .. k branch layers, each [1, m, H]."""
        m = x0.shape[1]
        mask = torch.ones(m, a + m, dtype=torch.bool, device=x0.device)
        mask[:, a:] = torch.tril(mask[:, a:])
        rep = self.nh // self.nkv
        x, zs = x0, [x0]
        for j in range(self.k):
            lay = self.layer(j)
            attn = lay.self_attn
            hn = lay.input_layernorm(x)
            q = self.lin(j, attn, "q_proj", hn).view(1, m, -1, self.hd)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            q = rot(q.transpose(1, 2).to(BF), cos, sin)
            kn, vn = self._kv(attn, hn, cos, sin, lambda nm, t, j=j, attn=attn: self.lin(j, attn, nm, t))
            Kp, Vp = kvp[self.gmap[j]]
            K = torch.cat([Kp[:, :, :a], kn], 2).repeat_interleave(rep, 1)
            V = torch.cat([Vp[:, :, :a], vn], 2).repeat_interleave(rep, 1)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            o = o.transpose(1, 2).reshape(1, m, -1)
            x = x + self.lin(j, attn, "o_proj", o)
            h2 = lay.post_attention_layernorm(x)
            mlp = lay.mlp
            x = x + self.lin(j, mlp, "down_proj",
                             mlp.act_fn(self.lin(j, mlp, "gate_proj", h2)) * self.lin(j, mlp, "up_proj", h2))
            zs.append(x)
        return zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--exits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--methods", nargs="+", default=["read", "lora"])
    ap.add_argument("--kv-map", default="own", choices=["own", "half", "spread"])
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--ce-w", type=float, default=1.0)
    ap.add_argument("--mse-w", type=float, default=0.1)
    ap.add_argument("--kl-w", type=float, default=0.0)
    ap.add_argument("--rank-w", type=float, default=0.0)
    ap.add_argument("--cands", type=int, default=16)
    ap.add_argument("--fork-w", type=float, default=3.0)
    ap.add_argument("--draft-rank", type=int, default=0)
    ap.add_argument("--lr-draft", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--blocks", type=int, default=8, help="anchors per sequence per step")
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-branch", type=float, default=0.0, help="0: 2e-4 for lora, 2e-5 for copy")
    ap.add_argument("--m-train", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--val-seqs", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--ms", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--fit-tokens", type=int, default=600_000)
    ap.add_argument("--val-tokens", type=int, default=15_000)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--mlp-width", type=int, default=2048)
    ap.add_argument("--mlp-steps", type=int, default=6000)
    ap.add_argument("--mlp-batch", type=int, default=4096)
    ap.add_argument("--mlp-lr", type=float, default=5e-4)
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--tie", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--save", default="exp112_draft_branch")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    assert all(m in METHODS for m in args.methods)
    dev, t0 = "cuda", time.time()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    rng = random.Random(0)
    M = max(args.ms)

    seqs = load_seqs(args)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    head = target.lm_head
    dmods = add_draft_lora(draft, args.draft_rank, target.config.vocab_size) if args.draft_rank else []
    if dmods:
        print(f"[setup] drafter LoRA r{args.draft_rank} on {len(dmods)} linears, "
              f"{sum(w.A.numel() + w.B.numel() for w in dmods) / 1e6:.1f}M", flush=True)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    hidx = [l + 1 for l in lids]
    H, nT, L = target.config.hidden_size, len(lids), len(target.model.layers)
    missing = {k: [j for j, h in enumerate(hidx) if h > k] for k in args.exits}
    tgts = {k: [f"tap{j}" for j in missing[k]] + ["final"] for k in args.exits}
    all_t = sorted({t for k in args.exits for t in tgts[k]}, key=lambda t: (t == "final", t))
    tpos = {t: all_t.index(t) for t in all_t}
    print(f"[setup] {len(seqs)} sequences, {L} layers, taps hs{hidx}, kv-map {args.kv_map}, "
          + ", ".join(f"k={k}: missing hs{[hidx[j] for j in v]}, prefix cache of layers {kv_map(args.kv_map, k, L)}"
                      for k, v in missing.items()), flush=True)

    def tgt_layer(t):
        return -1 if t == "final" else hidx[int(t[3:])]

    @torch.no_grad()
    def hidden(ids):
        return target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states

    @torch.no_grad()
    def seq_ctx(ids, branch):
        """real hidden states, rotary tables and the original prefix cache for one sequence"""
        hs = hidden(ids)
        pos = torch.arange(len(ids), device=dev)[None]
        cos, sin = target.model.rotary_emb(hs[0], pos)
        with torch.autocast("cuda", dtype=BF):
            kvp = branch.prefix_kv(hs, cos, sin)
        return hs, cos, sin, kvp

    # ---- start-up check: the untrained branch with its own cache must reproduce the target's real states ----
    with torch.no_grad(), torch.autocast("cuda", dtype=BF):
        ids, _lo = seqs[0]
        T = len(ids)
        for k in args.exits:
            br = Branch(target, k, "read", 0, kv_map("own", k, L))
            hs, cos, sin, kvp = seq_ctx(ids, br)
            a = T - 16
            zs = br(hs[0][:, a:], a, kvp, cos[:, a:], sin[:, a:])
            cs = [F.cosine_similarity(zs[j][0].float(), hs[j][0, a:].float(), dim=-1).mean().item()
                  for j in range(k + 1)]
            print(f"[check] k={k}: cosine(branch, real) per layer " + " ".join(f"{c:.4f}" for c in cs), flush=True)
            assert cs[-1] > 0.995, "the branch forward does not reproduce the target's first k layers"
            del hs, kvp, zs

    # ---- stage A (as exp108b): flat h_k / target states, ridge maps, MSE-trained residual MLP ----
    def collect(lo_i, n_tokens, cap_i):
        X = {k: torch.empty(n_tokens + args.max_len, H, dtype=BF) for k in args.exits}
        Y = torch.empty(n_tokens + args.max_len, len(all_t), H, dtype=BF)
        recs, n, i = [], 0, lo_i
        while n < n_tokens and i < cap_i:
            ids, lo = seqs[i]
            hs = hidden(ids)
            r = len(ids) - lo
            for k in args.exits:
                X[k][n : n + r] = hs[k][0, lo:].cpu()
            Y[n : n + r] = torch.stack([hs[tgt_layer(t)][0, lo:] for t in all_t], 1).cpu()
            recs.append(i)
            n += r
            i += 1
            del hs
        return {"X": {k: v[:n] for k, v in X.items()}, "Y": Y[:n], "recs": recs, "n": n}, i

    max_fit = int(0.6 * len(seqs))
    fit, i = collect(0, args.fit_tokens, max_fit)
    val, fit_end = collect(i, args.val_tokens, max_fit + (len(seqs) - max_fit) // 4)
    print(f"[data] fit {fit['n']} tokens / {len(fit['recs'])} seqs, val {val['n']} / {len(val['recs'])}, "
          f"{time.time() - t0:.0f}s", flush=True)
    assert val["n"] > 0, "no validation tokens: add data or lower --fit-tokens"
    fit_seqs, val_seqs = fit["recs"], val["recs"][: args.val_seqs]

    def chunks(n, size=16384):
        return [(a, min(a + size, n)) for a in range(0, n, size)]

    def moments(Tn):
        s, s2 = 0.0, 0.0
        for a, b in chunks(len(Tn)):
            v = Tn[a:b].to(dev).float()
            s = s + v.sum(0)
            s2 = s2 + (v * v).sum(0)
        mu = s / len(Tn)
        return mu, (s2 / len(Tn) - mu * mu).clamp_min(1e-6).sqrt()

    xstat = {k: moments(fit["X"][k]) for k in args.exits}
    ystat = moments(fit["Y"])

    # ---- held-out anchors (exp107 / exp108 protocol) ----
    rest = seqs[fit_end:]
    cand = [(si, s) for si, (ids, lo) in enumerate(rest) for s in range(max(lo, M + 2), len(ids) - bs)]
    random.Random(1).shuffle(cand)
    by_seq = {}
    for si, s in sorted(cand[: args.anchors]):
        by_seq.setdefault(si, []).append(s)

    def block_lg(fz, x, s, lag=0, adapted=False):
        DLoRA.on = adapted
        out = blocks_logits(draft, target, fz, x, [(s, lag)], bs)[0]
        DLoRA.on = False
        return out

    def run_block(fz, x, s, lag=0, adapted=False):
        return block_lg(fz, x, s, lag, adapted).argmax(-1).tolist()

    def rerank(mlp, fin_raw, dlog, dtype):
        """scores [n, K] of the drafter's top-K candidates and the candidates [n, K] (drafter order)"""
        lg = head(fin_raw.to(dtype)).float()
        dlog = dlog.float()
        cand = dlog.topk(args.cands, -1).indices
        return lg.gather(1, cand) + mlp.alpha * F.log_softmax(dlog, -1).gather(1, cand), cand

    def mean(v):
        return sum(v) / max(len(v), 1)

    rows_out, train_log = [], {}
    for k in args.exits:
        sel = torch.tensor([tpos[t] for t in tgts[k]])
        xm, xs = xstat[k]
        ym, ys = ystat[0][sel], ystat[1][sel]
        tk = tuple(tgts[k])
        n_out = len(tk)
        tapq = {int(t[3:]): q for q, t in enumerate(tk) if t != "final"}       # tap index -> head output
        fin = tk.index("final")
        gmap = kv_map(args.kv_map, k, L)

        acc = Ridge(H, tgts[k], H, dev)
        for a, b in chunks(fit["n"]):
            Xc = fit["X"][k][a:b].to(dev).float()
            Yc = fit["Y"][a:b, sel].to(dev).float()
            acc.add(Xc, {t: Yc[:, q] for q, t in enumerate(tk)})
        Xv = val["X"][k].to(dev).float()
        Yv = val["Y"][:, sel].to(dev).float()
        best = {}
        for lam in args.lams:
            for t, Wm in acc.solve(lam).items():
                q = tk.index(t)
                sc = F.cosine_similarity(apply_map(Wm, Xv) - ym[q], Yv[:, q] - ym[q], dim=-1).mean().item()
                if t not in best or sc > best[t][0]:
                    best[t] = (sc, lam, Wm)
        ridge = {t: best[t][2] for t in tk}
        del acc

        def ridge_std(X, ridge=ridge, ym=ym, ys=ys, tk=tk):
            with torch.autocast("cuda", enabled=False):
                X = X.float()
                return torch.stack([(apply_map(ridge[t], X) - ym[q]) / ys[q] for q, t in enumerate(tk)], 1)

        def val_ccos(mlp):
            mlp.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=BF):
                P = (ridge_std(Xv) + mlp((Xv - xm) / xs).float()) * ys + ym
            mlp.train()
            return F.cosine_similarity(P - ym, Yv - ym, dim=-1).mean().item()

        mlp0 = MLPReader(H, n_out, args.mlp_width, args.drop).to(dev)
        zero_mlp = copy.deepcopy(mlp0).eval()
        if any(m != "ridge" for m in args.methods):
            opt = torch.optim.AdamW(mlp0.parameters(), lr=args.mlp_lr, weight_decay=0.01)
            best_sc, best_state, bad = val_ccos(mlp0), copy.deepcopy(mlp0.state_dict()), 0
            for step in range(args.mlp_steps):
                for g in opt.param_groups:
                    g["lr"] = args.mlp_lr * cosine_lr(step, args.mlp_steps, 200)
                b = torch.randint(0, fit["n"], (args.mlp_batch,))
                X = fit["X"][k].index_select(0, b).to(dev).float()
                Ystd = (fit["Y"].index_select(0, b)[:, sel].to(dev).float() - ym) / ys
                with torch.autocast("cuda", dtype=BF):
                    pred = ridge_std(X) + mlp0((X - xm) / xs).float()
                loss = F.mse_loss(pred, Ystd)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp0.parameters(), 1.0)
                opt.step()
                if (step + 1) % 250 == 0:
                    sc = val_ccos(mlp0)
                    if sc > best_sc:
                        best_sc, best_state, bad = sc, copy.deepcopy(mlp0.state_dict()), 0
                    else:
                        bad += 1
                        if bad >= 4:
                            break
            mlp0.load_state_dict(best_state)
            print(f"[mlp k={k}] MSE stage done at step {step + 1}, val ccos {best_sc:.4f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        del Xv, Yv

        def inflight(ctx, a, s, branch, mlp):
            """branch pass over positions a .. s-1. Returns the drafter rows [s-a, nT, H] (raw space) and the
            head outputs in standardised space [s-a, n_out, H]."""
            hs, cos, sin, kvp = ctx
            zs = branch(hs[0][:, a:s], a, kvp, cos[:, a:s], sin[:, a:s])
            zk = zs[-1][0].float()
            P = ridge_std(zk) + mlp((zk - xm) / xs).float()
            raw = P * ys + ym
            row = torch.stack([raw[:, tapq[j]] if j in tapq else zs[hidx[j]][0].float() for j in range(nT)], 1)
            return row, P

        def loss_fn(si, n_blocks, branch, mlp, train):
            ids, lo = seqs[si]
            x = ids.to(dev)
            T = len(ids)
            lo_s = max(lo, args.m_train + 1)
            if T - bs - 1 <= lo_s:
                return None, None, None
            ctx = seq_ctx(ids, branch)
            hs = ctx[0]
            feats = extract_context_feature(hs, lids)
            anchors = rng.sample(range(lo_s, T - bs - 1), min(n_blocks, T - bs - 1 - lo_s))
            blocks, parts, fins, mse, kl = [], [], [], 0.0, torch.zeros((), device=dev)
            with torch.autocast("cuda", dtype=BF):
                for s in anchors:
                    m = rng.randint(1, args.m_train) if train else args.m_train // 2
                    row, P = inflight(ctx, s - m, s, branch, mlp)
                    Yreal = torch.stack([(hs[tgt_layer(t)][0, s - m : s].float() - ym[q]) / ys[q]
                                         for q, t in enumerate(tk)], 1)
                    mse = mse + F.mse_loss(P, Yreal)
                    if args.kl_w:
                        lg = head((P[:, fin] * ys[fin] + ym[fin]).to(hs[-1].dtype)).float()
                        tl = head(hs[-1][0, s - m : s]).float()
                        kl = kl + F.kl_div(F.log_softmax(lg, -1), F.softmax(tl, -1), reduction="batchmean")
                    fins.append(P[:, fin] * ys[fin] + ym[fin])
                    blocks.append((s, m))
                    parts.append(row.to(feats.dtype).view(1, m, nT * H))
                cx, noise, posi, mask, starts = pack_sub(draft, target, feats, x, blocks, bs, parts)
                DLoRA.on = bool(dmods)
                hid = draft(target_hidden=cx, noise_embedding=noise, position_ids=posi, attention_mask=mask,
                            past_key_values=None, use_cache=False)[0]
                DLoRA.on = False
                ce, _ = _weighted_ce(draft, target, hid, x, blocks, starts, bs, T, args.gamma)
                aux = args.kl_w * kl / len(blocks)
                if args.rank_w:
                    # the drafter's candidates for the tokens after the in-flight positions: the block it drafts
                    # from the last verified position s - m with real features (original drafter, no gradient)
                    with torch.no_grad():
                        DL = blocks_logits(draft, target, feats, x, [(s - m, 0) for s, m in blocks], bs)
                    rk = 0.0
                    for b, (s, m) in enumerate(blocks):
                        nv = min(m, bs - 1)
                        sc, cand = rerank(mlp, fins[b][:nv], DL[b, :nv], hs[-1].dtype)
                        tgt = x[s - m + 1 : s - m + 1 + nv]
                        hit = cand == tgt[:, None]
                        has = hit.any(1)
                        if has.any():
                            w = torch.where(cand[:, 0] != tgt, args.fork_w, 1.0)[has]
                            l = F.cross_entropy(sc[has], hit.float().argmax(1)[has], reduction="none")
                            rk = rk + (l * w).sum() / w.sum()
                    aux = aux + args.rank_w * rk / len(blocks)
            return ce, mse / len(blocks), aux

        def train(name, branch, mlp):
            mods = nn.ModuleDict({"branch": branch, "mlp": mlp})
            dpar = [p for w in dmods for p in (w.A, w.B)]
            mods["draft"] = nn.ParameterList(dpar)
            lrb = args.lr_branch or (2e-4 if name == "lora" else 2e-5)
            groups = [{"params": list(mlp.parameters()), "base": args.lr_head}]
            bp = [p for p in branch.parameters() if p.requires_grad]
            if bp:
                groups.append({"params": bp, "base": lrb})
            if dpar:
                groups.append({"params": dpar, "base": args.lr_draft})
            opt = torch.optim.AdamW(groups, lr=args.lr_head, weight_decay=0.01)
            params = [p for g in groups for p in g["params"]]

            def scorer():
                mods.eval()
                st = rng.getstate()
                rng.seed(123)
                tot, n = 0.0, 0
                with torch.no_grad():
                    for si in val_seqs:
                        ce, _, kl = loss_fn(si, 4, branch, mlp, False)
                        if ce is not None:
                            tot += ce.item() + kl.item()
                            n += 1
                rng.setstate(st)
                mods.train()
                return -tot / max(n, 1)

            def snap():
                return {n: v.detach().cpu().clone() for n, v in mods.state_dict().items()}

            best_sc, best_state, log, bad = scorer(), snap(), [], 0
            print(f"[{name} k={k}] start val {best_sc:.4f}, trainable "
                  f"{sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
            for step in range(args.steps):
                for g in opt.param_groups:
                    g["lr"] = g["base"] * cosine_lr(step, args.steps, min(200, args.steps // 10))
                for _ in range(10):
                    ce, mse, kl = loss_fn(rng.choice(fit_seqs), args.blocks, branch, mlp, True)
                    if ce is not None:
                        break
                loss = args.ce_w * ce + args.mse_w * mse + kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                    sc = scorer()
                    log.append((step + 1, ce.item(), mse.item(), kl.item(), sc))
                    print(f"[{name} k={k}] step {step + 1} ce {ce.item():.4f} mse {mse.item():.4f} "
                          f"aux {kl.item():.4f} val {sc:.4f} "
                          f"{time.time() - t0:.0f}s", flush=True)
                    if sc > best_sc:
                        best_sc, best_state, bad = sc, snap(), 0
                    else:
                        bad += 1
                        if bad >= args.patience:
                            print(f"[{name} k={k}] early stop at step {step + 1}", flush=True)
                            break
            mods.load_state_dict(best_state)
            mods.eval()
            train_log[f"{name}_k{k}"] = log

        @torch.no_grad()
        def evaluate(name, branch, mlp):
            ad = bool(dmods) and name in TRAINED
            tau, probe, forks = {}, [], {}
            fk = {n: 0 for n in ("err", "clean", "recall", "second", "rr1", "resolved", "fa")}
            for n_done, (si, anchors) in enumerate(by_seq.items()):
                ids, _lo = rest[si]
                x = ids.to(dev)
                ctx = seq_ctx(ids, branch)
                hs = ctx[0]
                feats = extract_context_feature(hs, lids)
                truth_all = x.tolist()
                logits_real = head(hs[-1][0]).float()
                with torch.autocast("cuda", dtype=BF):
                    for s in anchors:
                        truth = truth_all[s + 1 :]
                        Lr = block_lg(feats[:, :s], x, s)
                        chain = Lr.argmax(-1).tolist()
                        e = matched(chain, truth)
                        tau.setdefault("real", []).append(e + 1)
                        if ad:
                            tau.setdefault("real_ad", []).append(
                                matched(run_block(feats[:, :s], x, s, adapted=True), truth) + 1)
                        for m in args.ms:
                            tau.setdefault(f"removed_m{m}", []).append(
                                matched(run_block(feats[:, : s - m], x, s, m), truth) + 1)
                            row, P = inflight(ctx, s - m, s, branch, mlp)
                            fz = feats[:, :s].clone()
                            fz[0, s - m : s] = row.to(fz.dtype).view(m, nT * H)
                            tau.setdefault(f"m{m}", []).append(
                                matched(run_block(fz, x, s, adapted=ad), truth) + 1)
                            if m == M:
                                top = head((P[:, fin] * ys[fin] + ym[fin]).to(hs[-1].dtype)).argmax(-1)
                                probe.append((top == logits_real[s - m : s].argmax(-1)).float().cpu())
                        # one partial pass over the drafted block up to its first error (drafted = true tokens
                        # there): the reranker's first disagreement with the draft is the fork
                        last = min(e, len(chain) - 1)
                        _row, P = inflight(ctx, s, s + last + 1, branch, mlp)
                        fraw = P[:, fin] * ys[fin] + ym[fin]
                        sc, cand = rerank(mlp, fraw, Lr[: last + 1], hs[-1].dtype)
                        pick = cand.gather(1, sc.argmax(1, keepdim=True))[:, 0].tolist()
                        dis = [i for i in range(last + 1) if pick[i] != chain[i]]
                        j = dis[0] if dis else -1
                        if e < len(chain):
                            fk["err"] += 1
                            fk["recall"] += truth[e] in cand[e].tolist()
                            fk["second"] += int(cand[e, 1]) == truth[e]
                            fk["rr1"] += pick[e] == truth[e]
                            fk["resolved"] += j == e and pick[e] == truth[e]
                            fk["fa"] += 0 <= j < e
                        else:
                            fk["clean"] += 1
                            fk["fa"] += j >= 0
                        if e < len(chain):
                            p = s + e
                            pr = torch.softmax(logits_real[p], -1).topk(2).values
                            tie = "near-tie" if (pr[0] - pr[1]).item() < args.tie else "clear"
                            top = head(fraw[-1:].to(hs[-1].dtype)).float()[0].topk(3)
                            top = top.indices.tolist()
                            f = forks.setdefault(tie, [0, 0, 0])
                            f[0] += top[0] == truth_all[p + 1]
                            f[1] += truth_all[p + 1] in top
                            f[2] += 1
                del hs, feats, ctx
                if n_done % 50 == 0:
                    print(f"[eval {name} k={k}] {n_done}/{len(by_seq)} sequences, {time.time() - t0:.0f}s",
                          flush=True)
            real = mean(tau["real"])
            r = {"k": k, "method": name, "kv-map": args.kv_map if name in ("lora", "copy") else "own",
                 "real": real, "real adapted": mean(tau["real_ad"]) if ad else real}
            for m in args.ms:
                rem = mean(tau[f"removed_m{m}"])
                r[f"removed m{m}"] = rem
                r[f"tau m{m}"] = mean(tau[f"m{m}"])
                r[f"share m{m}"] = (mean(tau[f"m{m}"]) - rem) / max(real - rem, 1e-9)
            r["t+1 top-1"] = torch.cat(probe).mean().item()
            c = forks.get("clear", [0, 0, 0])
            r["fork top-1"], r["fork top-3"], r["forks"] = c[0] / max(c[2], 1), c[1] / max(c[2], 1), c[2]
            ne = max(fk["err"], 1)
            r["cand recall"], r["2nd"], r["rr top-1"] = fk["recall"] / ne, fk["second"] / ne, fk["rr1"] / ne
            r["resolved"], r["false alarm"] = fk["resolved"] / ne, fk["fa"] / max(fk["err"] + fk["clean"], 1)
            r["alpha"] = float(mlp.alpha)
            rows_out.append(r)
            print_table([r], list(r), f"k={k} {name}")

        for name in args.methods:
            torch.cuda.empty_cache()
            trained = name in TRAINED
            own_map = name not in ("lora", "copy")           # a frozen branch only makes sense on its own cache
            branch = Branch(target, k, name if name in ("lora", "copy") else "read", args.rank,
                            kv_map("own", k, L) if own_map else gmap).to(dev)
            mlp = copy.deepcopy(zero_mlp if name == "ridge" else mlp0)
            mlp.alpha = nn.Parameter(torch.ones((), device=dev))
            for w in dmods:
                w.reset()
            if trained:
                mlp.train()
                train(name, branch, mlp)
            mlp.eval()
            branch.eval()
            evaluate(name, branch, mlp)
            del branch, mlp
        print(f"[k={k}] done, {time.time() - t0:.0f}s", flush=True)

    print_table(rows_out, list(rows_out[0]), "Share of real, probe and forks (frozen drafter, held-out anchors)")
    save_json(args.save, {"args": vars(args), "rows": rows_out, "train_log": train_log})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
