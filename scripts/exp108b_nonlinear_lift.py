"""Experiment 108b: exp108 with the "reader failed, not the information" objections removed.

exp108 (150k fit tokens from 572 of ~19.8k sequences): the MLP reader overfits from the first check on (Qwen3-8B,
k = 4: val centred cosine 0.713 at step 250, falling while the training loss falls). So exp108 is data-limited and
cannot say "missing". Changes here:

  data      --fit-tokens 600k by default (4x exp108; exp108 is the first point of the data curve), stored once in
            flat host tensors (no duplicate copies), sequences kept as prompt part + offset into the flat arrays
  mlp       residual MLP on ridge, width 2048, dropout 0.1, eval every 100 steps, early stop on patience
  seq       causal transformer reader, dropout 0.1, same evaluation / early stop
  dthru     the strongest check: the MLP reader (initialised from the trained mlp) fine-tuned THROUGH the frozen
            drafter, with the drafter's own position-weighted block cross-entropy (DFlash recipe, gamma 0.9) on
            blocks whose last m context positions get the reader's taps (m uniform in 1..--m-train), plus a small
            standardised-MSE term to the real taps. Tests whether the drafter can use information that tap-MSE
            readers do not reconstruct. Several blocks per sequence share one drafter pass (pack_depth layout:
            every block gets its own copy of its substituted rows).

Everything is reported exactly as exp108 (closeness train / val, t+1 probe, share of the gap at m = 8 / 16 with the
frozen drafter, forks), so exp108 and exp108b rows compare directly. Readers are fitted on the first sequences and all
evaluation uses later, disjoint sequences.

  python scripts/exp108b_nonlinear_lift.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
      --data /scratch-shared/$USER/hspec_lagft/data
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

from dflash.model import _draft_value, _raw_input_embeddings, extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp107_lift_diag import Ridge, apply_map, load_seqs  # noqa: E402

from hspec.lagtrain import _weighted_ce  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

READERS = ("ridge", "mlp", "seq", "dthru")


class MLPReader(nn.Module):
    """Residual on the ridge map, standardised target space: y = ridge(x) + f(x). Zero-initialised output."""

    def __init__(self, H, n_out, width, drop):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(H), nn.Dropout(drop), nn.Linear(H, width), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(width, width), nn.GELU(), nn.Dropout(drop), nn.Linear(width, n_out * H))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.n_out, self.H = n_out, H

    def forward(self, x):
        return self.net(x).view(len(x), self.n_out, self.H)


class SeqReader(nn.Module):
    """Causal transformer over the standardised h_k sequence, residual on ridge. Zero-initialised output."""

    def __init__(self, H, n_out, d, layers, heads, drop):
        super().__init__()
        self.inp = nn.Linear(H, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=drop, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_out * H)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.n_out, self.H = n_out, H

    def forward(self, x):
        T = len(x)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        h = self.body(self.inp(x)[None], mask=mask, is_causal=True)[0]
        return self.out(self.norm(h)).view(T, self.n_out, self.H)


def cosine_lr(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


def pack_sub(draft, head_src, feats, ids, blocks, bs, parts):
    """pack_depth (hspec.lagtrain) with given substitute rows: block (s, g) attends to the real context before
    s - g and to its own copy parts[i] [1, g, W] of the rows s - g .. s - 1 (differentiable)."""
    T = feats.shape[1]
    dev = ids.device
    ppos, pspans = [], []
    c = T
    for s, g in blocks:
        ppos.append(torch.arange(s - g, s, device=dev))
        pspans.append((c, c + g))
        c += g
    ctx = torch.cat([feats] + list(parts), dim=1)
    C = ctx.shape[1]
    toks, qpos, qspans, starts = [], [], [], []
    q = 0
    for s, _g in blocks:
        t = torch.full((bs,), draft.mask_token_id, dtype=torch.long, device=dev)
        t[0] = ids[s]
        toks.append(t)
        qpos.append(torch.arange(s, s + bs, device=dev))
        qspans.append((q, q + bs))
        starts.append(q + 1)
        q += bs
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    noise = _raw_input_embeddings(head_src, torch.cat(toks)[None], scale)
    pos = torch.cat([torch.arange(T, device=dev)] + ppos + qpos)[None]
    mask = torch.zeros((q, C + q), dtype=torch.bool, device=dev)
    for (s, g), (a, b), (pa, pb) in zip(blocks, qspans, pspans):
        mask[a:b, : s - g] = True
        mask[a:b, pa:pb] = True
        mask[a:b, C + a : C + b] = True
    return ctx, noise, pos, mask[None, None], starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--exits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--readers", nargs="+", default=list(READERS))
    ap.add_argument("--ms", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--fit-tokens", type=int, default=600_000)
    ap.add_argument("--val-tokens", type=int, default=15_000)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--mlp-width", type=int, default=2048)
    ap.add_argument("--mlp-steps", type=int, default=6000)
    ap.add_argument("--mlp-batch", type=int, default=4096)
    ap.add_argument("--mlp-lr", type=float, default=5e-4)
    ap.add_argument("--seq-d", type=int, default=1024)
    ap.add_argument("--seq-layers", type=int, default=4)
    ap.add_argument("--seq-heads", type=int, default=16)
    ap.add_argument("--seq-steps", type=int, default=5000, help="optimiser steps, --seq-accum sequences each")
    ap.add_argument("--seq-accum", type=int, default=4)
    ap.add_argument("--seq-lr", type=float, default=3e-4)
    ap.add_argument("--dthru-steps", type=int, default=1500)
    ap.add_argument("--dthru-blocks", type=int, default=8, help="blocks per sequence per step")
    ap.add_argument("--dthru-lr", type=float, default=1e-4)
    ap.add_argument("--dthru-mse", type=float, default=0.1, help="weight of the tap-MSE anchor term")
    ap.add_argument("--dthru-val-seqs", type=int, default=24)
    ap.add_argument("--m-train", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10, help="evaluations without improvement before stopping")
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--tie", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--save", default="exp108b_nonlinear_lift")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    assert all(r in READERS for r in args.readers) and "ridge" in args.readers, "ridge is the base of the others"
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
    norm, head = target.model.norm, target.lm_head
    lids, bs = list(draft.target_layer_ids), draft.block_size
    hidx = [l + 1 for l in lids]
    H, nT = target.config.hidden_size, len(lids)
    missing = {k: [j for j, h in enumerate(hidx) if h > k] for k in args.exits}
    tgts = {k: [f"tap{j}" for j in missing[k]] + ["final"] for k in args.exits}
    all_t = sorted({t for k in args.exits for t in tgts[k]}, key=lambda t: (t == "final", t))
    tpos = {t: all_t.index(t) for t in all_t}
    print(f"[setup] {len(seqs)} sequences, taps hs{hidx}, "
          + ", ".join(f"k={k}: missing hs{[hidx[j] for j in v]}" for k, v in missing.items()), flush=True)

    def tgt_layer(t):
        return -1 if t == "final" else hidx[int(t[3:])]

    @torch.no_grad()
    def hidden(ids):
        return target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states

    # ---- collect into flat host tensors: X[k] [N, H], Y [N, n_all_t, H] at response positions; per record the
    # prompt part of h_k (for the seq reader) and the offset into the flat arrays.
    def collect(lo_i, n_tokens, cap_i):
        X = {k: torch.empty(n_tokens + args.max_len, H, dtype=torch.bfloat16) for k in args.exits}
        Y = torch.empty(n_tokens + args.max_len, len(all_t), H, dtype=torch.bfloat16)
        recs, n, i = [], 0, lo_i
        while n < n_tokens and i < cap_i:
            ids, lo = seqs[i]
            hs = hidden(ids)
            r = len(ids) - lo
            for k in args.exits:
                X[k][n : n + r] = hs[k][0, lo:].cpu()
            Y[n : n + r] = torch.stack([hs[tgt_layer(t)][0, lo:] for t in all_t], 1).cpu()
            recs.append({"seq": i, "lo": lo, "off": n, "n": r,
                         "pre": {k: hs[k][0, :lo].to(torch.bfloat16).cpu() for k in args.exits}})
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

    def chunks(n, size=16384):
        return [(a, min(a + size, n)) for a in range(0, n, size)]

    def moments(T):
        s, s2 = 0.0, 0.0
        for a, b in chunks(len(T)):
            v = T[a:b].to(dev).float()
            s = s + v.sum(0)
            s2 = s2 + (v * v).sum(0)
        mu = s / len(T)
        return mu, (s2 / len(T) - mu * mu).clamp_min(1e-6).sqrt()

    xstat = {k: moments(fit["X"][k]) for k in args.exits}
    ystat = moments(fit["Y"])

    def seq_x(D, r, k):
        """full h_k sequence of record r (prompt + response), on the device, float"""
        return torch.cat([r["pre"][k], D["X"][k][r["off"] : r["off"] + r["n"]]]).to(dev).float()

    def ccos(a, b, mu):
        return F.cosine_similarity(a - mu, b - mu, dim=-1)

    readers, report_close, train_log = {}, [], {}
    for k in args.exits:
        sel = torch.tensor([tpos[t] for t in tgts[k]])
        xm, xs = xstat[k]
        ym, ys = ystat[0][sel], ystat[1][sel]
        n_out = len(sel)
        tk = tuple(tgts[k])

        # -- ridge
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
                sc = ccos(apply_map(Wm, Xv), Yv[:, q], ym[q]).mean().item()
                if t not in best or sc > best[t][0]:
                    best[t] = (sc, lam, Wm)
        del acc, Xv, Yv
        ridge = {t: best[t][2] for t in tk}

        def ridge_std(Xraw, ridge=ridge, ym=ym, ys=ys, tk=tk):
            with torch.autocast("cuda", enabled=False):
                Xraw = Xraw.float()
                return torch.stack([(apply_map(ridge[t], Xraw) - ym[q]) / ys[q] for q, t in enumerate(tk)], 1)

        def predict_raw(name, Xseq, pos, model=None, ridge_std=ridge_std, xm=xm, xs=xs, ym=ym, ys=ys):
            """predictions in raw space for positions pos of one full h_k sequence, [len(pos), n_out, H]"""
            Xp = Xseq[pos]
            base = ridge_std(Xp)
            if name in ("mlp", "dthru"):
                base = base + model((Xp - xm) / xs).float()
            elif name == "seq":
                base = base + model((Xseq[: int(pos.max()) + 1] - xm) / xs)[pos].float()
            return base * ys + ym

        def score(name, model=None, D=None, n_recs=None):
            """mean centred cosine per output over the records of D (val by default)"""
            D = val if D is None else D
            recs = D["recs"] if n_recs is None else D["recs"][:n_recs]
            cs = [[] for _ in range(n_out)]
            if model is not None:
                model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for r in recs:
                    Xs = seq_x(D, r, k)
                    pos = torch.arange(r["lo"], r["lo"] + r["n"], device=dev)
                    P = predict_raw(name, Xs, pos, model)
                    Y = D["Y"][r["off"] : r["off"] + r["n"], sel].to(dev).float()
                    for q in range(n_out):
                        cs[q].append(ccos(P[:, q], Y[:, q], ym[q]).cpu())
            if model is not None:
                model.train()
            return [torch.cat(c).mean().item() for c in cs]

        def run_training(name, model, step_fn, steps, lr, scorer):
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
            best_sc, best_state, log, bad = scorer(model), copy.deepcopy(model.state_dict()), [], 0
            print(f"[{name} k={k}] start (init) val {best_sc:.4f}", flush=True)
            for step in range(steps):
                for g in opt.param_groups:
                    g["lr"] = lr * cosine_lr(step, steps, min(200, steps // 10))
                opt.zero_grad(set_to_none=True)
                loss = step_fn()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if (step + 1) % args.eval_every == 0 or step + 1 == steps:
                    sc = scorer(model)
                    log.append((step + 1, loss, sc))
                    print(f"[{name} k={k}] step {step + 1} loss {loss:.4f} val {sc:.4f}", flush=True)
                    if sc > best_sc:
                        best_sc, best_state, bad = sc, copy.deepcopy(model.state_dict()), 0
                    else:
                        bad += 1
                        if bad >= args.patience:
                            print(f"[{name} k={k}] early stop at step {step + 1}", flush=True)
                            break
            model.load_state_dict(best_state)
            model.eval()
            train_log[f"{name}_k{k}"] = log
            return model

        fns = {"ridge": None}

        def mse_scorer(name):
            return lambda model: sum(score(name, model)) / n_out

        if "mlp" in args.readers or "dthru" in args.readers:
            model = MLPReader(H, n_out, args.mlp_width, args.drop).to(dev)

            def mlp_step(model=model):
                b = torch.randint(0, fit["n"], (args.mlp_batch,))
                X = fit["X"][k].index_select(0, b).to(dev, non_blocking=True).float()
                Ystd = (fit["Y"].index_select(0, b)[:, sel].to(dev, non_blocking=True).float() - ym) / ys
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = ridge_std(X) + model((X - xm) / xs).float()
                loss = F.mse_loss(pred, Ystd)
                loss.backward()
                return loss.item()

            fns["mlp"] = run_training("mlp", model, mlp_step, args.mlp_steps, args.mlp_lr, mse_scorer("mlp"))

        if "seq" in args.readers:
            model = SeqReader(H, n_out, args.seq_d, args.seq_layers, args.seq_heads, args.drop).to(dev)
            order = list(range(len(fit["recs"])))
            ptr = [len(order)]

            def seq_step(model=model):
                tot = 0.0
                for _ in range(args.seq_accum):
                    if ptr[0] >= len(order):
                        rng.shuffle(order)
                        ptr[0] = 0
                    r = fit["recs"][order[ptr[0]]]
                    ptr[0] += 1
                    Xs = seq_x(fit, r, k)
                    pos = torch.arange(r["lo"], r["lo"] + r["n"], device=dev)
                    Ystd = (fit["Y"][r["off"] : r["off"] + r["n"], sel].to(dev).float() - ym) / ys
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = ridge_std(Xs[pos]) + model((Xs - xm) / xs)[pos].float()
                    loss = F.mse_loss(pred, Ystd) / args.seq_accum
                    loss.backward()
                    tot += loss.item()
                return tot

            fns["seq"] = run_training("seq", model, seq_step, args.seq_steps, args.seq_lr, mse_scorer("seq"))

        if "dthru" in args.readers:
            model = copy.deepcopy(fns["mlp"]).train()
            tap_q = [q for q, t in enumerate(tk) if t != "final"]
            tap_j = [int(tk[q][3:]) for q in tap_q]

            def dthru_loss(r_seq, n_blocks, model, train):
                """drafter block CE for n_blocks anchors of one sequence, last m rows from the reader"""
                ids, lo = seqs[r_seq]
                x = ids.to(dev)
                T = len(ids)
                hs = hidden(ids)
                feats = extract_context_feature(hs, lids)
                Xs = hs[k][0].float()
                lo_s = max(lo, args.m_train + 1)
                if T - bs - 1 <= lo_s:
                    return None, None
                anchors = rng.sample(range(lo_s, T - bs - 1), min(n_blocks, T - bs - 1 - lo_s))
                blocks, parts, mse = [], [], 0.0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for s in anchors:
                        m = rng.randint(1, args.m_train) if train else args.m_train // 2
                        pos = torch.arange(s - m, s, device=dev)
                        Xp = Xs[pos]
                        P = (ridge_std(Xp) + model((Xp - xm) / xs).float())[:, tap_q]       # standardised
                        Yreal = torch.stack([(hs[tgt_layer(tk[q])][0, pos].float() - ym[q]) / ys[q]
                                             for q in tap_q], 1)
                        mse = mse + F.mse_loss(P, Yreal)
                        raw = P * ys[tap_q] + ym[tap_q]
                        row = feats[:, s - m : s].clone().view(1, m, nT, H)
                        row = torch.cat([raw[:, tap_j.index(j)][None, :, None] if j in tap_j else row[:, :, j : j + 1]
                                         for j in range(nT)], 2).to(feats.dtype).view(1, m, nT * H)
                        blocks.append((s, m))
                        parts.append(row)
                    ctx, noise, posi, mask, starts = pack_sub(draft, target, feats, x, blocks, bs, parts)
                    hid = draft(target_hidden=ctx, noise_embedding=noise, position_ids=posi, attention_mask=mask,
                                past_key_values=None, use_cache=False)[0]
                    ce, _ = _weighted_ce(draft, target, hid, x, blocks, starts, bs, T, args.gamma)
                del hs
                return ce, mse / len(blocks)

            val_seqs = [r["seq"] for r in val["recs"][: args.dthru_val_seqs]]
            fit_seqs = [r["seq"] for r in fit["recs"]]

            def dthru_scorer(model):
                model.eval()
                st = rng.getstate()
                rng.seed(123)
                tot, n = 0.0, 0
                with torch.no_grad():
                    for sq in val_seqs:
                        ce, _ = dthru_loss(sq, 4, model, False)
                        if ce is not None:
                            tot += ce.item()
                            n += 1
                rng.setstate(st)
                model.train()
                return -tot / max(n, 1)                     # higher is better

            def dthru_step(model=model):
                for _ in range(10):
                    ce, mse = dthru_loss(rng.choice(fit_seqs), args.dthru_blocks, model, True)
                    if ce is not None:
                        break
                loss = ce + args.dthru_mse * mse
                loss.backward()
                return ce.item()

            fns["dthru"] = run_training("dthru", model, dthru_step, args.dthru_steps, args.dthru_lr, dthru_scorer)

        for name in args.readers:
            tr = score(name, fns[name], fit, 40)
            va = score(name, fns[name])
            for q, t in enumerate(tk):
                report_close.append({"k": k, "reader": name, "target": t if t == "final" else f"L{lids[int(t[3:])]}",
                                     "train ccos": tr[q], "val ccos": va[q]})
        readers[k] = (fns, predict_raw)
        print(f"[fit] k={k} done, {time.time() - t0:.0f}s", flush=True)
    del fit, val
    torch.cuda.empty_cache()
    print_table(report_close, list(report_close[0]), "Closeness: centred cosine of the read state to the real one")

    # ---- evaluation on held-out sequences (exp107 / exp108 protocol) ----
    rest = seqs[fit_end:]
    cand = [(si, s) for si, (ids, lo) in enumerate(rest) for s in range(max(lo, M + 2), len(ids) - bs)]
    random.Random(1).shuffle(cand)
    cand = sorted(cand[: args.anchors])
    by_seq = {}
    for si, s in cand:
        by_seq.setdefault(si, []).append(s)
    tau, probe, forks = {}, {}, {}

    def rec(name, v):
        tau.setdefault(name, []).append(v)

    def run_block(fz, x, s, lag=0):
        L = blocks_logits(draft, target, fz, x, [(s, lag)], bs)[0]
        return L.argmax(-1).tolist()

    with torch.no_grad():
        for n_done, (si, anchors) in enumerate(by_seq.items()):
            ids, lo = rest[si]
            x = ids.to(dev)
            T = len(ids)
            hs = hidden(ids)
            feats = extract_context_feature(hs, lids)
            truth_all = x.tolist()
            logits_real = head(hs[-1][0]).float()
            resp = torch.arange(lo, T - 1, device=dev)
            gold = logits_real[resp].argmax(-1)
            allpos = torch.arange(T, device=dev)
            preds = {}
            for k in args.exits:
                fns, predict_raw = readers[k]
                Xs = hs[k][0].float()
                fin = tgts[k].index("final")
                for name in args.readers:
                    preds[(k, name)] = predict_raw(name, Xs, allpos, fns[name]).to(hs[-1].dtype)
                    top = head(preds[(k, name)][resp, fin]).argmax(-1)
                    probe.setdefault(f"{name}_k{k}", []).append((top == gold).float().cpu())
                top = head(norm(hs[k][0, resp])).argmax(-1)
                probe.setdefault(f"raw_exit_k{k}", []).append((top == gold).float().cpu())
            for s in anchors:
                truth = truth_all[s + 1 :]
                chain = run_block(feats[:, :s], x, s)
                rec("real", matched(chain, truth) + 1)
                e = matched(chain, truth)
                if e < len(chain):
                    p = s + e
                    tok = truth_all[p + 1]
                    pr = torch.softmax(logits_real[p], -1).topk(2).values
                    tie = "near-tie" if (pr[0] - pr[1]).item() < args.tie else "clear"
                    for k in args.exits:
                        fin = tgts[k].index("final")
                        res = {"raw_exit": head(norm(hs[k][0, p : p + 1])).float()[0]}
                        for name in args.readers:
                            res[name] = head(preds[(k, name)][p : p + 1, fin]).float()[0]
                        for rname, lg in res.items():
                            top = lg.topk(3).indices.tolist()
                            f = forks.setdefault((f"{rname}_k{k}", tie), [0, 0, 0])
                            f[0] += top[0] == tok
                            f[1] += tok in top
                            f[2] += 1
                for m in args.ms:
                    rec(f"removed_m{m}", matched(run_block(feats[:, : s - m], x, s, m), truth) + 1)
                    for k in args.exits:
                        for name in args.readers:
                            fz = feats[:, :s].clone()
                            blk = fz[0, s - m : s].view(m, nT, H)
                            P = preds[(k, name)][s - m : s]
                            for q, t in enumerate(tgts[k]):
                                if t != "final":
                                    blk[:, int(t[3:])] = P[:, q].to(blk.dtype)
                            fz[0, s - m : s] = blk.view(m, nT * H)
                            rec(f"{name}_k{k}_m{m}", matched(run_block(fz, x, s), truth) + 1)
            del hs, feats, preds
            if n_done % 20 == 0:
                print(f"[run] {n_done}/{len(by_seq)} sequences, {time.time() - t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    real = mean(tau["real"])
    rows = []
    for m in args.ms:
        rem = mean(tau[f"removed_m{m}"])
        r = {"m": m, "real": real, "removed": rem}
        for k in args.exits:
            for name in args.readers:
                r[f"{name}_k{k}"] = (mean(tau[f"{name}_k{k}_m{m}"]) - rem) / max(real - rem, 1e-9)
        rows.append(r)
    print_table(rows, list(rows[0]), "Share of real (tau - removed) / (real - removed); real / removed are tau")
    out = {"args": vars(args), "share": rows, "closeness": report_close, "train_log": train_log}
    prow = [{"reader": n, "t+1 top-1": torch.cat(v).mean().item()} for n, v in sorted(probe.items())]
    print_table(prow, list(prow[0]), "Next-token probe: top-1 of lm_head(read final state) vs the target's token")
    out["probe"] = prow
    frow = [{"resolver": r, "fork type": tie, "forks": c, "top-1": t1 / c, "top-3": t3 / c}
            for (r, tie), (t1, t3, c) in sorted(forks.items())]
    print_table(frow, list(frow[0]), "Forks: resolving the drafter's first error (real-feature block)")
    out["forks"] = frow
    save_json(args.save, out)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
