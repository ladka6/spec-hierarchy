"""Experiment 108: is the deep-tap information in a shallow state missing, or only not linearly readable?

exp87 / exp107: a LINEAR reader of h_k (the residual stream after the first k layers) is weak on targets without exit
training (t+1 probe at layer 4: Qwen3 0.31, plain Llama3 0.33, LayerSkip 0.59; ridge lift share of the gap at
k = 4 / 8: Qwen3 0.33 / 0.46, LayerSkip 0.59 / 0.78). Stacking more shallow layers added nothing (exp107 A2).
Open question: is the information there but encoded nonlinearly, or not computed yet at depth k?

Readers, all h_k -> {the drafter's missing deep taps, the final state}, fitted on the target's own greedy text:
  ridge   closed-form linear map (exp107's), refitted on the same tokens as the others
  mlp     per-token residual MLP on top of the ridge map (zero-initialised output: starts exactly at ridge)
  seq     small causal transformer over the h_k sequence (context positions included), residual on ridge:
          tests whether the information is spread over positions rather than missing
Training: standardised targets (per-dimension mean / std), MSE in that space (= normalised MSE), AdamW, cosine
schedule, best checkpoint by validation centred cosine. Nothing in the target or the drafter is trained.

Evaluation on held-out sequences:
  closeness   centred cosine to the real state, per missing tap and the final state (train and val, to separate
              underfitting from missing information)
  probe       top-1 of lm_head(predicted final state) vs the target's greedy token (comparable to exp87's t+1 probe);
              the raw exit head lm_head(norm(h_k)) is reported for reference
  share       exp107 A1 protocol: the frozen drafter, the last m context positions get the reader's taps;
              share = (tau - removed) / (real - removed)
  forks       top-1 / top-3 of the lifted final state at the drafter's first error (exp107 A4)

Reading the result (Qwen3-8B): nonlinear share at k = 8 near LayerSkip's linear 0.78 -> encoded, a reader-based
recipe can work on targets without exit training. Nonlinear share near the linear 0.46 -> missing at that depth, the
method needs an exit-trained target. Run LayerSkip Llama3-8B as the control (does nonlinearity add there too?).

  python scripts/exp108_nonlinear_lift.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
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

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp107_lift_diag import Ridge, apply_map, load_seqs  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

READERS = ("ridge", "mlp", "seq")


class MLPReader(nn.Module):
    """Residual on the ridge map, in standardised target space: y = ridge(x) + f(x)."""

    def __init__(self, H, n_out, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, width), nn.GELU(), nn.Linear(width, width),
                                 nn.GELU(), nn.Linear(width, n_out * H))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.n_out, self.H = n_out, H

    def forward(self, x):                       # x [N, H] standardised -> [N, n_out, H]
        return self.net(x).view(len(x), self.n_out, self.H)


class SeqReader(nn.Module):
    """Causal transformer over the standardised h_k sequence, residual on the ridge map."""

    def __init__(self, H, n_out, d, layers, heads):
        super().__init__()
        self.inp = nn.Linear(H, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_out * H)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.n_out, self.H = n_out, H

    def forward(self, x):                       # x [T, H] standardised (one sequence) -> [T, n_out, H]
        T = len(x)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        h = self.body(self.inp(x)[None], mask=mask, is_causal=True)[0]
        return self.out(self.norm(h)).view(T, self.n_out, self.H)


def cosine_lr(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--exits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--readers", nargs="+", default=list(READERS))
    ap.add_argument("--ms", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--fit-tokens", type=int, default=150_000, help="response tokens for fitting (all readers)")
    ap.add_argument("--val-tokens", type=int, default=10_000)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--mlp-width", type=int, default=4096)
    ap.add_argument("--mlp-steps", type=int, default=4000)
    ap.add_argument("--mlp-batch", type=int, default=4096)
    ap.add_argument("--mlp-lr", type=float, default=5e-4)
    ap.add_argument("--seq-d", type=int, default=1024)
    ap.add_argument("--seq-layers", type=int, default=4)
    ap.add_argument("--seq-heads", type=int, default=16)
    ap.add_argument("--seq-steps", type=int, default=4000, help="optimiser steps, --seq-accum sequences each")
    ap.add_argument("--seq-accum", type=int, default=4)
    ap.add_argument("--seq-lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--tie", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--save", default="exp108_nonlinear_lift")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    assert all(r in READERS for r in args.readers) and "ridge" in args.readers, "ridge is the base of the others"
    dev, t0 = "cuda", time.time()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
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
    print(f"[setup] {len(seqs)} sequences, taps hs{hidx}, "
          + ", ".join(f"k={k}: missing hs{[hidx[j] for j in v]}" for k, v in missing.items()), flush=True)

    def tgt_layer(t):
        return -1 if t == "final" else hidx[int(t[3:])]

    @torch.no_grad()
    def hidden(ids):
        return target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states

    # ---- collect: per sequence the full h_k (context included, for the seq reader) and the targets at response
    # positions. Kept on the host in bf16.
    def collect(lo_i, n_tokens, cap_i):
        out, n, i = [], 0, lo_i
        while n < n_tokens and i < cap_i:
            ids, lo = seqs[i]
            i += 1
            hs = hidden(ids)
            idx = torch.arange(lo, len(ids))
            rec = {"idx": idx, "x": {k: hs[k][0].to(torch.bfloat16).cpu() for k in args.exits},
                   "y": torch.stack([hs[tgt_layer(t)][0, idx.to(dev)] for t in all_t], 1).to(torch.bfloat16).cpu()}
            out.append(rec)
            n += len(idx)
            del hs
        return out, i

    max_fit = int(0.6 * len(seqs))
    fit, i = collect(0, args.fit_tokens, max_fit)
    val, fit_end = collect(i, args.val_tokens, max_fit + (len(seqs) - max_fit) // 4)
    n_fit = sum(len(r["idx"]) for r in fit)
    n_val = sum(len(r["idx"]) for r in val)
    print(f"[data] fit {n_fit} tokens / {len(fit)} seqs, val {n_val} / {len(val)}, {time.time() - t0:.0f}s",
          flush=True)
    assert n_val > 0, "no validation tokens: add data or lower --fit-tokens"

    # standardisation statistics (fit data)
    def moments(get):
        s, s2, n = 0.0, 0.0, 0
        for r in fit:
            v = get(r).float()
            s = s + v.sum(0)
            s2 = s2 + (v * v).sum(0)
            n += len(v)
        mu = s / n
        return mu.to(dev), (s2 / n - mu * mu).clamp_min(1e-6).sqrt().to(dev)

    xstat = {k: moments(lambda r, k=k: r["x"][k][r["idx"]]) for k in args.exits}
    ystat = moments(lambda r: r["y"])                                  # [n_all_t, H] each
    tpos = {t: all_t.index(t) for t in all_t}

    def ccos(a, b, mu):
        return F.cosine_similarity(a - mu, b - mu, dim=-1)

    readers, report_close, train_log = {}, [], {}
    for k in args.exits:
        sel = torch.tensor([tpos[t] for t in tgts[k]])
        xm, xs = xstat[k]
        ym, ys = ystat[0][sel], ystat[1][sel]                          # [n_out, H]
        n_out = len(sel)

        # -- ridge on the same tokens
        acc = Ridge(H, tgts[k], H, dev)
        for r in fit:
            X = r["x"][k][r["idx"]].to(dev).float()
            Y = r["y"][:, sel].to(dev).float()
            acc.add(X, {t: Y[:, q] for q, t in enumerate(tgts[k])})
        best = {}
        for lam in args.lams:
            for t, Wm in acc.solve(lam).items():
                q = tgts[k].index(t)
                sc = torch.cat([ccos(apply_map(Wm, r["x"][k][r["idx"]].to(dev).float()),
                                     r["y"][:, sel[q]].to(dev).float(), ym[q]) for r in val]).mean().item()
                if t not in best or sc > best[t][0]:
                    best[t] = (sc, lam, Wm)
        del acc
        ridge = {t: best[t][2] for t in tgts[k]}

        def ridge_std(Xraw, ridge=ridge, ym=ym, ys=ys, tk=tuple(tgts[k])):
            """ridge prediction in standardised target space, [N, n_out, H] (fp32, as exp107)"""
            with torch.autocast("cuda", enabled=False):
                Xraw = Xraw.float()
                return torch.stack([(apply_map(ridge[t], Xraw) - ym[q]) / ys[q] for q, t in enumerate(tk)], 1)

        def predict_raw(name, Xraw_seq, pos, model=None, ridge_std=ridge_std, xm=xm, xs=xs, ym=ym, ys=ys):
            """reader predictions in raw space for positions pos of one sequence, [len(pos), n_out, H]"""
            Xp = Xraw_seq[pos]
            base = ridge_std(Xp)
            if name == "mlp":
                base = base + model((Xp - xm) / xs)
            elif name == "seq":
                full = model((Xraw_seq[: int(pos.max()) + 1] - xm) / xs)
                base = base + full[pos]
            return base.float() * ys + ym

        def val_score(name, model=None, recs=None):
            recs = val if recs is None else recs
            cs = [[] for _ in range(n_out)]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for r in recs:
                    Xs = r["x"][k].to(dev).float()
                    pos = r["idx"].to(dev)
                    P = predict_raw(name, Xs, pos, model).float()
                    Y = r["y"][:, sel].to(dev).float()
                    for q in range(n_out):
                        cs[q].append(ccos(P[:, q], Y[:, q], ym[q]).cpu())
            return [torch.cat(c).mean().item() for c in cs]

        fns = {"ridge": None}

        # -- mlp: flatten fit tokens
        if "mlp" in args.readers:
            Xf = torch.cat([r["x"][k][r["idx"]] for r in fit]).pin_memory()
            Yf = torch.cat([r["y"][:, sel] for r in fit]).pin_memory()
            model = MLPReader(H, n_out, args.mlp_width).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=0.01)
            best_sc, best_state, log = -1.0, None, []
            for step in range(args.mlp_steps):
                for g in opt.param_groups:
                    g["lr"] = args.mlp_lr * cosine_lr(step, args.mlp_steps, 200)
                b = torch.randint(0, len(Xf), (args.mlp_batch,))
                X = Xf[b].to(dev, non_blocking=True).float()
                Ystd = (Yf[b].to(dev, non_blocking=True).float() - ym) / ys
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = ridge_std(X) + model((X - xm) / xs)
                loss = F.mse_loss(pred.float(), Ystd)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if (step + 1) % args.eval_every == 0 or step + 1 == args.mlp_steps:
                    model.eval()
                    sc = sum(val_score("mlp", model)) / n_out
                    model.train()
                    log.append((step + 1, loss.item(), sc))
                    print(f"[mlp k={k}] step {step + 1} loss {loss.item():.4f} val ccos {sc:.4f}", flush=True)
                    if sc > best_sc:
                        best_sc, best_state = sc, copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            model.eval()
            fns["mlp"] = model
            train_log[f"mlp_k{k}"] = log
            del Xf, Yf, opt

        # -- seq: one sequence per forward, gradient accumulation
        if "seq" in args.readers:
            model = SeqReader(H, n_out, args.seq_d, args.seq_layers, args.seq_heads).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=args.seq_lr, weight_decay=0.01)
            best_sc, best_state, log = -1.0, None, []
            order = list(range(len(fit)))
            ptr = len(order)
            for step in range(args.seq_steps):
                for g in opt.param_groups:
                    g["lr"] = args.seq_lr * cosine_lr(step, args.seq_steps, 200)
                opt.zero_grad(set_to_none=True)
                tot = 0.0
                for _ in range(args.seq_accum):
                    if ptr >= len(order):
                        random.shuffle(order)
                        ptr = 0
                    r = fit[order[ptr]]
                    ptr += 1
                    Xs = r["x"][k].to(dev).float()
                    pos = r["idx"].to(dev)
                    Ystd = (r["y"][:, sel].to(dev).float() - ym) / ys
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = ridge_std(Xs[pos]) + model((Xs - xm) / xs)[pos]
                    loss = F.mse_loss(pred.float(), Ystd) / args.seq_accum
                    loss.backward()
                    tot += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if (step + 1) % args.eval_every == 0 or step + 1 == args.seq_steps:
                    model.eval()
                    sc = sum(val_score("seq", model)) / n_out
                    model.train()
                    log.append((step + 1, tot, sc))
                    print(f"[seq k={k}] step {step + 1} loss {tot:.4f} val ccos {sc:.4f}", flush=True)
                    if sc > best_sc:
                        best_sc, best_state = sc, copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            model.eval()
            fns["seq"] = model
            train_log[f"seq_k{k}"] = log
            del opt

        # closeness: train (first 40 fit sequences) and val
        for name in args.readers:
            tr = val_score(name, fns[name], fit[:40])
            va = val_score(name, fns[name])
            for q, t in enumerate(tgts[k]):
                report_close.append({"k": k, "reader": name,
                                     "target": t if t == "final" else f"L{lids[int(t[3:])]}",
                                     "train ccos": tr[q], "val ccos": va[q]})
        readers[k] = (fns, predict_raw)
        print(f"[fit] k={k} done, {time.time() - t0:.0f}s", flush=True)
    del fit, val
    torch.cuda.empty_cache()

    print_table(report_close, list(report_close[0]), "Closeness: centred cosine of the read state to the real one "
                                                     "(train ~ val: capacity is not the limit)")

    # ---- evaluation on held-out sequences (as exp107) ----
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

    with torch.no_grad():                      # no autocast: drafter numerics as in exp107
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
            # all reader predictions for this sequence, once (positions 0..T-1)
            allpos = torch.arange(T, device=dev)
            preds = {}
            for k in args.exits:
                fns, predict_raw = readers[k]
                Xs = hs[k][0].float()
                for name in args.readers:
                    preds[(k, name)] = predict_raw(name, Xs, allpos, fns[name]).to(hs[-1].dtype)
                fin = tgts[k].index("final")
                for name in args.readers:
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
