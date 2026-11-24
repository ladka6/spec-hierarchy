"""Experiment 118: is 3/4 depth the best place to read the target, or only what THIS drafter learned to read?

exp111 says the drafter lives on its tap at about 3/4 depth; exp115 says the last layer holds at least as much
about the coming tokens. exp111 measures the trained drafter's preference: its fusion layer (fc: 5 taps -> one
vector) was trained once, with all taps present. Here the interface is refitted: for ONE target layer at a time a
new projection of the same size (H -> drafter width) replaces the fusion, the drafter itself stays frozen, and
every depth gets the same training. If 3/4 depth still wins, the preference is a property of the target's states;
if the last layer (or an earlier one) wins, it was the drafter's conditioning.

Per depth, two uses, each with its own projection:
  all    every context position is fed from that one layer
  last   only the last m positions are (m sampled from --ms in training); older positions keep the original
         fusion of all real taps. This is the in-flight situation, and tells which single state is worth producing.
Projection: closed-form start (ridge map from the layer's state to the original fused feature), then trained through
the frozen drafter with the DFlash block loss (--steps; 0 = ridge only). Rows report both.

Table (held-out anchors of exp112, tau = accepted drafted tokens + 1):
  ridge / trained, for "all" and for the last 8 and 16 positions; references: the original drafter with all taps,
  and no features at all for the last m positions.

  python scripts/exp118_tap_depth.py --data /scratch-shared/$USER/hspec_lagft/data
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp107_lift_diag import load_seqs  # noqa: E402
from exp108b_nonlinear_lift import cosine_lr  # noqa: E402

from hspec.lagtrain import block_loss  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

BF = torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--depths", nargs="+", type=float, default=[0.25, 0.47, 0.58, 0.72, 0.83, 0.94, 1.0],
                    help="fractions of the depth; the state after layer round(f * L) - 1 = hidden_states[round(f * L)]")
    ap.add_argument("--modes", nargs="+", default=["all", "last"])
    ap.add_argument("--ms", nargs="+", type=int, default=[4, 8, 16], help="last-m lengths sampled in training")
    ap.add_argument("--eval-ms", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--blocks", type=int, default=8, help="anchors per sequence per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--val-seqs", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--fit-seqs", type=int, default=150)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--fit-tokens", type=int, default=600_000, help="as exp112: fixes the train / held-out split")
    ap.add_argument("--val-tokens", type=int, default=15_000)
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--save", default="exp118_tap_depth")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    assert all(m in ("all", "last") for m in args.modes)
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
    H, L = target.config.hidden_size, len(target.model.layers)
    fc0 = draft.fc                                           # the original fusion: 5 taps -> drafter width
    Hd = fc0.out_features
    draft.fc = nn.Identity()                                 # from here on the drafter is fed FUSED features
    lis = sorted({max(1, min(L, round(f * L))) for f in args.depths})
    Mx = max(args.ms + args.eval_ms)
    print(f"[setup] {len(seqs)} sequences, {L} layers, taps {lids}, depths (hidden_states index) {lis}, "
          f"modes {args.modes}, {time.time() - t0:.0f}s", flush=True)

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
    val_seqs = val_recs[: args.val_seqs]
    rest = seqs[fit_end:]
    cand_a = [(si, s) for si, (ids, lo) in enumerate(rest) for s in range(max(lo, Mx + 2), len(ids) - bs)]
    random.Random(1).shuffle(cand_a)
    by_seq = {}
    for si, s in sorted(cand_a[: args.anchors]):
        by_seq.setdefault(si, []).append(s)

    @torch.no_grad()
    def hidden(ids):
        return target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states

    @torch.no_grad()
    def fused(hs):
        """the original fused feature of every position [1, T, Hd] (what the unmodified drafter is fed)"""
        return fc0(extract_context_feature(hs, lids))

    def tau_of(fz, x, s, truth, lag=0):
        with torch.no_grad(), torch.autocast("cuda", dtype=BF):
            return matched(blocks_logits(draft, target, fz, x, [(s, lag)], bs)[0].argmax(-1).tolist(), truth) + 1

    def mean(v):
        return sum(v) / max(len(v), 1)

    # ---- closed-form start: ridge map from each layer's state to the original fused feature ----
    with torch.no_grad():
        xtx = {li: torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev) for li in lis}
        xty = {li: torch.zeros(H + 1, Hd, dtype=torch.float64, device=dev) for li in lis}
        for si in fit_seqs[: args.fit_seqs]:
            ids, lo = seqs[si]
            hs = hidden(ids)
            y = fused(hs)[0, lo:].double()
            for li in lis:
                x = hs[li][0, lo:].double()
                w = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                xa = torch.cat([x, torch.ones_like(x[:, :1])], -1) * w
                xtx[li] += xa.T @ xa
                xty[li] += xa.T @ (y * w)
            del hs
        sols = {}
        for li in lis:
            A = xtx[li] + args.ridge * xtx[li].diagonal().mean() * torch.eye(H + 1, dtype=torch.float64, device=dev)
            sols[li] = torch.linalg.solve(A, xty[li]).float()
        del xtx, xty
    print(f"[fit] ridge maps for {len(lis)} depths, {time.time() - t0:.0f}s", flush=True)

    def new_proj(li):
        p = nn.Linear(H, Hd, bias=True).to(dev)
        with torch.no_grad():
            p.weight.copy_(sols[li][:H].T)
            p.bias.copy_(sols[li][H])
        return p

    def feats_for(proj, hs, li, fz0, s, mode, m):
        """fused features of positions < s: from `proj` of layer li everywhere (all) or at the last m positions"""
        with torch.autocast("cuda", dtype=BF):
            if mode == "all":
                return proj(hs[li][:, :s])
            new = proj(hs[li][:, s - m : s])
            return torch.cat([fz0[:, : s - m], new.to(fz0.dtype)], 1)

    def loss_fn(proj, si, mode, n_blocks, grad):
        ids, lo = seqs[si]
        T = len(ids)
        if T - bs - max(lo, Mx + 2) < 1:
            return None
        hs = hidden(ids)
        fz0 = fused(hs)
        x = ids.to(dev)
        tot = 0.0
        with torch.set_grad_enabled(grad):
            for _ in range(n_blocks):
                s = rng.randrange(max(lo, Mx + 2), T - bs)
                fz = feats_for(proj, hs, li_cur[0], fz0, s, mode, rng.choice(args.ms))
                with torch.autocast("cuda", dtype=BF):
                    tot = tot + block_loss(draft, target, fz, x, [(s, 0)], bs, args.gamma)[0]
        return tot / n_blocks

    li_cur = [lis[0]]

    def train(tag, proj, mode):
        opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.0)

        def scorer():
            st = rng.getstate()
            rng.seed(123)
            tot, n = 0.0, 0
            for si in val_seqs:
                out = loss_fn(proj, si, mode, 4, False)
                if out is not None:
                    tot += float(out)
                    n += 1
            rng.setstate(st)
            return -tot / max(n, 1)

        best_sc = scorer()
        best_state = {k: v.detach().clone() for k, v in proj.state_dict().items()}
        print(f"[{tag}] start val {best_sc:.4f}", flush=True)
        for step in range(args.steps):
            for g in opt.param_groups:
                g["lr"] = args.lr * cosine_lr(step, args.steps, min(100, max(args.steps // 10, 1)))
            loss = None
            for _ in range(10):
                loss = loss_fn(proj, rng.choice(fit_seqs), mode, args.blocks, True)
                if loss is not None:
                    break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step()
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                sc = scorer()
                print(f"[{tag}] step {step + 1} loss {float(loss):.4f} val {sc:.4f} {time.time() - t0:.0f}s", flush=True)
                if sc > best_sc:
                    best_sc = sc
                    best_state = {k: v.detach().clone() for k, v in proj.state_dict().items()}
        proj.load_state_dict(best_state)

    ref_cache = {}

    @torch.no_grad()
    def evaluate(projs):
        """projs: {name: (proj, mode)}; returns {name: {scope: tau}} with scopes "all", "m8", ..."""
        out = {n: {} for n in projs}
        acc = {(n, sc): [] for n, (_, mode) in projs.items()
               for sc in (["all"] if mode == "all" else [f"m{m}" for m in args.eval_ms])}
        ref = {k: [] for k in ["real"] + [f"removed m{m}" for m in args.eval_ms]}
        for si, anchors in by_seq.items():
            ids, _lo = rest[si]
            x = ids.to(dev)
            hs = hidden(ids)
            fz0 = fused(hs)
            truth_all = x.tolist()
            for s in anchors:
                truth = truth_all[s + 1 :]
                if (si, s) not in ref_cache:
                    r = {"real": tau_of(fz0[:, :s], x, s, truth)}
                    for m in args.eval_ms:
                        r[f"removed m{m}"] = tau_of(fz0[:, : s - m], x, s, truth, m)
                    ref_cache[(si, s)] = r
                for k, v in ref_cache[(si, s)].items():
                    ref[k].append(v)
                for n, (proj, mode) in projs.items():
                    if mode == "all":
                        acc[(n, "all")].append(tau_of(feats_for(proj, hs, li_cur[0], fz0, s, "all", 0), x, s, truth))
                    else:
                        for m in args.eval_ms:
                            acc[(n, f"m{m}")].append(
                                tau_of(feats_for(proj, hs, li_cur[0], fz0, s, "last", m), x, s, truth))
            del hs, fz0
        for (n, sc), v in acc.items():
            out[n][sc] = mean(v)
        return out, {k: mean(v) for k, v in ref.items()}

    rows, refs = [], {}
    for li in lis:
        li_cur[0] = li
        projs = {}
        for mode in args.modes:
            projs[f"ridge {mode}"] = (new_proj(li), mode)
            if args.steps:
                p = new_proj(li)
                train(f"layer {li - 1} {mode}", p, mode)
                projs[f"trained {mode}"] = (p, mode)
        res, refs = evaluate(projs)
        row = {"layer": li - 1, "depth": li / L, "tap": "tap" if li - 1 in lids else ""}
        for n, d in res.items():
            for sc, v in d.items():
                row[f"{n.split()[0]} {sc}"] = v
        rows.append(row)
        print_table([row], list(row), f"layer {li - 1}")
        del projs
        torch.cuda.empty_cache()

    print_table([refs], list(refs), "References (tau): original drafter with all taps / no features for the last m")
    print_table(rows, list(rows[0]), "One target layer through a refitted projection, frozen drafter (held-out anchors)")
    save_json(args.save, {"args": vars(args), "rows": rows, "refs": refs})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
