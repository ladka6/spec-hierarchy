"""Feasibility test: latent block diffusion over the verifier's hidden states (hspec/statediff.py).

Question: can a meaningfully small denoiser (initialised from Qwen3-0.6B, optionally truncated)
predict the 8B target's states over the next block well enough to draft from them, and does
iterating (K denoising steps) buy what extra depth would (iteration substituting for depth)?

Training: frozen target gives real states; per sequence `--blocks` anchors s; loss on standardised
states (MSE + per-slice cosine) + token CE through the target head on the predicted last state.
Validation (same 48 held-out sequences as the other runs, 8 anchors each), for K in 1, 2, 4, 8:
  cos_Lx       cosine of predicted vs true state per slice
  agree        predicted next token == target's token, over the block
  tau          1 (bonus x_{s+1}) + leading correct drafted tokens x_{s+2} .. (as DFlash's tau)
  tau_dflash   DFlash (original drafter, real features up to s, anchor x_{s+1}) on the same anchors

  python scripts/train_statediff.py --data .../hspec_lagft/data --out .../statediff_L28 --n-layers 28
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.statediff import StateDenoiser, StateStats, decode_tokens, target_states  # noqa: E402


def lead(hits):
    n = 0
    while n < len(hits) and hits[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16", help="for layer ids and the DFlash reference")
    ap.add_argument("--small", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-layers", type=int, default=0, help="first n layers of the small model (0 = all)")
    ap.add_argument("--regress", action="store_true", help="train only at t = 0 (single-pass predictor control)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=8, help="anchors per sequence")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5, help="pretrained layers")
    ap.add_argument("--lr-new", type=float, default=3e-4, help="new projections / embeddings")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--p-zero", type=float, default=0.3, help="share of training blocks at t = 0")
    ap.add_argument("--lam-cos", type=float, default=1.0)
    ap.add_argument("--lam-tok", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--steps-eval", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    tag = f"L{args.n_layers or 'all'}" + ("_regress" if args.regress else "")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
    data = [d for d in data if len(d["ids"]) <= args.max_len]
    rng.shuffle(data)
    val, train = data[: args.val_seqs], data[args.val_seqs :]

    target = load_target(args.target, dev)
    target.requires_grad_(False)
    draft = load_draft(args.draft, dev)
    lids = list(draft.target_layer_ids)
    small = load_target(args.small, dev)
    H = target.config.hidden_size
    S = len(lids) + 1                               # slices: DFlash layers + last hidden
    D = S * H
    B = args.block

    @torch.no_grad()
    def tpass(ids):
        out = target(ids[None], output_hidden_states=True)
        return target_states(out.hidden_states, lids)[0], out.logits[0].argmax(-1)

    # standardisation statistics + whether the last hidden state is already normed
    with torch.no_grad():
        xs = []
        for d in train[:40]:
            st, _ = tpass(d["ids"].long().to(dev))
            xs.append(st[4:])
        stats = StateStats.fit(xs)
        ids0 = train[0]["ids"].long().to(dev)
        st0, tp0 = tpass(ids0)
        normed = bool((target.lm_head(st0[:, -H:]).argmax(-1) == tp0).float().mean() > 0.99)
    last = slice(D - H, D)
    stats_last = StateStats(stats.mean[last], stats.std[last])
    print(f"[{tag}] state dim {D} ({S} slices), last hidden normed: {normed}", flush=True)

    model = StateDenoiser(small, D, args.n_layers or None).to(dev).float()
    del small
    n_layer_params = sum(p.numel() for p in model.layers.parameters())
    print(f"[{tag}] denoiser layers {len(model.layers)}, {n_layer_params / 1e6:.0f} M layer params "
          f"({n_layer_params / sum(p.numel() for p in target.model.layers.parameters()):.3f} of the target's), "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f} M total", flush=True)
    groups = [{"params": model.pretrained_params(), "lr": args.lr}, {"params": model.new_params(), "lr": args.lr_new}]
    if dev == "cpu":
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=0.0, betas=(0.9, 0.95))
    total = int(len(train) * args.epochs / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def anchors(d, n, r):
        T = len(d["ids"])
        lo, hi = max(d["n_prompt"], 4), T - B - 2
        return r.sample(range(lo, hi), min(n, hi - lo)) if hi > lo else []

    def loss_fn(ids, st_n, tp, blocks):
        x0 = torch.stack([st_n[s + 1 : s + 1 + B] for s in blocks])            # [nb, B, D]
        nb = len(blocks)
        t = torch.zeros(nb, device=dev) if args.regress else torch.rand(nb, device=dev)
        if not args.regress:
            t = torch.where(torch.rand(nb, device=dev) < args.p_zero, torch.zeros_like(t), t)
        eps = torch.randn_like(x0)
        xt = t[:, None, None] * x0 + (1 - t[:, None, None]) * eps
        pred = model(ids, st_n, blocks, xt, t, B)
        mse = F.mse_loss(pred, x0)
        cos = torch.stack([1 - F.cosine_similarity(pred[..., j * H : (j + 1) * H], x0[..., j * H : (j + 1) * H],
                                                  dim=-1).mean() for j in range(S)]).mean()
        logits = decode_tokens(pred[..., last], stats_last, target, normed)    # predicts x_{p+1} from state p
        lab = torch.stack([tp[s + 1 : s + 1 + B] for s in blocks])
        tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1))
        return mse + args.lam_cos * cos + args.lam_tok * tok, (mse, cos, tok)

    history = []

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        vr = random.Random(123)
        gen = torch.Generator(device=dev).manual_seed(0)
        res = {K: {"cos": [[] for _ in range(S)], "agree": [], "tau": []} for K in args.steps_eval}
        tau_df = []
        for d in val:
            ids = d["ids"].long().to(dev)
            blocks = anchors(d, 8, vr)
            if not blocks:
                continue
            st, tp = tpass(ids)
            st_n = stats.norm(st)
            feats = st[None, :, : D - H].to(next(draft.parameters()).dtype)
            for s in blocks:                           # DFlash reference: anchor s+1, features up to s
                lg = draft_logits_lagged(draft, target, feats, ids, s + 1, 0, draft.block_size)
                truth = ids[s + 2 : s + 2 + lg.shape[0]]
                tau_df.append(1 + lead((lg[: len(truth)].argmax(-1) == truth).tolist()))
            for K in args.steps_eval:
                pred = model.sample(ids, st_n, blocks, B, D, K, generator=gen)
                x0 = torch.stack([st_n[s + 1 : s + 1 + B] for s in blocks])
                for j in range(S):
                    res[K]["cos"][j].append(F.cosine_similarity(pred[..., j * H : (j + 1) * H],
                                                                 x0[..., j * H : (j + 1) * H], dim=-1).mean().item())
                tok = decode_tokens(pred[..., last], stats_last, target, normed).argmax(-1)    # x_{s+2} ..
                for b, s in enumerate(blocks):
                    truth = ids[s + 2 : s + 2 + B]
                    hits = (tok[b, : len(truth)] == truth).tolist()
                    res[K]["agree"].append(sum(hits) / max(len(hits), 1))
                    res[K]["tau"].append(1 + lead(hits))
        model.train()
        rec = {"step": step, "tau_dflash": sum(tau_df) / len(tau_df)}
        for K, r in res.items():
            rec[K] = {"tau": sum(r["tau"]) / len(r["tau"]), "agree": sum(r["agree"]) / len(r["agree"]),
                      "cos": [sum(c) / len(c) for c in r["cos"]]}
        history.append(rec)
        print(f"[{tag} step {step}] tau_dflash {rec['tau_dflash']:.2f} | " + " | ".join(
            f"K{K}: tau {rec[K]['tau']:.2f} agree {rec[K]['agree']:.3f} cos "
            + "/".join(f"{c:.2f}" for c in rec[K]["cos"]) for K in args.steps_eval), flush=True)

    evaluate(0)
    step, t0 = 0, time.time()
    order = list(range(len(train)))
    model.train()
    while step < total:
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if step >= total:
                break
            opt.zero_grad(set_to_none=True)
            parts = [0.0, 0.0, 0.0]
            n = 0
            for j in order[i : i + args.batch]:
                d = train[j]
                ids = d["ids"].long().to(dev)
                blocks = anchors(d, args.blocks, rng)
                if not blocks:
                    continue
                st, tp = tpass(ids)
                loss, (m, c, tk) = loss_fn(ids, stats.norm(st), tp, blocks)
                (loss / args.batch).backward()
                parts = [parts[0] + float(m), parts[1] + float(c), parts[2] + float(tk)]
                n += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[{tag} step {step}/{total}] mse {parts[0] / max(n, 1):.3f} cos {parts[1] / max(n, 1):.3f} "
                      f"tok {parts[2] / max(n, 1):.3f} {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0:
                evaluate(step)
    evaluate(step)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(args), "state": model.state_dict(), "mean": stats.mean, "std": stats.std,
                "normed": normed}, out / "statediff.pt")
    (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1, default=str))
    print(f"[{tag}] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
