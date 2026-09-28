"""Train early-exit heads on the frozen target (see hspec/exit.py), distilling the target's
next-token distribution (KL) at sampled response positions. All (kind, k) heads train in one
loop off a single target forward per sequence.

  python scripts/train_exit.py --data /scratch-shared/$USER/hspec_lagft/data --out .../exit_heads.pt
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.exit import ExitHead, head_name, kl_loss, save_heads  # noqa: E402
from hspec.models import load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[12, 18, 24, 30])
    ap.add_argument("--kinds", nargs="+", default=["lin", "layer"])
    ap.add_argument("--rows", type=int, default=256, help="sampled positions per sequence")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr-lin", type=float, default=1e-4)
    ap.add_argument("--lr-layer", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw8bit", "adamw"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
    data = [d for d in data if len(d["ids"]) <= args.max_len]
    rng.shuffle(data)
    val, train = data[: args.val_seqs], data[args.val_seqs :]
    print(f"{len(train)} train / {len(val)} val sequences", flush=True)

    target = load_target(args.target, args.device)
    for p in target.parameters():
        p.requires_grad_(False)
    L = target.config.num_hidden_layers
    assert all(0 < k < L for k in args.layers), f"layers must be in 1..{L - 1}"
    heads = {head_name(kind, k): ExitHead(target, kind, k) for kind in args.kinds for k in args.layers}
    raws = {head_name("raw", k): ExitHead(target, "raw", k) for k in args.layers}
    groups = [{"params": h.trainable(), "lr": args.lr_lin if h.kind == "lin" else args.lr_layer} for h in heads.values()]
    for h in heads.values():
        h.train()
        for p in h.trainable():
            p.requires_grad_(True)
    print("trainable parameters: " + ", ".join(f"{n} {sum(p.numel() for p in h.trainable()) / 1e6:.0f}M"
                                              for n, h in heads.items()), flush=True)
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    total = int(len(train) * args.epochs / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def fwd(ids):
        with torch.no_grad():
            out = target(ids[None], output_hidden_states=True)
        return out.hidden_states, out.logits[0]

    def rows_of(d, n, r):
        lo, hi = d["n_prompt"] - 1, len(d["ids"]) - 1          # positions predicting response tokens
        idx = list(range(lo, hi))
        return torch.tensor(sorted(r.sample(idx, min(n, len(idx)))), device=args.device)

    def evaluate():
        for h in heads.values():
            h.eval()
        agree = {n: [0, 0] for n in list(raws) + list(heads)}
        with torch.no_grad():
            for d in val:
                ids = d["ids"].long().to(args.device)
                hs, logits = fwd(ids)
                rows = torch.arange(d["n_prompt"] - 1, len(ids) - 1, device=args.device)
                ref = logits[rows].argmax(-1)
                for n, h in {**raws, **heads}.items():
                    pred = h.logits(hs[h.k], rows=rows)[0].argmax(-1)
                    agree[n][0] += int((pred == ref).sum())
                    agree[n][1] += len(rows)
        for h in heads.values():
            h.train()
        return " ".join(f"{n} {a / max(b, 1):.3f}" for n, (a, b) in agree.items())

    print(f"[step 0] val top1 agreement: {evaluate()}", flush=True)
    step, t0 = 0, time.time()
    order = list(range(len(train)))
    while step < total:
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if step >= total:
                break
            opt.zero_grad(set_to_none=True)
            tot = {n: 0.0 for n in heads}
            for j in order[i : i + args.batch]:
                d = train[j]
                ids = d["ids"].long().to(args.device)
                hs, logits = fwd(ids)
                rows = rows_of(d, args.rows, rng)
                teacher = F.log_softmax(logits[rows].float(), -1)
                del logits
                loss = 0.0
                for n, h in heads.items():
                    l = kl_loss(h.logits(hs[h.k], rows=rows)[0], teacher)
                    tot[n] += float(l)
                    loss = loss + l
                (loss / args.batch).backward()
            for g in groups:
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[step {step}/{total}] kl " + " ".join(f"{n} {v / args.batch:.3f}" for n, v in tot.items())
                      + f" {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0:
                print(f"[step {step}] val top1 agreement: {evaluate()}", flush=True)
    print(f"[final] val top1 agreement: {evaluate()}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_heads(heads, args.out)
    print(f"saved heads to {args.out}")


if __name__ == "__main__":
    main()
