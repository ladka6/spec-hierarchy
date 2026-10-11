"""Distil the small tracker (Qwen3-0.6B) on the target's own next-token distributions.

The multi-round refresh loop is limited by the tracker's token agreement with the target (0.80 for
Qwen3-0.6B vs 0.97 for a 4-bit copy): it decides where to restart and which token to insert. This
fine-tunes the small model with forward KL to the frozen target (Qwen3-8B) on the target-greedy
training sequences, response tokens only, and reports greedy agreement on the 48 held-out sequences.
The result is saved with save_pretrained, so exp32 / exp29 can use it via --small <dir>.

  python scripts/train_tracker_kd.py --data .../hspec_lagft/data --out .../tracker_kd_06b
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

from hspec.models import load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--small", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--temp", type=float, default=1.0, help="distillation temperature")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
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
    small = load_target(args.small, dev).float()
    small.train()
    params = [p for p in small.parameters() if p.requires_grad]
    if dev == "cpu":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = int(len(train) * args.epochs / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    @torch.no_grad()
    def evaluate(step):
        small.eval()
        hit = n = 0
        kl = []
        for d in val:
            ids = d["ids"].long().to(dev)
            r = slice(d["n_prompt"] - 1, len(ids) - 1)
            tl = target(ids[None]).logits[0, r].float()
            sl = small(ids[None]).logits[0, r].float()
            hit += int((sl.argmax(-1) == tl.argmax(-1)).sum())
            n += sl.shape[0]
            kl.append(F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True,
                               reduction="batchmean").item())
        small.train()
        print(f"[kd step {step}] val agreement with target {hit / n:.4f}  KL {sum(kl) / len(kl):.3f}", flush=True)

    evaluate(0)
    step, t0 = 0, time.time()
    order = list(range(len(train)))
    while step < total:
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if step >= total:
                break
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for j in order[i : i + args.batch]:
                d = train[j]
                ids = d["ids"].long().to(dev)
                r = slice(d["n_prompt"] - 1, len(ids) - 1)
                with torch.no_grad():
                    tl = target(ids[None]).logits[0, r].float() / args.temp
                sl = small(ids[None]).logits[0, r].float() / args.temp
                loss = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True,
                                reduction="batchmean") * args.temp ** 2
                (loss / args.batch).backward()
                tot += float(loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[kd step {step}/{total}] KL {tot / args.batch:.3f} {(time.time() - t0) / step:.2f}s/step",
                      flush=True)
            if step % args.eval_every == 0:
                evaluate(step)
    evaluate(step)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    small.to(torch.bfloat16).save_pretrained(out)
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.small).save_pretrained(out)
    except Exception:  # noqa: BLE001
        pass
    print(f"[kd] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
