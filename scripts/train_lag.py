"""Fine-tune the DFlash drafter to draft with lagged hidden states (see hspec/lagtrain.py).

  --max-lag 16   lag-tolerant drafter (lag 0 with probability --p-zero, else 1..16)
  --max-lag 0    control: same data and steps, standard blocks only

Features come from the frozen bf16 target; the drafter's embeddings and LM head are the
target's (frozen), only the drafter's own layers are trained.

  python scripts/train_lag.py --data /scratch-shared/$USER/hspec_lagft/data --out .../drafter_lag16
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.lagtrain import block_loss, sample_blocks  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lag", type=int, default=16)
    ap.add_argument("--p-zero", type=float, default=0.25)
    ap.add_argument("--blocks", type=int, default=24, help="blocks sampled per sequence")
    ap.add_argument("--batch", type=int, default=4, help="sequences per optimizer step")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=250)
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
    print(f"{len(train)} train / {len(val)} val sequences, "
          f"{sum(len(d['ids']) - d['n_prompt'] for d in train)} response tokens", flush=True)

    target = load_target(args.target)
    for p in target.parameters():
        p.requires_grad_(False)
    draft = load_draft(args.draft)
    draft.train()
    params = [p for p in draft.parameters() if p.requires_grad]
    print(f"trainable drafter parameters: {sum(p.numel() for p in params) / 1e6:.0f} M", flush=True)
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = int(len(train) * args.epochs / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    bs = draft.block_size

    def feats_of(ids):
        with torch.no_grad():
            out = target(ids[None], output_hidden_states=True, logits_to_keep=1)
            return extract_context_feature(out.hidden_states, draft.target_layer_ids)

    def evaluate():
        draft.eval()
        vrng = random.Random(123)
        res = {}
        with torch.no_grad():
            for lag in (0, 4, 8):
                losses, accs = [], []
                for d in val:
                    ids = d["ids"].long().to(args.device)
                    blocks = [(s, min(lag, s - 1)) for s, _ in
                              sample_blocks(d["n_prompt"], len(ids), 8, bs, 0, 1.0, vrng)]
                    if not blocks:
                        continue
                    l, a = block_loss(draft, target, feats_of(ids), ids, blocks, bs, args.gamma)
                    losses.append(float(l))
                    accs.append(float(a))
                res[lag] = (sum(losses) / len(losses), sum(accs) / len(accs))
        draft.train()
        return " ".join(f"lag{k}: loss {v[0]:.3f} acc1 {v[1]:.3f}" for k, v in res.items())

    print(f"[step 0] val {evaluate()}", flush=True)
    step, t0 = 0, time.time()
    order = list(range(len(train)))
    while step < total:
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if step >= total:
                break
            opt.zero_grad(set_to_none=True)
            tot_l, tot_a, n = 0.0, 0.0, 0
            for j in order[i : i + args.batch]:
                d = train[j]
                ids = d["ids"].long().to(args.device)
                blocks = sample_blocks(d["n_prompt"], len(ids), args.blocks, bs, args.max_lag, args.p_zero, rng)
                if not blocks:
                    continue
                loss, acc = block_loss(draft, target, feats_of(ids), ids, blocks, bs, args.gamma)
                (loss / args.batch).backward()
                tot_l += float(loss)
                tot_a += float(acc)
                n += 1
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[step {step}/{total}] loss {tot_l / max(n, 1):.3f} acc1 {tot_a / max(n, 1):.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0:
                print(f"[step {step}] val {evaluate()}", flush=True)
    print(f"[final] val {evaluate()}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    draft.eval()
    draft.save_pretrained(out)
    print(f"saved drafter to {out}")


if __name__ == "__main__":
    main()
