"""Fusion vs K/V injection ablation (see hspec/fusion.py).

Fine-tunes the DFlash drafter from the official checkpoint with real target features, one mode
per run, same data / steps / seed for every mode. The kv run is the control (same fine-tune,
DFlash architecture). Validation reports per-offset top-1 accuracy and tau = mean accepted
tokens per block under greedy verification (leading correct drafts + 1), on held-out
target-greedy sequences.

  python scripts/train_fusion.py --mode fuse_all --data /scratch-shared/$USER/hspec_lagft/data \
      --out /scratch-shared/$USER/hspec_fusion/fuse_all
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.fusion import MODES, Fuser, accepted, block_loss  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=24, help="blocks sampled per sequence")
    ap.add_argument("--batch", type=int, default=4, help="sequences per optimizer step")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--fuse-lr", type=float, default=3e-4, help="lr of the new fusion maps")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
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
    if args.max_train:
        train = train[: args.max_train]
    print(f"[{args.mode}] {len(train)} train / {len(val)} val sequences", flush=True)

    target = load_target(args.target)
    for p in target.parameters():
        p.requires_grad_(False)
    draft = load_draft(args.draft)
    H = draft.config.hidden_size
    fuser = Fuser(H, len(draft.layers), args.mode).to(args.device, torch.bfloat16)
    draft.train()
    fuser.train()
    dparams = [p for p in draft.parameters() if p.requires_grad]
    fparams = list(fuser.parameters())
    import bitsandbytes as bnb
    groups = [{"params": dparams, "lr": args.lr}]
    if fparams:
        groups.append({"params": fparams, "lr": args.fuse_lr})
    opt = bnb.optim.AdamW8bit(groups, weight_decay=0.0, betas=(0.9, 0.95))
    print(f"[{args.mode}] trainable: drafter {sum(p.numel() for p in dparams) / 1e6:.0f} M, "
          f"fusion {sum(p.numel() for p in fparams) / 1e6:.1f} M", flush=True)
    total = int(len(train) * args.epochs / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    bs = draft.block_size

    def feats_of(ids):
        with torch.no_grad():
            out = target(ids[None], output_hidden_states=True, logits_to_keep=1)
            return extract_context_feature(out.hidden_states, draft.target_layer_ids)

    def anchors_of(d, n, r):
        lo, hi = max(d["n_prompt"], 1), len(d["ids"]) - 2
        return r.sample(range(lo, hi), min(n, hi - lo)) if hi > lo else []

    history = []

    def evaluate(step):
        draft.eval()
        fuser.eval()
        vrng = random.Random(123)
        losses, taus = [], []
        hit = [[0, 0] for _ in range(bs - 1)]
        with torch.no_grad():
            for d in val:
                ids = d["ids"].long().to(args.device)
                anchors = anchors_of(d, 16, vrng)
                if not anchors:
                    continue
                loss, correct = block_loss(draft, fuser, target, feats_of(ids), ids, anchors, bs, args.gamma)
                losses.append(float(loss))
                for row in correct:
                    if None in row:          # block runs past the end: skip for tau
                        continue
                    taus.append(accepted(row))
                    for k, c in enumerate(row):
                        hit[k][0] += int(c)
                        hit[k][1] += 1
        draft.train()
        fuser.train()
        acc = [h / max(n, 1) for h, n in hit]
        rec = {"step": step, "loss": sum(losses) / len(losses), "tau": sum(taus) / len(taus), "acc": acc}
        history.append(rec)
        print(f"[{args.mode} step {step}] val loss {rec['loss']:.3f} tau {rec['tau']:.3f} "
              f"acc@1 {acc[0]:.3f} acc@4 {acc[3]:.3f} acc@8 {acc[7]:.3f} acc@15 {acc[-1]:.3f}", flush=True)

    evaluate(0)
    step, t0 = 0, time.time()
    order = list(range(len(train)))
    while step < total:
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if step >= total:
                break
            opt.zero_grad(set_to_none=True)
            tot, n = 0.0, 0
            for j in order[i : i + args.batch]:
                d = train[j]
                ids = d["ids"].long().to(args.device)
                anchors = anchors_of(d, args.blocks, rng)
                if not anchors:
                    continue
                loss, _ = block_loss(draft, fuser, target, feats_of(ids), ids, anchors, bs, args.gamma)
                (loss / args.batch).backward()
                tot += float(loss)
                n += 1
            torch.nn.utils.clip_grad_norm_(dparams + fparams, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[{args.mode} step {step}/{total}] loss {tot / max(n, 1):.3f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0:
                evaluate(step)
    evaluate(step)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    draft.eval()
    draft.save_pretrained(out)
    torch.save({"mode": args.mode, "state": fuser.state_dict()}, out / "fuser.pt")
    (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    print(f"[{args.mode}] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
