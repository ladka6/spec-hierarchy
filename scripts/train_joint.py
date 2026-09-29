"""Joint drafter training over several targets (upper bound for one shared drafter).

A DFlash drafter learns from its target's hidden states, so every training sequence must be
encoded by the target that produced it. Several 8B targets do not fit on one GPU together, so
training rotates: each round loads one target at a time (random order) and trains on the next
chunk of that target's data, until every target's data has been used once per epoch. The
drafter, optimizer and schedule persist across swaps; only the frozen target is reloaded.

  python scripts/train_joint.py --pairs /path/math200=/data_math200 /path/code200=/data_code200 ... \
      --out .../drafter_joint
"""

from __future__ import annotations

import argparse
import gc
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
    ap.add_argument("--pairs", nargs="+", required=True, help="target_path=data_dir")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4, help="target swaps per epoch = rounds x targets")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=32)
    ap.add_argument("--max-per-target", type=int, default=0, help="cap training sequences per target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw8bit", "adamw"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    sets = []
    for pair in args.pairs:
        tpath, dpath = pair.split("=", 1)
        data = []
        for f in sorted(Path(dpath).glob("shard*.pt")):
            data += torch.load(f)
        data = [d for d in data if len(d["ids"]) <= args.max_len]
        random.Random(args.seed).shuffle(data)
        train = data[args.val_seqs :]
        if args.max_per_target:
            train = train[: args.max_per_target]
        sets.append({"target": tpath, "val": data[: args.val_seqs], "train": train})
        print(f"{tpath}: {len(sets[-1]['train'])} train / {args.val_seqs} val", flush=True)

    draft = load_draft(args.draft, args.device)
    draft.train()
    params = [p for p in draft.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    n_train = sum(len(s["train"]) for s in sets)
    total = int(n_train * args.epochs / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    bs = draft.block_size

    # chunks: each target's train set split into `rounds * epochs` pieces, interleaved at random
    n_chunks = max(1, int(round(args.rounds * args.epochs)))
    plan = []
    for k, s in enumerate(sets):
        seqs = list(s["train"]) * math.ceil(args.epochs)
        seqs = seqs[: int(len(s["train"]) * args.epochs)]
        size = math.ceil(len(seqs) / n_chunks)
        plan += [(k, seqs[i : i + size]) for i in range(0, len(seqs), size)]
    rng.shuffle(plan)

    target, loaded = None, None
    step, t0 = 0, time.time()

    def use(k):
        nonlocal target, loaded
        if loaded == k:
            return
        target = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        target = load_target(sets[k]["target"], args.device)
        for p in target.parameters():
            p.requires_grad_(False)
        loaded = k

    def feats_of(ids):
        with torch.no_grad():
            out = target(ids[None], output_hidden_states=True, logits_to_keep=1)
            return extract_context_feature(out.hidden_states, draft.target_layer_ids)

    for ci, (k, chunk) in enumerate(plan):
        use(k)
        print(f"[chunk {ci + 1}/{len(plan)}] target {sets[k]['target']}, {len(chunk)} sequences", flush=True)
        for i in range(0, len(chunk) - args.batch + 1, args.batch):
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for d in chunk[i : i + args.batch]:
                ids = d["ids"].long().to(args.device)
                blocks = sample_blocks(d["n_prompt"], len(ids), args.blocks, bs, 0, 1.0, rng)
                if not blocks:
                    continue
                loss, _ = block_loss(draft, target, feats_of(ids), ids, blocks, bs, args.gamma)
                (loss / args.batch).backward()
                tot += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[step {step}/{total}] loss {tot / args.batch:.3f} {(time.time() - t0) / step:.2f}s/step", flush=True)

    draft.eval()
    vr = random.Random(123)
    with torch.no_grad():
        for k, s in enumerate(sets):
            use(k)
            accs = []
            for d in s["val"]:
                ids = d["ids"].long().to(args.device)
                blocks = [(b, 0) for b, _ in sample_blocks(d["n_prompt"], len(ids), 8, bs, 0, 1.0, vr)]
                if blocks:
                    accs.append(float(block_loss(draft, target, feats_of(ids), ids, blocks, bs, args.gamma)[1]))
            print(f"[final] val acc1 on {s['target']}: {sum(accs) / max(len(accs), 1):.3f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(out)
    print(f"saved joint drafter to {out}")


if __name__ == "__main__":
    main()
