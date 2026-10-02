"""Train a feature head (hspec/feathead.py) for a frozen DFlash drafter.

For sampled anchors s of target-generated sequences, a fresh drafter block gives hidden rows for
positions s+1 .. s+bs-1; the head maps (row p+1, token x_p) to the context vector c_p the drafter
would get from the target. Loss: (1 - cos) + relative MSE. Validation reports the cosine per
offset j = p - s (how far into the unverified region).

  python scripts/train_feathead.py --data .../hspec_lagft/data --out .../feathead.pt
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

from hspec.feathead import (FeatHead, block_hidden, context_vectors, feat_loss, predict_context,  # noqa: E402
                            save_head)
from hspec.lagtrain import sample_blocks  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
    data = [d for d in data if len(d["ids"]) <= args.max_len]
    rng.shuffle(data)
    val, train = data[: args.val_seqs], data[args.val_seqs :]
    if args.max_train:
        train = train[: args.max_train]
    print(f"{len(train)} train / {len(val)} val sequences", flush=True)

    target = load_target(args.target, args.device)
    draft = load_draft(args.draft, args.device)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    bs = draft.block_size
    H = draft.config.hidden_size
    head = FeatHead(H, args.mult).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    total = max(1, int(len(train) * args.epochs / args.batch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def pairs(d, n_blocks, r):
        ids = d["ids"].long().to(args.device)
        with torch.no_grad():
            feats = extract_context_feature(target(ids[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            c = context_vectors(draft, feats)[0]
            anchors = [s for s, _ in sample_blocks(d["n_prompt"], len(ids), n_blocks, bs, 0, 1.0, r)
                       if s + bs - 1 < len(ids)]
            if not anchors:
                return None
            hid = block_hidden(draft, target, feats, ids, anchors, bs)
        preds, reals, offs = [], [], []
        for a, s in enumerate(anchors):
            preds.append(predict_context(head, draft, target, hid[a].float(), ids, s, bs))
            reals.append(c[s : s + bs - 1])
            offs.append(torch.arange(bs - 1, device=args.device))
        return torch.cat(preds), torch.cat(reals), torch.cat(offs)

    def evaluate():
        head.eval()
        vr = random.Random(123)
        by = [[] for _ in range(bs - 1)]
        with torch.no_grad():
            for d in val:
                out = pairs(d, 8, vr)
                if out is None:
                    continue
                pr, re, of = out
                cos = torch.nn.functional.cosine_similarity(pr.float(), re.float(), dim=-1)
                for j in range(bs - 1):
                    by[j] += cos[of == j].tolist()
        head.train()
        return [sum(v) / max(len(v), 1) for v in by]

    head = head.float()
    v0 = evaluate()
    print(f"[step 0] val cos by offset: {' '.join(f'{x:.3f}' for x in v0)}", flush=True)
    step, t0 = 0, time.time()
    head.train()
    seqs = list(train) * math.ceil(args.epochs)
    seqs = seqs[: int(len(train) * args.epochs)]
    for i in range(0, len(seqs) - args.batch + 1, args.batch):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for d in seqs[i : i + args.batch]:
            out = pairs(d, args.blocks, rng)
            if out is None:
                continue
            loss, _ = feat_loss(out[0], out[1])
            (loss / args.batch).backward()
            tot += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            print(f"[step {step}/{total}] loss {tot / args.batch:.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
    v1 = evaluate()
    print(f"[final] val cos by offset: {' '.join(f'{x:.3f}' for x in v1)}  mean {sum(v1) / len(v1):.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_head(head, args.out, H, args.mult)
    print(f"saved feature head to {args.out}")


if __name__ == "__main__":
    main()
