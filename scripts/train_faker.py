"""Train a faker (hspec/faker.py) for a frozen DFlash drafter and frozen target.

Per sequence: one target pass gives the real context vectors c; for sampled anchors s the faker
emulates c for s .. s+bs-2 from the real c before s and the tokens, and the frozen drafter's blocks
re-anchored at s+i must match their real-feature logits (KL) plus a cosine/MSE term on c_hat.

Validation reports per offset j = p - s the cosine of c_hat to c, and the drafter's top-1 agreement
(fake vs real features) over all re-anchored blocks.

  python scripts/train_faker.py --data .../hspec_lagft/data --out .../faker.pt
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

from hspec.faker import Faker, faker_loss, save_faker  # noqa: E402
from hspec.feathead import context_vectors  # noqa: E402
from hspec.lagtrain import sample_blocks  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--anchors", type=int, default=4, help="anchors per sequence per step")
    ap.add_argument("--offsets", type=int, default=6, help="re-anchored blocks per anchor in the KL term")
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4, help="sequences per optimizer step")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
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
    faker = Faker(H, args.dim, args.layers, args.heads, args.window, bs).to(args.device)
    n_par = sum(p.numel() for p in faker.parameters())
    print(f"faker: {n_par / 1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(faker.parameters(), lr=args.lr, weight_decay=0.0)
    total = max(1, int(len(train) * args.epochs / args.batch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def seq_ctx(d):
        ids = d["ids"].long().to(args.device)
        with torch.no_grad():
            feats = extract_context_feature(target(ids[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            c = context_vectors(draft, feats)[0].float()
        return ids, c

    def anchors_for(d, n, r):
        T = len(d["ids"])
        return [s for s, _ in sample_blocks(d["n_prompt"], T, n, bs, 0, 1.0, r) if s + 2 * bs <= T]

    def evaluate():
        faker.eval()
        vr = random.Random(123)
        cos_by = [[] for _ in range(bs - 1)]
        agree = []
        with torch.no_grad():
            for d in val:
                ids, c = seq_ctx(d)
                for s in anchors_for(d, 4, vr):
                    _, st = faker_loss(faker, draft, target, c, ids, s, bs, 0.0)
                    for j, v in enumerate(st["cos"].tolist()):
                        cos_by[j].append(v)
                    agree.append(st["agree"])
        faker.train()
        return [sum(v) / max(len(v), 1) for v in cos_by], sum(agree) / max(len(agree), 1)

    v0, a0 = evaluate()
    print(f"[step 0] val cos by offset: {' '.join(f'{x:.3f}' for x in v0)}  drafter top-1 agreement {a0:.3f}",
          flush=True)
    step, t0 = 0, time.time()
    faker.train()
    seqs = (list(train) * math.ceil(args.epochs))[: int(len(train) * args.epochs)]
    for i in range(0, len(seqs) - args.batch + 1, args.batch):
        opt.zero_grad(set_to_none=True)
        tot, kls, ags, n = 0.0, 0.0, 0.0, 0
        for d in seqs[i : i + args.batch]:
            ids, c = seq_ctx(d)
            anc = anchors_for(d, args.anchors, rng)
            for s in anc:
                offs = sorted(rng.sample(range(1, bs - 1), min(args.offsets, bs - 2)))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, st = faker_loss(faker, draft, target, c, ids, s, bs, args.kl_weight, offs)
                (loss / (args.batch * max(len(anc), 1))).backward()
                tot += float(loss.detach())
                kls += st["kl"]
                ags += st["agree"]
                n += 1
        torch.nn.utils.clip_grad_norm_(faker.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 25 == 0 and n:
            print(f"[step {step}/{total}] loss {tot / n:.4f} kl {kls / n:.4f} agree {ags / n:.3f} "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % 500 == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_faker(faker, args.out)
    v1, a1 = evaluate()
    print(f"[final] val cos by offset: {' '.join(f'{x:.3f}' for x in v1)}  mean {sum(v1) / len(v1):.3f}  "
          f"drafter top-1 agreement {a1:.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_faker(faker, args.out)
    print(f"saved faker to {args.out}")


if __name__ == "__main__":
    main()
