"""Pilot: can a drafter built on a pretrained small LM beat the DFlash architecture at equal budget?

Same data (target-greedy sequences), same steps, same blocks, same validation for every arm:
  dflash_scratch   the DFlash drafter architecture (5 layers, target width), weights re-initialised,
                   trained with DFlash's own block loss (the fair small-budget baseline)
  pdraft           PDraft (hspec/pdraft.py): Qwen3-0.6B (or its first n layers) + target features
  pdraft_nofeat    PDraft without target features (does the target context matter?)
Validation every --eval-every steps on the 48 held-out sequences (8 anchors each):
  tau     accepted tokens per block under greedy verification (leading correct + 1)
  acc@k   top-1 accuracy at block offset k
and, at step 0, the official fully trained DFlash drafter on the same anchors (reference line).
Checkpoint curves (tau vs training tokens) are the deliverable.

  python scripts/train_pilot.py --arch pdraft --data .../hspec_lagft/data --out .../pilot_pdraft
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

from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.shallow import drafter_block_loss  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.pdraft import PDraft, block_ce  # noqa: E402


def lead(hits):
    n = 0
    while n < len(hits) and hits[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["dflash_scratch", "pdraft", "pdraft_nofeat"])
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--small", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-layers", type=int, default=0, help="PDraft: first n layers of the small LM (0 = all)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=16, help="anchors per sequence")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = arch default (scratch 3e-4, pretrained 5e-5)")
    ap.add_argument("--lr-new", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    tag = args.arch + (f"_L{args.n_layers}" if args.n_layers and args.arch != "dflash_scratch" else "")
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
    official = load_draft(args.draft, dev)
    official.requires_grad_(False)
    lids = list(official.target_layer_ids)
    bs = official.block_size
    W = len(lids) * target.config.hidden_size

    if args.arch == "dflash_scratch":
        model = load_draft(args.draft, dev)
        model.apply(model._init_weights)                 # same architecture, fresh weights
        model.float().train()
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr or 3e-4}]
    else:
        small = load_target(args.small, dev)
        model = PDraft(small, W, args.n_layers or None, use_feats=args.arch == "pdraft").to(dev).float()
        del small
        model.train()
        groups = [{"params": model.pretrained_params(), "lr": args.lr or 5e-5},
                  {"params": model.new_params(), "lr": args.lr_new}]
    n_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"[{tag}] trainable {n_params / 1e6:.0f} M params", flush=True)
    if dev == "cpu":
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=0.0, betas=(0.9, 0.95))
    total = int(len(train) * args.epochs / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    @torch.no_grad()
    def feats_of(ids):
        out = target(ids[None], output_hidden_states=True, logits_to_keep=1)
        return extract_context_feature(out.hidden_states, lids)

    def anchors(d, n, r):
        T = len(d["ids"])
        lo, hi = max(d["n_prompt"], 1), T - 2
        return r.sample(range(lo, hi), min(n, hi - lo)) if hi > lo else []

    def loss_and_hits(ids, feats, anc):
        if args.arch == "dflash_scratch":
            loss, _ = drafter_block_loss(model, target, feats, feats, ids, [(s, 0) for s in anc], bs, args.gamma)
            return loss, None
        logits = model(ids, feats[0], anc, bs)
        loss, hit, valid = block_ce(logits, ids, anc, args.gamma)
        return loss, (hit, valid)

    history = []
    seen_tokens = [0]

    @torch.no_grad()
    def evaluate(step, reference=False):
        model.eval()
        vr = random.Random(123)
        taus, ref = [], []
        acc = torch.zeros(bs - 1)
        cnt = torch.zeros(bs - 1)
        for d in val:
            ids = d["ids"].long().to(dev)
            anc = anchors(d, 8, vr)
            if not anc:
                continue
            feats = feats_of(ids)
            for s in anc:
                truth = ids[s + 1 : s + bs]
                if args.arch == "dflash_scratch":
                    _, corr = drafter_block_loss(model, target, feats, feats, ids, [(s, 0)], bs, args.gamma)
                    pred = torch.tensor([truth[k] if c else -1 for k, c in enumerate(corr[0]) if c is not None],
                                        device=dev)
                else:
                    pred = model(ids, feats[0], [s], bs)[0].argmax(-1)
                hits = (pred[: len(truth)] == truth).cpu()
                taus.append(1 + lead(hits.tolist()))
                acc[: len(hits)] += hits.float()
                cnt[: len(hits)] += 1
                if reference:
                    rp = draft_logits_lagged(official, target, feats, ids, s, 0, bs).argmax(-1)
                    ref.append(1 + lead((rp[: len(truth)] == truth).tolist()))
        model.train()
        a = (acc / cnt.clamp_min(1)).tolist()
        rec = {"step": step, "tokens_seen": seen_tokens[0], "tau": sum(taus) / len(taus), "acc": a}
        if reference:
            rec["tau_official_dflash"] = sum(ref) / len(ref)
        history.append(rec)
        print(f"[{tag} step {step}] tokens {seen_tokens[0]} tau {rec['tau']:.3f} acc@1 {a[0]:.3f} acc@4 {a[3]:.3f} "
              f"acc@8 {a[7]:.3f} acc@15 {a[-1]:.3f}"
              + (f" | official DFlash tau {rec['tau_official_dflash']:.3f}" if reference else ""), flush=True)

    evaluate(0, reference=True)
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
                ids = d["ids"].long().to(dev)
                anc = anchors(d, args.blocks, rng)
                if not anc:
                    continue
                loss, _ = loss_and_hits(ids, feats_of(ids), anc)
                (loss / args.batch).backward()
                tot += float(loss)
                n += 1
                seen_tokens[0] += len(anc) * (bs - 1)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[{tag} step {step}/{total}] loss {tot / max(n, 1):.3f} {(time.time() - t0) / step:.2f}s/step",
                      flush=True)
            if step % args.eval_every == 0:
                evaluate(step)
    evaluate(step)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    torch.save({"args": vars(args), "state": model.state_dict()}, out / "model.pt")
    print(f"[{tag}] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
