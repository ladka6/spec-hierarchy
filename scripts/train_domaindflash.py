"""Does a DFlash drafter forget? Domain fine-tuning of the official drafter, tau measured on every domain.

Fine-tunes the official Qwen3-8B DFlash drafter (DFlash's own training setup: target features of layers
1/9/17/25/33 as K/V context, anchor + 15 masks, position-decayed CE on the target's greedy tokens) on the
target-generated data of one or more domains (scripts/gen_domaindata.py), optionally starting from an earlier
run's drafter (--init) for sequential A -> B adaptation. Every --eval-every steps: tau (leading correct + 1) and
acc@k on the held-out set of EACH domain, so forgetting of earlier domains is visible.

  python scripts/train_domaindflash.py --root /scratch-shared/$USER/hspec_domains --train code \
      --steps 1000 --save .../drafter_code.pt --out .../run_code
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

from dflash.model import _draft_value, _output_head, _raw_input_embeddings  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402


def lead(hits):
    n = 0
    while n < len(hits) and hits[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--root", required=True, help="gen_domaindata.py output (one directory per domain)")
    ap.add_argument("--train", nargs="*", default=[], help="domains to train on (mixed); none = evaluation only")
    ap.add_argument("--eval-domains", nargs="+", default=["math", "think", "code", "chat"])
    ap.add_argument("--init", default="", help="drafter state from an earlier run (--save)")
    ap.add_argument("--steps", type=int, default=1000, help="training sequences (one optimiser step each)")
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--val-anchors", type=int, default=8)
    ap.add_argument("--val-seqs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--max-len", type=int, default=1400)
    ap.add_argument("--save", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rng = random.Random(0)
    tag = args.tag or ("+".join(args.train) or "eval") + ("" if not args.init else f"<-{Path(args.init).stem}")

    root = Path(args.root)

    def ok(d):
        return len(d["ids"]) <= args.max_len and len(d["ids"]) - d["n_prompt"] > 18

    val = {dm: [d for d in torch.load(root / dm / "val.pt") if ok(d)][: args.val_seqs] for dm in args.eval_domains}
    train = []
    for dm in args.train:
        train += [d for d in torch.load(root / dm / "train.pt") if ok(d)]
    rng.shuffle(train)
    if train:
        train = (train * (1 + args.steps // max(1, len(train))))[: args.steps]
    print(f"[{tag}] train {len(train)} seqs from {args.train or '-'}; val " +
          ", ".join(f"{k} {len(v)}" for k, v in val.items()), flush=True)

    target = load_target(args.target, dev)
    target.requires_grad_(False)
    draft = load_draft(args.draft, dev).float()
    if args.init:
        draft.load_state_dict({k: v.float() for k, v in torch.load(args.init, map_location=dev).items()})
    bs = draft.block_size
    lids = list(draft.target_layer_ids)
    head = _output_head(target)
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))

    @torch.no_grad()
    def feats_of(ids):
        out = target.model(ids[None], output_hidden_states=True)
        return torch.cat([out.hidden_states[l + 1][0] for l in lids], -1)

    def forward(ids, feats, anchors):
        T = feats.shape[0]
        toks, bpos = [], []
        for s in anchors:
            t = torch.full((bs,), draft.mask_token_id, dtype=torch.long, device=dev)
            t[0] = ids[s]
            toks.append(t)
            bpos.append(torch.arange(s, s + bs, device=dev))
        Q = bs * len(anchors)
        mask = torch.zeros(Q, T + Q, dtype=torch.bool, device=dev)
        for b, s in enumerate(anchors):
            r = slice(b * bs, (b + 1) * bs)
            mask[r, :s] = True
            mask[r, T + b * bs : T + (b + 1) * bs] = True
        noise = _raw_input_embeddings(target, torch.cat(toks)[None], scale).float()
        pos = torch.cat([torch.arange(T, device=dev)] + bpos)[None]
        h = draft(target_hidden=feats[None].float(), noise_embedding=noise, position_ids=pos,
                  attention_mask=mask[None, None], past_key_values=None, use_cache=False)[0]
        h = h.view(len(anchors), bs, -1)[:, 1:]
        return draft.compute_logits(h.to(head.weight.dtype), head).float()

    def anchors_of(d, n, r):
        lo, hi = max(d["n_prompt"], 1), len(d["ids"]) - bs
        return r.sample(range(lo, hi), min(n, hi - lo)) if hi > lo else []

    history = []

    @torch.no_grad()
    def evaluate(step):
        draft.eval()
        rec = {"step": step}
        for dm, seqs in val.items():
            r = random.Random(123)
            taus, acc, cnt = [], torch.zeros(bs - 1), 0
            for d in seqs:
                ids = d["ids"].long().to(dev)
                anc = anchors_of(d, args.val_anchors, r)
                if not anc:
                    continue
                pred = forward(ids, feats_of(ids), anc).argmax(-1)
                for b, s in enumerate(anc):
                    hits = (pred[b] == ids[s + 1 : s + bs]).cpu()
                    taus.append(1 + lead(hits.tolist()))
                    acc += hits.float()
                    cnt += 1
            rec[dm] = {"tau": sum(taus) / len(taus), "acc1": float(acc[0] / cnt), "acc8": float(acc[7] / cnt)}
        history.append(rec)
        draft.train()
        print(f"[{tag} step {step}] tau " + "  ".join(f"{dm} {rec[dm]['tau']:.3f}" for dm in val), flush=True)

    evaluate(0)
    if train:
        draft.train()
        if dev == "cpu":
            opt = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
        else:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(draft.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
        total = len(train)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda st: min(1.0, (st + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(st, total) / total)))
        w = args.gamma ** torch.arange(bs - 1, device=dev, dtype=torch.float32)
        t0 = time.time()
        for step, d in enumerate(train, 1):
            ids = d["ids"].long().to(dev)
            anc = anchors_of(d, args.anchors, rng)
            if not anc:
                continue
            logits = forward(ids, feats_of(ids), anc)
            lab = torch.stack([ids[s + 1 : s + bs] for s in anc])
            ce = F.cross_entropy(logits.flatten(0, 1), lab.flatten(), reduction="none").view(len(anc), -1)
            loss = (ce * w).sum() / (w.sum() * len(anc))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 100 == 0:
                print(f"[{tag} step {step}/{total}] loss {float(loss):.3f} {(time.time() - t0) / step:.2f}s/seq",
                      flush=True)
            if step % args.eval_every == 0 and step < total:
                evaluate(step)
        evaluate(total)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    if args.save and train:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().to(torch.bfloat16).cpu() for k, v in draft.state_dict().items()}, args.save)
        print(f"[{tag}] saved drafter to {args.save}", flush=True)
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    main()
