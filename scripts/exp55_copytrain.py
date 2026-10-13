"""Experiment 55: train a cheap copy of the target for its role in speculation.

The feature-refresh method (hspec/refresh.py, exp27) needs a cheap copy of the target that (a) gives the DFlash
drafter stand-in features for tokens the target has not processed yet and (b) predicts the target's next token.
A 4-bit copy works (exp27: 12.53 accepted vs 12.54 with real features) but costs too much; cheaper generic copies
fail (2-bit round-to-nearest: 8.44; 18 of 36 layers: gain gone). They were never trained for the job.

Here the copy gets LoRA updates on its decoder linears, trained with
  drafter loss  cross-entropy of the drafter's block predictions conditioned on the COPY's features against the
                drafter's predictions conditioned on the TARGET's features (top-32 soft targets), position weights
                gamma^(k-1)
  agreement     cross-entropy of the copy's next-token logits on the target's greedy text (the data IS the
                target's greedy output)
Phases (the target and the copy never need to be in memory together):
  teacher  target + drafter: anchors per sequence, the drafter's top-32 predictions with real features
  train    copy (+ LoRA) + drafter (frozen): optimise; validation = drafter argmax agreement (copy features vs
           real features) and copy-target next-token agreement on held-out sequences
The trained copy loads as "lora:<out dir>" (hspec.models.load_mid) for exp27.

  python scripts/exp55_copytrain.py --phase teacher --root .../hspec_domains --teacher .../teacher.pt
  python scripts/exp55_copytrain.py --phase train --base rtn2:Qwen/Qwen3-8B --teacher .../teacher.pt --out .../copy_rtn2
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

from dflash.model import _output_head, extract_context_feature  # noqa: E402

from hspec.copylora import add_lora, save_lora  # noqa: E402
from hspec.lagtrain import pack  # noqa: E402
from hspec.models import load_draft, load_mid, load_target  # noqa: E402


def block_logits(draft, head_src, feats, ids, anchors, bs):
    noise, pos, mask, starts = pack(draft, head_src, feats, ids, [(s, 0) for s in anchors], bs)
    hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                past_key_values=None, use_cache=False)[0]
    return draft.compute_logits(torch.stack([hid[st : st + bs - 1] for st in starts]), _output_head(head_src))


def load_seqs(root, domains, split, max_len, n, seed):
    seqs = []
    for dm in domains:
        seqs += [d for d in torch.load(Path(root) / dm / f"{split}.pt") if len(d["ids"]) <= max_len]
    random.Random(seed).shuffle(seqs)
    return seqs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["teacher", "train"])
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--root", default="", help="gen_domaindata.py output (teacher phase)")
    ap.add_argument("--domains", nargs="+", default=["math", "code", "chat", "think"])
    ap.add_argument("--n-train", type=int, default=2400)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--base", default="rtn2:Qwen/Qwen3-8B", help="copy spec for load_mid (train phase)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-agree", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    draft = load_draft(args.draft, dev)
    draft.requires_grad_(False)
    bs, lids = draft.block_size, list(draft.target_layer_ids)

    if args.phase == "teacher":
        target = load_target(args.target, dev)
        rng = random.Random(0)
        out = {}
        for split, n in (("train", args.n_train), ("val", args.n_val)):
            recs = []
            for d in load_seqs(args.root, args.domains, split, args.max_len, n, 0):
                ids = d["ids"].long().to(dev)
                lo, hi = max(d["n_prompt"], 1), len(ids) - bs
                if hi <= lo:
                    continue
                anc = sorted(rng.sample(range(lo, hi), min(args.anchors, hi - lo)))
                with torch.inference_mode():
                    hs = target(ids[None], output_hidden_states=True, logits_to_keep=1).hidden_states
                    feats = extract_context_feature(hs, lids)
                    lp = torch.log_softmax(block_logits(draft, target, feats, ids, anc, bs).float(), -1)
                    v, i = lp.topk(args.topk, -1)
                recs.append({"ids": d["ids"], "n_prompt": d["n_prompt"], "anchors": anc,
                             "t_lp": v.half().cpu(), "t_idx": i.int().cpu()})
            out[split] = recs
            print(f"[teacher] {split}: {len(recs)} sequences", flush=True)
        torch.save(out, args.teacher)
        print(f"[teacher] saved {args.teacher}", flush=True)
        return

    tdata = torch.load(args.teacher)
    train, val = tdata["train"], tdata["val"]
    copy = load_mid(args.base, dev)
    copy.requires_grad_(False)
    mods = add_lora(copy, args.rank, args.alpha)
    params = [p for m in mods.values() for p in (m.A, m.B)]
    print(f"[train] base {args.base}: LoRA on {len(mods)} linears ({sum(p.numel() for p in params) / 1e6:.1f}M params), "
          f"{len(train)} train / {len(val)} val sequences", flush=True)
    if dev == "cuda":
        copy.gradient_checkpointing_enable()
        copy.config.use_cache = False
        copy.enable_input_require_grads()
    w = args.gamma ** torch.arange(bs - 1, device=dev, dtype=torch.float32)

    def run(d, grad):
        ids = d["ids"].long().to(dev)
        with torch.set_grad_enabled(grad):
            out = copy(ids[None], output_hidden_states=True, use_cache=False)
            feats = extract_context_feature(out.hidden_states, lids)
            lp = torch.log_softmax(block_logits(draft, copy, feats, ids, d["anchors"], bs).float(), -1)
            t_lp, t_idx = d["t_lp"].to(dev).float(), d["t_idx"].to(dev).long()
            p = torch.softmax(t_lp, -1)                                      # renormalised top-k teacher
            ce_d = -(p * lp.gather(-1, t_idx)).sum(-1)                       # [A, bs-1]
            l_draft = (ce_d * w).sum() / (w.sum() * ce_d.shape[0])
            r = slice(d["n_prompt"] - 1, len(ids) - 1)
            logits = out.logits[0, r].float()
            l_agree = F.cross_entropy(logits, ids[r.start + 1 : r.stop + 1])
            stats = {"draft_agree": float((lp.argmax(-1) == t_idx[..., 0]).float().mean()),
                     "copy_agree": float((logits.argmax(-1) == ids[r.start + 1 : r.stop + 1]).float().mean())}
        return l_draft, l_agree, stats

    def evaluate(step):
        copy.eval()
        tot = {"draft_agree": 0.0, "copy_agree": 0.0, "l_draft": 0.0}
        for d in val:
            l_d, _, st = run(d, False)
            tot["draft_agree"] += st["draft_agree"]
            tot["copy_agree"] += st["copy_agree"]
            tot["l_draft"] += float(l_d)
        copy.train()
        n = len(val)
        print(f"[eval step {step}] drafter argmax agreement (copy vs real features) {tot['draft_agree'] / n:.4f}  "
              f"copy-target next-token agreement {tot['copy_agree'] / n:.4f}  drafter loss {tot['l_draft'] / n:.4f}",
              flush=True)

    evaluate(0)
    if args.epochs == 0:                                    # evaluation only (reference copies, e.g. bnb4)
        return
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    copy.train()
    t0, step = time.time(), 0
    rng = random.Random(1)
    for _ in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for j in order:
            step += 1
            l_d, l_a, _ = run(train[j], True)
            loss = l_d + args.w_agree * l_a
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 50 == 0:
                print(f"[train {step}/{total}] drafter {float(l_d):.3f} agree {float(l_a):.3f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 and step < total:
                evaluate(step)
    evaluate(total)
    if args.out:
        save_lora(args.out, mods, {"base": args.base, "rank": args.rank, "alpha": args.alpha, "args": vars(args)})
        print(f"[train] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
