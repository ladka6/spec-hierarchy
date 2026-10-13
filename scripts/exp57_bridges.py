"""Experiment 57: a copy made of the target's own tap layers plus small bridges for the removed spans.

The taps8 copy (exp55) keeps the layers the DFlash drafter reads (1, 9, 17, 25, 33) plus 0, 34, 35 and drops the
spans 2-8, 10-16, 18-24, 26-32 (identity). Trained with LoRA it reached tau 9.20 (exp27 conf2) against 11.90 for the
4-bit copy. Here each removed span is replaced by a bridge (hspec/bridges.py) that predicts what the span added to
the residual stream; the kept layers stay the target's frozen layers, so the copy needs no weights of its own beyond
the bridges.

Steps (one run = one bridge kind):
  fit    closed-form affine part of every bridge (ridge regression, Ghosted Layers / SHIFT-LLM style) on the teacher
         sequences, from the target's activations
  train  (kind mlp / attn, --epochs > 0) bridges trained end to end, kept layers frozen, with
           hidden   relative MSE to the target's hidden states at every span end and every tap feature
           drafter  cross-entropy of the drafter's block predictions on the copy's features against its top-32
                    predictions on real features (exp55 teacher.pt), position weights gamma^(k-1)
           agree    cross-entropy of the copy's next-token logits on the target's greedy text
The teacher and the student are the same model: hspec.bridges.BRIDGE["on"] switches the bridges off (full target).
Validation as exp55 (drafter argmax agreement, copy-target agreement). Loads in exp27 as "bridge:<out dir>".

  python scripts/exp57_bridges.py --kind mlp --teacher .../teacher.pt --out .../bridge_mlp
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp55_copytrain import block_logits  # noqa: E402

from hspec.bridges import BRIDGE, attach_bridges, fit_linear_, save_bridges, spans_of  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402


def rel_mse(s, t):
    """Mean over tokens of ||s - t||^2 / ||t||^2 (per-token, so sink tokens with huge norms do not dominate)."""
    s, t = s.float(), t.float()
    return ((s - t).pow(2).sum(-1) / t.pow(2).sum(-1).clamp_min(1e-6)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--teacher", required=True, help="exp55 teacher phase output")
    ap.add_argument("--kept", default="0-1-9-17-25-33-34-35")
    ap.add_argument("--kind", choices=["lin", "mlp", "attn"], default="mlp")
    ap.add_argument("--rank", type=int, default=2048)
    ap.add_argument("--fit-seqs", type=int, default=256, help="sequences for the closed-form affine fit (0: no fit)")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--fit-mode", choices=["indep", "seq"], default="indep",
                    help="seq: each span fitted on the copy's own inputs (earlier bridges on), see fit_linear_")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-lin", type=float, default=3e-5, help="the fitted affine parts move slower")
    ap.add_argument("--w-hid", type=float, default=1.0)
    ap.add_argument("--w-agree", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--n-train", type=int, default=0, help="0: all teacher sequences")
    ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tdata = torch.load(args.teacher)
    train, val = tdata["train"], tdata["val"]
    if args.n_train:
        train = train[: args.n_train]
    model = load_target(args.target, dev)
    model.requires_grad_(False)
    draft = load_draft(args.draft, dev)
    draft.requires_grad_(False)
    bs, lids = draft.block_size, list(draft.target_layer_ids)
    kept = [int(x) for x in args.kept.split("-")]
    n_layers = len(model.model.layers)
    spans = spans_of(kept, n_layers)
    bridges = attach_bridges(model, kept, args.kind, args.rank)
    n_par = sum(p.numel() for p in bridges.parameters())
    print(f"[bridges] kept {kept}, spans {spans}, kind {args.kind}, {n_par / 1e6:.1f}M params, "
          f"{len(train)} train / {len(val)} val sequences", flush=True)

    if args.fit_seqs:
        t0 = time.time()
        fit_linear_(model, bridges, [d["ids"].long() for d in train[: args.fit_seqs]], args.ridge, dev,
                    sequential=args.fit_mode == "seq")
        print(f"[fit] closed-form affine parts ({args.fit_mode}) from {args.fit_seqs} sequences in {time.time() - t0:.0f}s", flush=True)

    # hidden-state indices to match: span ends (input of the next kept layer) and the drafter's tap features
    hid_idx = sorted({b + 1 for _, b in spans if b + 1 < n_layers} | {l + 1 for l in lids})
    w = args.gamma ** torch.arange(bs - 1, device=dev, dtype=torch.float32)

    def run(d, grad):
        ids = d["ids"].long().to(dev)
        with torch.no_grad():
            BRIDGE["on"] = False
            t_hs = model(ids[None], output_hidden_states=True, logits_to_keep=1).hidden_states
            t_hs = [t_hs[i] for i in hid_idx]
            BRIDGE["on"] = True
        with torch.set_grad_enabled(grad):
            out = model(ids[None], output_hidden_states=True, use_cache=False)
            l_hid = sum(rel_mse(out.hidden_states[i], t) for i, t in zip(hid_idx, t_hs)) / len(hid_idx)
            feats = extract_context_feature(out.hidden_states, lids)
            lp = torch.log_softmax(block_logits(draft, model, feats, ids, d["anchors"], bs).float(), -1)
            t_lp, t_idx = d["t_lp"].to(dev).float(), d["t_idx"].to(dev).long()
            ce_d = -(torch.softmax(t_lp, -1) * lp.gather(-1, t_idx)).sum(-1)
            l_draft = (ce_d * w).sum() / (w.sum() * ce_d.shape[0])
            r = slice(d["n_prompt"] - 1, len(ids) - 1)
            logits = out.logits[0, r].float()
            tgt = ids[r.start + 1 : r.stop + 1]
            l_agree = F.cross_entropy(logits, tgt)
            stats = {"draft_agree": float((lp.argmax(-1) == t_idx[..., 0]).float().mean()),
                     "copy_agree": float((logits.argmax(-1) == tgt).float().mean())}
        return l_hid, l_draft, l_agree, stats

    def evaluate(step):
        bridges.eval()
        tot = {"draft_agree": 0.0, "copy_agree": 0.0, "l_draft": 0.0, "l_hid": 0.0}
        for d in val:
            l_h, l_d, _, st = run(d, False)
            tot["draft_agree"] += st["draft_agree"]
            tot["copy_agree"] += st["copy_agree"]
            tot["l_draft"] += float(l_d)
            tot["l_hid"] += float(l_h)
        bridges.train()
        n = len(val)
        print(f"[eval step {step}] drafter argmax agreement (copy vs real features) {tot['draft_agree'] / n:.4f}  "
              f"copy-target next-token agreement {tot['copy_agree'] / n:.4f}  drafter loss {tot['l_draft'] / n:.4f}  "
              f"hidden rel-mse {tot['l_hid'] / n:.4f}", flush=True)

    meta = {"target": args.target, "kept": kept, "kind": args.kind, "rank": args.rank, "args": vars(args)}
    evaluate(0)
    if args.kind == "lin" or args.epochs == 0:
        if args.out:
            save_bridges(args.out, bridges, meta)
            print(f"[bridges] saved {args.out}", flush=True)
        return

    lin_p = [p for m in bridges.values() for p in m.lin.parameters()]
    rest = [p for n_, p in bridges.named_parameters() if ".lin." not in n_]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr}, {"params": lin_p, "lr": args.lr_lin}], weight_decay=0.0)
    total = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    params = list(bridges.parameters())
    bridges.train()
    t0, step = time.time(), 0
    rng = random.Random(1)
    for _ in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for j in order:
            step += 1
            l_h, l_d, l_a, _ = run(train[j], True)
            loss = args.w_hid * l_h + l_d + args.w_agree * l_a
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 50 == 0:
                print(f"[train {step}/{total}] hidden {float(l_h):.4f} drafter {float(l_d):.3f} "
                      f"agree {float(l_a):.3f} {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 and step < total:
                evaluate(step)
    evaluate(total)
    if args.out:
        save_bridges(args.out, bridges, meta)
        print(f"[bridges] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
