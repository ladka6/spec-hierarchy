"""Train a dependency head on a frozen DFlash drafter (hspec/dephead.py).

  --mode tf     teacher forcing: prior = true token, target = target distribution after the
                true prefix (TreeFlash-style training)
  --mode tree   on-policy: also every node of the DDTree the drafter proposes at the anchor,
                prior = the node's token, target = the target model's distribution after the
                node's path (one tree-masked target forward per block, reusing the prefix KV)

Loss: KL(target || head) per row, weighted gamma^depth. Only the head is trained.

  python scripts/train_dephead.py --mode tree --data /scratch-shared/$USER/hspec_lagft/data --out .../dephead_tree.pt
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

from dflash.model import _make_cache, _output_head, extract_context_feature  # noqa: E402

from hspec.dephead import DepHead, backbone_rows, head_logits, save_dephead  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.pipeline import crop  # noqa: E402
from hspec.tree import build_ddtree, tree_inputs, tree_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["tf", "tree"], required=True)
    ap.add_argument("--anchors", type=int, default=4, help="blocks per sequence")
    ap.add_argument("--budget", type=int, default=32, help="tree nodes per block (tree mode)")
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=32)
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
    print(f"{len(train)} train / {len(val)} val sequences, mode {args.mode}", flush=True)

    target = load_target(args.target, args.device)
    draft = load_draft(args.draft, args.device)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    bs = draft.block_size
    D = bs - 1
    dep = DepHead(target.config.hidden_size, args.inter).to(args.device, next(target.parameters()).dtype)
    params = list(dep.parameters())
    print(f"head parameters: {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = int(len(train) * args.epochs / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def rows_for(d, rrng, mode):
        """(hidden rows [N,H], prior tokens [N], target log-probs [N,V], weights [N]) for one sequence."""
        ids = d["ids"].long().to(args.device)
        T = len(ids)
        lo, hi = d["n_prompt"], T - bs - 1
        if hi <= lo:
            return None
        with torch.no_grad():
            cache = _make_cache(target.config)
            out = target(ids[None], output_hidden_states=True, past_key_values=cache, use_cache=True)
            feats = extract_context_feature(out.hidden_states, draft.target_layer_ids)
            full_lp = torch.log_softmax(out.logits[0].float(), -1)
            anchors = sorted(rrng.sample(range(lo, hi), min(args.anchors, hi - lo)), reverse=True)
            H = backbone_rows(draft, target, feats, ids, anchors, bs)          # [A, D, Hd]
            hs, pt, tl, w = [], [], [], []
            for a, s in enumerate(anchors):                                    # truth path rows
                hs.append(H[a])
                pt.append(ids[s : s + D])
                tl.append(full_lp[s : s + D])
                w.append(torch.tensor([args.gamma ** k for k in range(D)], device=args.device))
            if mode == "tree":
                for a, s in enumerate(anchors):                                # descending s: crop as we go
                    crop(cache, s)
                    lg = draft.compute_logits(H[a], _output_head(target)).float()
                    tree = build_ddtree(lg, args.budget)
                    tids, tpos = tree_inputs(ids[s], tree, s, args.device)
                    to = target(tids, position_ids=tpos, attention_mask=tree_mask(tree, s, args.device),
                                past_key_values=cache, use_cache=True)
                    crop(cache, s)
                    lp = torch.log_softmax(to.logits[0].float(), -1)            # row r: after node r's path
                    depths = [0] + list(tree.depths)
                    toks = [int(ids[s])] + list(tree.tokens)
                    keep = [r for r in range(len(toks)) if depths[r] < D]
                    hs.append(H[a][[depths[r] for r in keep]])
                    pt.append(torch.tensor([toks[r] for r in keep], device=args.device))
                    tl.append(lp[keep])
                    w.append(torch.tensor([args.gamma ** depths[r] for r in keep], device=args.device))
        return torch.cat(hs), torch.cat(pt), torch.cat(tl), torch.cat(w)

    def loss_on(batch):
        h, t, tl, w = batch
        ql = torch.log_softmax(head_logits(draft, target, dep, h, t).float(), -1)
        kl = (tl.exp() * (tl - ql)).sum(-1)
        acc = (ql.argmax(-1) == tl.argmax(-1)).float()
        return (kl * w).sum() / w.sum(), acc.mean()

    def evaluate():
        dep.eval()
        vr = random.Random(123)
        res = {}
        with torch.no_grad():
            for mode in ("tf", "tree"):
                ls, accs = [], []
                for d in val:
                    b = rows_for(d, vr, mode)
                    if b is None:
                        continue
                    l, a = loss_on(b)
                    ls.append(float(l))
                    accs.append(float(a))
                res[mode] = (sum(ls) / len(ls), sum(accs) / len(accs))
        dep.train()
        return " ".join(f"{m}-rows: kl {v[0]:.3f} top1 {v[1]:.3f}" for m, v in res.items())

    print(f"[step 0] val {evaluate()}", flush=True)
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
                b = rows_for(train[j], rng, args.mode)
                if b is None:
                    continue
                l, _ = loss_on(b)
                (l / args.batch).backward()
                tot += float(l)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[step {step}/{total}] kl {tot / args.batch:.3f} {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0:
                print(f"[step {step}] val {evaluate()}", flush=True)
    print(f"[final] val {evaluate()}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_dephead(dep, args.out)
    print(f"saved head to {args.out}")


if __name__ == "__main__":
    main()
