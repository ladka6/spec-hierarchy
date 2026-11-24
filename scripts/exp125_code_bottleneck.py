"""Experiment 125: how much must a speculative feature carry once the drafter is trained to read it?

exp84 gave the frozen drafter the real deep features of the in-flight positions restricted to a rank-r linear
subspace: 256 of 4096 dimensions gave 0.68 of the ceiling, 1024 gave 0.92. That is the "imitate the target's state"
route. Here everything is trained for the drafter's own loss: the in-flight positions get a learned r-dimensional
CODE of their real features (encoder: 5 taps -> r, decoder: r -> the drafter's fused feature space), and the drafter
is fine-tuned together with the encoder and decoder on the block loss only. No term asks the code to resemble the
target's state. This is the upper bound of a speculative-feature model that would have to PREDICT r numbers per
in-flight token: if a small r is enough, the thing to predict is small.

Dual mode. A block at anchor s with lag g gets
  context   real fused features of positions < s - g, and the decoded code of positions s - g .. s - 1
  queries   the in-flight tokens x_{s-g} .. x_s and the MASKs, as in lag training (the token path stays)
Arms (--code):
  0      no code rows: plain lag training (the baseline, same data and steps)
  r > 0  code of r dimensions
  real   the real fused features at the in-flight positions (the ceiling of this layout)
Start: the lag-trained drafter. The decoder's last layer starts at zero.

Reported on held-out sequences, per lag (4, 8, 16) and for standard blocks (lag 0): loss, first drafted token
right, tau = accepted drafted tokens + 1; at the start and at the end of training.

  python scripts/exp125_code_bottleneck.py --data /scratch-shared/$USER/hspec_lagft/data \
      --draft ~/hspec_ckpt/drafter_lag16_fp32 --code 64 --out /scratch-shared/$USER/hspec_code/r64
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import _output_head, extract_context_feature  # noqa: E402

from hspec.lagtrain import pack, sample_blocks  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


class Code(nn.Module):
    """5 taps of one position -> r numbers -> a vector in the drafter's fused feature space"""

    def __init__(self, width, r, out, hid=1024):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.enc = nn.Sequential(nn.Linear(width, hid), nn.GELU(), nn.Linear(hid, r))
        self.dec = nn.Sequential(nn.Linear(r, hid), nn.GELU(), nn.Linear(hid, out))
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, feats):
        return self.dec(self.enc(self.norm(feats.float())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", required=True, help="the lag-trained drafter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--code", default="64", help="0 (lag baseline), r > 0, or real")
    ap.add_argument("--max-lag", type=int, default=16)
    ap.add_argument("--p-zero", type=float, default=0.25)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-code", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=48, help="as train_lag.py: the same held-out sequences")
    ap.add_argument("--val-blocks", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="exp125_code")
    args = ap.parse_args()
    dev, t0 = "cuda", time.time()
    mode = "real" if args.code == "real" else ("none" if int(args.code) == 0 else "code")

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
    print(f"{len(train)} train / {len(val)} val sequences, arm: {args.code}", flush=True)

    target = load_target(args.target, dev)
    for p in target.parameters():
        p.requires_grad_(False)
    draft = load_draft(args.draft, dev).float()
    draft.train()
    lids, bs = list(draft.target_layer_ids), draft.block_size
    fc = draft.fc                                            # trained with the drafter, applied here
    draft.fc = nn.Identity()                                 # the drafter is fed fused features
    width, Hd = fc.in_features, fc.out_features
    code = Code(width, int(args.code), Hd).to(dev) if mode == "code" else None
    dparams = [p for p in list(draft.parameters()) + list(fc.parameters()) if p.requires_grad]
    groups = [{"params": dparams, "lr": args.lr}]
    if code is not None:
        groups.append({"params": list(code.parameters()), "lr": args.lr_code})
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    total = max(1, int(len(train) * args.epochs / args.batch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    print(f"trainable: drafter {sum(p.numel() for p in dparams) / 1e6:.0f} M"
          + (f", code {sum(p.numel() for p in code.parameters()) / 1e6:.1f} M" if code is not None else "")
          + f", {total} steps, {time.time() - t0:.0f}s", flush=True)

    def feats_of(ids):
        with torch.no_grad():
            out = target(ids[None], output_hidden_states=True, logits_to_keep=1)
            return extract_context_feature(out.hidden_states, lids)

    def forward(feats, ids, blocks):
        """hidden states of the packed blocks; the in-flight positions of each block read the arm's features"""
        T = ids.shape[0]
        real = fc(feats.float())                             # [1, T, Hd]
        noise, pos, mask, starts = pack(draft, target, real, ids, blocks, bs)
        if mode == "none":
            ctx = real
        else:
            extra = real if mode == "real" else code(feats)
            ctx = torch.cat([real, extra.to(real.dtype)], 1)  # [1, 2T, Hd]: real rows, then the in-flight rows
            q = noise.shape[1]
            m = torch.zeros((q, 2 * T + q), dtype=torch.bool, device=ids.device)
            m[:, :T] = mask[0, 0, :, :T]
            m[:, 2 * T :] = mask[0, 0, :, T:]
            a = 0
            for blk in blocks:
                s, g = blk[0], blk[1]
                m[a : a + g + bs, T + s - g : T + s] = True
                a += g + bs
            mask = m[None, None]
            pos = torch.cat([torch.arange(T, device=ids.device)[None], pos], 1)
        hid = draft(target_hidden=ctx, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                    past_key_values=None, use_cache=False)[0]
        return hid, starts

    def loss_fn(feats, ids, blocks, stats=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid, starts = forward(feats, ids, blocks)
            T = ids.shape[0]
            rows, labels, weights, first = [], [], [], []
            for blk, st in zip(blocks, starts):
                s = blk[0]
                for k in range(1, bs):
                    if s + k >= T:
                        break
                    rows.append(st + k - 1)
                    labels.append(int(ids[s + k]))
                    weights.append(args.gamma ** (k - 1))
                    if k == 1:
                        first.append(len(rows) - 1)
            rows_t = torch.tensor(rows, device=ids.device)
            labels_t = torch.tensor(labels, device=ids.device)
            w = torch.tensor(weights, device=ids.device, dtype=torch.float32)
            logits = draft.compute_logits(hid[rows_t], _output_head(target)).float()
        ce = F.cross_entropy(logits, labels_t, reduction="none")
        loss = (ce * w).sum() / w.sum()
        ok = logits.argmax(-1) == labels_t
        acc1 = ok[first].float().mean()
        if not stats:
            return loss, acc1
        taus, i = [], 0                                      # accepted drafted tokens + 1 per block
        for blk in blocks:
            n = min(bs - 1, T - 1 - blk[0])
            run = ok[i : i + n].tolist()
            taus.append(1 + (run.index(False) if False in run else n))
            i += n
        return loss, acc1, taus

    def evaluate():
        draft.eval()
        vrng = random.Random(123)
        rows = []
        with torch.no_grad():
            for lag in (0, 4, 8, 16):
                ls, a1, taus = [], [], []
                for d in val:
                    ids = d["ids"].long().to(dev)
                    blocks = [(s, min(lag, s - 1)) for s, _ in
                              sample_blocks(d["n_prompt"], len(ids), args.val_blocks, bs, 0, 1.0, vrng)]
                    if not blocks:
                        continue
                    l, a, t = loss_fn(feats_of(ids), ids, blocks, stats=True)
                    ls.append(float(l))
                    a1.append(float(a))
                    taus += t
                rows.append({"lag": lag, "loss": sum(ls) / len(ls), "first token": sum(a1) / len(a1),
                             "tau": sum(taus) / len(taus), "blocks": len(taus)})
        draft.train()
        return rows

    def show(tag, rows):
        print(f"[{tag}] " + "  ".join(f"lag{r['lag']}: loss {r['loss']:.3f} first {r['first token']:.3f} "
                                       f"tau {r['tau']:.3f}" for r in rows), flush=True)

    start = evaluate()
    show("step 0", start)
    step, order = 0, list(range(len(train)))
    while step < total:
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if step >= total:
                break
            opt.zero_grad(set_to_none=True)
            tot_l, tot_a, n = 0.0, 0.0, 0
            for j in order[i : i + args.batch]:
                d = train[j]
                ids = d["ids"].long().to(dev)
                blocks = sample_blocks(d["n_prompt"], len(ids), args.blocks, bs, args.max_lag, args.p_zero, rng)
                if not blocks:
                    continue
                loss, acc = loss_fn(feats_of(ids), ids, blocks)
                (loss / args.batch).backward()
                tot_l += float(loss.detach())
                tot_a += float(acc)
                n += 1
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[step {step}/{total}] loss {tot_l / max(n, 1):.3f} acc1 {tot_a / max(n, 1):.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            if step % args.eval_every == 0 and step < total:
                show(f"step {step}", evaluate())
    end = evaluate()
    show("final", end)

    table = []
    for a, b in zip(start, end):
        table.append({"arm": args.code, "lag": a["lag"], "tau start": a["tau"], "tau end": b["tau"],
                      "first start": a["first token"], "first end": b["first token"],
                      "loss start": a["loss"], "loss end": b["loss"], "blocks": b["blocks"]})
    print_table(table, list(table[0]), f"In-flight positions read: {args.code} (drafter trained with it, "
                                       f"{total} steps; tau = accepted drafted tokens + 1)")
    save_json(args.save, {"args": vars(args), "start": start, "end": end, "table": table})
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"draft": {k: v.detach().to(torch.bfloat16) for k, v in draft.state_dict().items()},
                    "fc": {k: v.detach().to(torch.bfloat16) for k, v in fc.state_dict().items()},
                    "code": code.state_dict() if code is not None else None, "args": vars(args)}, out / "state.pt")
        print(f"saved {out / 'state.pt'}", flush=True)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
