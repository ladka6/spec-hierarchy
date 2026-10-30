"""Experiment 75: upper bound for a "future summary". Does DFlash draft better if it gets a compact, TRUE summary of
the block it is drafting? (offline, Colab-ready)

If even the true summary does not help, there is nothing worth having the target predict. If it helps, the next step
is whether the target can predict that summary in its verify pass (one placeholder + gated LoRA, exp72 machinery):
one vector is a much easier target than 16 per-position states, and it needs no per-position token.

Summary of the block at anchor s: the target's real states at its deepest feature layer for positions s+1 .. s+B
(they have read the future tokens), averaged over --chunks equal chunks (1 = one mean; 4 = NCM-like coarse codes),
then squeezed through a learned bottleneck of d dimensions (down H -> d per chunk, up d*chunks -> drafter width).
d limits how much it can leak the tokens; the result is tau as a function of d.
The summary is ADDED to the drafter's input embeddings of the whole block (anchor + masks): no change to attention.

Fair comparison (DFlash has never seen such an input): every condition, including d = 0 (no summary), is the same
short fine-tune of DFlash from its checkpoint, same data, steps, lr and seed. "dflash_orig" = the untouched checkpoint.

  python scripts/exp75_futuresummary.py --cache /content/seqs.pt                      # d in 0 16 64 256
  python scripts/exp75_futuresummary.py --cache /content/seqs.pt --chunks 4 --ds 0 16 64
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import _output_head, extract_context_feature  # noqa: E402

from hspec.lagtrain import pack  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


class Bottleneck(nn.Module):
    def __init__(self, H, d, chunks, out_dim):
        super().__init__()
        self.down = nn.Linear(H, d)
        self.up = nn.Linear(d * chunks, out_dim)
        nn.init.zeros_(self.up.weight)          # starts as "no summary": the fine-tune begins exactly at DFlash
        nn.init.zeros_(self.up.bias)

    def forward(self, chunk_means):             # [A, chunks, H] -> [A, out_dim]
        z = self.down(chunk_means)
        return self.up(z.flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences, e.g. from exp69")
    ap.add_argument("--ds", nargs="+", type=int, default=[0, 16, 64, 256], help="bottleneck sizes (0 = no summary)")
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--blocks", type=int, default=16, help="anchors per sequence per step")
    ap.add_argument("--lr", type=float, default=3e-5, help="drafter")
    ap.add_argument("--lr-b", type=float, default=1e-3, help="bottleneck")
    ap.add_argument("--gamma", type=float, default=0.9, help="position weight decay in the loss (as DFlash)")
    ap.add_argument("--anchors", type=int, default=500, help="eval anchors")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    args.save = args.save or f"exp75_futuresummary_c{args.chunks}"
    dev = "cuda"
    t0 = time.time()

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    nv = int(len(seqs) * args.val_frac)
    val, train = seqs[:nv], seqs[nv:]
    target = load_target(args.target, dev)
    for p_ in target.parameters():
        p_.requires_grad_(False)
    head = _output_head(target)
    H = target.config.hidden_size

    probe = load_draft(args.draft, dev)
    lids, bs = list(probe.target_layer_ids), probe.block_size
    B = bs - 1
    deep = lids[-1] + 1                          # hidden_states index of the deepest feature layer
    del probe
    torch.cuda.empty_cache()

    @torch.no_grad()
    def target_pass(x):
        out = target(x[None], output_hidden_states=True, use_cache=False)
        return extract_context_feature(out.hidden_states, lids), out.hidden_states[deep][0].float()

    def summary_input(hdeep, anchors):
        """[A, chunks, H]: chunk means of the real deep states at positions s+1 .. s+B."""
        blk = torch.stack([hdeep[s + 1 : s + 1 + B] for s in anchors])               # [A, B, H]
        return torch.stack([c.mean(1) for c in blk.chunk(args.chunks, dim=1)], 1)

    def run(draft, bott, feats, x, anchors, hdeep, train_mode):
        """Packed DFlash pass over all anchors, with the summary added to each block's input embeddings."""
        noise, pos, mask, starts = pack(draft, target, feats, x, [(s, 0) for s in anchors], bs)
        if bott is not None:
            add = bott(summary_input(hdeep, anchors)).to(noise.dtype)                  # [A, W]
            rows = torch.zeros(noise.shape[1], add.shape[1], device=dev, dtype=noise.dtype)
            for a, st in enumerate(starts):
                rows = rows.index_add(0, torch.arange(st - 1, st + B, device=dev),
                                      add[a][None].expand(B + 1, -1))
            noise = noise + rows[None]
        hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                    past_key_values=None, use_cache=False)[0]
        hb = torch.stack([hid[st : st + B] for st in starts])                          # [A, B, W]
        return draft.compute_logits(hb, head).float()                                  # [A, B, V]

    def anchors_in(ids, lo, n, rng):
        hi = len(ids) - bs - 1
        if hi <= lo + 1:
            return []
        return sorted(rng.sample(range(lo + 1, hi), min(n, hi - lo - 1)))

    # fixed eval anchors on held-out sequences
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 1, len(ids) - bs - 1)]
    random.Random(0).shuffle(cand)
    ev = {}
    for i, s in sorted(cand[: args.anchors]):
        ev.setdefault(i, []).append(s)

    @torch.no_grad()
    def evaluate(draft, bott):
        draft.eval()
        tau, acc, acc4, n = 0.0, 0.0, 0.0, 0
        for i, anchors in ev.items():
            x = val[i][0].to(dev)
            feats, hdeep = target_pass(x)
            lg = run(draft, bott, feats, x, anchors, hdeep, False)
            pred = lg.argmax(-1)
            for a, s in enumerate(anchors):
                truth = x[s + 1 : s + 1 + B]
                r = (pred[a] == truth).float()
                tau += matched(pred[a].tolist(), truth.tolist()) + 1
                acc += float(r.mean())
                acc4 += float(r[:4].mean())
                n += 1
        return {"tau": tau / n, "acc": acc / n, "acc@1-4": acc4 / n}

    rows_out = []
    d0 = load_draft(args.draft, dev)
    r = evaluate(d0, None)
    rows_out.append({"condition": "dflash_orig", "d": "-", **r})
    print(f"[eval] dflash_orig: {r}", flush=True)
    del d0
    torch.cuda.empty_cache()

    w = torch.tensor([args.gamma ** k for k in range(B)], device=dev)
    for d in args.ds:
        torch.manual_seed(0)
        rng = random.Random(1)
        draft = load_draft(args.draft, dev)
        draft.train()
        width = draft.config.hidden_size
        bott = Bottleneck(H, d, args.chunks, width).to(dev) if d > 0 else None
        groups = [{"params": [p for p in draft.parameters() if p.requires_grad], "lr": args.lr}]
        if bott is not None:
            groups.append({"params": list(bott.parameters()), "lr": args.lr_b})
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))
        log = []
        for step in range(args.steps):
            ids, lo = train[rng.randrange(len(train))]
            x = ids.to(dev)
            anchors = anchors_in(ids, lo, args.blocks, rng)
            if not anchors:
                continue
            feats, hdeep = target_pass(x)
            lg = run(draft, bott, feats, x, anchors, hdeep, True)                     # [A, B, V]
            labels = torch.stack([x[s + 1 : s + 1 + B] for s in anchors])
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels.reshape(-1), reduction="none").view(-1, B)
            loss = (ce * w).sum(1).mean() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            log.append((float(loss.detach()), float((lg[:, 0].argmax(-1) == labels[:, 0]).float().mean())))
            if step % 100 == 0 or step == args.steps - 1:
                tail = log[-100:]
                print(f"[train d={d}] step {step}: loss {sum(a for a, _ in tail)/len(tail):.4f}  "
                      f"acc@1 {sum(b for _, b in tail)/len(tail):.3f}  {time.time()-t0:.0f}s", flush=True)
        r = evaluate(draft, bott)
        rows_out.append({"condition": "finetune" if d == 0 else "finetune + true summary", "d": d, **r})
        print(f"[eval] d={d}: {r}", flush=True)
        del draft, bott, opt
        torch.cuda.empty_cache()

    print_table(rows_out, list(rows_out[0]), f"upper bound: DFlash fine-tuned with a TRUE summary of the block "
                                             f"({args.chunks} chunk mean(s) of the real L{deep - 1} states through a "
                                             f"d-dim bottleneck) vs the same fine-tune without (d = 0); tau out of {B + 1}")
    save_json(args.save, {"args": vars(args), "rows": rows_out})
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
