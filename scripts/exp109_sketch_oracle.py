"""Experiment 109: amortized lookahead, oracle ceiling. Does a block drafter gain from right-hand anchors (a sketch of
the future answer), if the sketch is perfect? (Colab / Snellius; --data for a large training set)

Idea: a sketch of the whole answer (sparse future tokens) is produced once and amortised over many blocks; the block
drafter becomes an infiller between the verified left context and the sketch's right anchors. This script measures
the ceiling with ORACLE sketches (the target's own future tokens) and how fast it degrades with wrong anchors. If the
oracle gain is small, the idea is dead before any sketch source is built.

Sketch layouts (anchors are true target tokens at chosen positions, corrupted with probability q):
  beyond   J anchors after the block: positions s + B + gap + (j - 1) * P, j = 1..J   (no in-block help: clean)
  grid     sparse grid every P tokens from the response start (as a real sketch would be); grid positions inside the
           block are revealed (excluded from the loss; in the chain they are the sketch's token, so a corrupted one
           is an error), grid positions after the block up to --lookahead are appended as anchors
Anchors are extra drafter inputs with their own position ids (RoPE handles the gaps); the block attends to them
bidirectionally, exactly like to its own masks. Corruption = a token from another position of the same sequence
(plausible but wrong).

Conditions (tau = matched drafted tokens + 1, out of block_size, on held-out anchors):
  dflash_orig          untouched drafter, no sketch
  control              same fine-tune without anchors (same data, steps, lr, seed): separates the fine-tune effect
  sketch, q            sketch-trained drafter with anchors corrupted at rate q (q = 0 is the oracle ceiling)
  sketch, none         sketch-trained drafter without anchors (does it still work when the sketch is missing?)
In grid mode also tau_net = tau - revealed tokens inside the matched prefix (gain from the anchors themselves).

  python scripts/exp109_sketch_oracle.py --data /scratch-shared/$USER/hspec_lagft/data --cache ~/seqs.pt \
      --layout beyond --period 16 --n-right 2
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

from dflash.model import _draft_value, _output_head, _raw_input_embeddings, extract_context_feature  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


def load_pairs(path):
    out = []
    for r in torch.load(path):
        ids, lo = (r["ids"], r["n_prompt"]) if isinstance(r, dict) else r
        out.append((torch.as_tensor(ids).long(), int(lo)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="", help="directory of gen_traindata shards (training)")
    ap.add_argument("--cache", default="", help="(ids, n_prompt) sequences for evaluation (and training without --data)")
    ap.add_argument("--layout", choices=["beyond", "grid"], default="beyond")
    ap.add_argument("--period", type=int, default=16, help="P: distance between sketch anchors")
    ap.add_argument("--n-right", type=int, default=2, help="beyond: number of anchors after the block")
    ap.add_argument("--gap", type=int, default=-1, help="beyond: first anchor at s + B + gap (default: P)")
    ap.add_argument("--lookahead", type=int, default=-1, help="grid: anchors up to s + B + lookahead (default: 2P)")
    ap.add_argument("--q-train", type=float, default=0.15, help="anchor corruption rate during training")
    ap.add_argument("--p-none", type=float, default=0.15, help="share of training blocks without any anchor")
    ap.add_argument("--q-eval", type=float, nargs="+", default=[0.0, 0.2, 0.5])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--blocks", type=int, default=8, help="anchors per sequence per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--anchors", type=int, default=600, help="evaluation anchors")
    ap.add_argument("--val-frac", type=float, default=0.2, help="eval share of --cache when it is also the train set")
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    P = args.period
    gap = P if args.gap < 0 else args.gap
    look = 2 * P if args.lookahead < 0 else args.lookahead
    args.save = args.save or f"exp109_sketch_{args.layout}_P{P}" + (f"_J{args.n_right}" if args.layout == "beyond" else "")
    dev, t0 = "cuda", time.time()

    # ---- data
    if args.data:
        train = []
        for f in sorted(Path(args.data).expanduser().glob("*.pt")):
            train += [(r["ids"].long(), int(r["n_prompt"])) for r in torch.load(f)]
        val = load_pairs(args.cache) if args.cache else []
        if not val:
            random.Random(0).shuffle(train)
            nv = max(int(0.03 * len(train)), 30)
            val, train = train[:nv], train[nv:]
    else:
        seqs = load_pairs(args.cache)
        random.Random(0).shuffle(seqs)
        nv = int(len(seqs) * args.val_frac)
        val, train = seqs[:nv], seqs[nv:]

    target = load_target(args.target, dev)
    for p_ in target.parameters():
        p_.requires_grad_(False)
    head = _output_head(target)
    probe = load_draft(args.draft, dev)
    lids, bs, mask_id = list(probe.target_layer_ids), probe.block_size, probe.mask_token_id
    scale = float(_draft_value(probe.config, "input_embedding_scale", 1.0))
    B = bs - 1
    del probe
    torch.cuda.empty_cache()
    w = torch.tensor([args.gamma ** i for i in range(B)], device=dev)
    reach = B + (gap + (args.n_right - 1) * P if args.layout == "beyond" else look)    # furthest anchor offset
    train = [(ids, lo) for ids, lo in train if len(ids) - lo > bs + reach + 4]
    val = [(ids, lo) for ids, lo in val if len(ids) - lo > bs + reach + 4]
    print(f"[data] {len(train)} train / {len(val)} eval sequences, block {bs}, layout {args.layout}, P {P}, "
          f"reach {reach}", flush=True)

    @torch.no_grad()
    def target_feats(x):
        out = target(x[None], output_hidden_states=True, use_cache=False)
        return extract_context_feature(out.hidden_states, lids)

    def sketch(x, lo, s, q, rng, none=False):
        """(revealed {offset k: token}, right anchors [(pos, token)]) for the block at anchor s."""
        if none:
            return {}, []
        T = len(x)
        if args.layout == "beyond":
            pos = [s + B + gap + j * P for j in range(args.n_right)]
            inside = []
        else:
            first = lo + ((s + 1 - lo + P - 1) // P) * P                      # first grid point > s
            grid = list(range(first, s + B + look + 1, P))
            inside = [p for p in grid if p <= s + B]
            pos = [p for p in grid if p > s + B]
        pos = [p for p in pos if p < T]

        def tok(p):
            if q > 0 and rng.random() < q:
                p2 = rng.randrange(lo, T)
                while p2 == p and T - lo > 1:
                    p2 = rng.randrange(lo, T)
                return int(x[p2])
            return int(x[p])

        return {p - s: tok(p) for p in inside}, [(p, tok(p)) for p in pos]

    def pack(feats, x, blocks):
        """blocks: [(s, revealed, right)]. Returns noise [1,Q,H], pos [1,T+Q], mask [1,1,Q,T+Q], starts."""
        T = feats.shape[1]
        toks, poss, spans, starts = [], [], [], []
        q = 0
        for s, rev, right in blocks:
            t = torch.full((bs + len(right),), mask_id, dtype=torch.long, device=dev)
            t[0] = x[s]
            for k, v in rev.items():
                t[k] = v
            for j, (_p, v) in enumerate(right):
                t[bs + j] = v
            p = torch.cat([torch.arange(s, s + bs, device=dev),
                           torch.tensor([pp for pp, _ in right], dtype=torch.long, device=dev)])
            toks.append(t)
            poss.append(p)
            spans.append((q, q + len(t), s))
            starts.append(q)
            q += len(t)
        noise = _raw_input_embeddings(target, torch.cat(toks)[None], scale)
        pos = torch.cat([torch.arange(T, device=dev)] + poss)[None]
        mask = torch.zeros((q, T + q), dtype=torch.bool, device=dev)
        for a, b, s in spans:
            mask[a:b, :s] = True
            mask[a:b, T + a : T + b] = True
        return noise, pos, mask[None, None], starts

    def block_logits(draft, feats, x, blocks):
        """[A, B, V]: logits for positions s+1 .. s+B of every block."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            noise, pos, mask, starts = pack(feats, x, blocks)
            hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                        past_key_values=None, use_cache=False)[0]
            hb = torch.stack([hid[st + 1 : st + 1 + B] for st in starts])
            return draft.compute_logits(hb, head)

    def anchors_in(ids, lo, n, rng):
        hi = len(ids) - bs - 1 - reach
        if hi <= lo + 1:
            return []
        return sorted(rng.sample(range(lo + 1, hi), min(n, hi - lo - 1)))

    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 1, len(ids) - bs - 1 - reach)]
    random.Random(0).shuffle(cand)
    ev = {}
    for i, s in sorted(cand[: args.anchors]):
        ev.setdefault(i, []).append(s)

    @torch.no_grad()
    def evaluate(draft, q, none):
        draft.eval()
        rng = random.Random(7)                       # same corruptions for every drafter
        tau, net, n = 0.0, 0.0, 0
        for i, an in ev.items():
            ids, lo = val[i]
            x = ids.to(dev)
            feats = target_feats(x)
            blocks = []
            for s in an:
                rev, right = sketch(x, lo, s, q, rng, none)
                blocks.append((s, rev, right))
            pred = block_logits(draft, feats, x, blocks).argmax(-1).tolist()
            for (s, rev, _r), ch in zip(blocks, pred):
                for k, v in rev.items():
                    ch[k - 1] = v
                truth = x[s + 1 : s + 1 + B].tolist()
                m = matched(ch, truth)
                tau += m + 1
                net += m + 1 - sum(1 for k in rev if k <= m)
                n += 1
        draft.train()
        return tau / n, net / n

    def finetune(with_sketch):
        torch.manual_seed(0)
        rng = random.Random(2)
        draft = load_draft(args.draft, dev).float()          # fp32 master weights, bf16 autocast (as exp77)
        draft.train()
        params = [p for p in draft.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))
        log = []
        for step in range(args.steps):
            ids, lo = train[rng.randrange(len(train))]
            x = ids.to(dev)
            an = anchors_in(ids, lo, args.blocks, rng)
            if not an:
                continue
            feats = target_feats(x)
            blocks, keep = [], []
            for s in an:
                none = (not with_sketch) or rng.random() < args.p_none
                rev, right = sketch(x, lo, s, args.q_train, rng, none)
                blocks.append((s, rev, right))
                km = torch.ones(B, device=dev)
                for k in rev:
                    km[k - 1] = 0.0                                   # revealed positions are not predicted
                keep.append(km)
            labels = torch.stack([x[s + 1 : s + 1 + B] for s in an])
            lg = block_logits(draft, feats, x, blocks).float()
            ce = F.cross_entropy(lg.flatten(0, 1), labels.flatten(), reduction="none").view(len(an), B)
            wk = w[None] * torch.stack(keep)
            loss = (ce * wk).sum() / wk.sum().clamp_min(1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            log.append(float(loss.detach()))
            if step % 100 == 0 or step == args.steps - 1:
                tail = log[-100:]
                print(f"[train {'sketch' if with_sketch else 'control'}] step {step}: loss "
                      f"{sum(tail) / len(tail):.4f}  {time.time() - t0:.0f}s", flush=True)
        return draft

    rows = []
    d0 = load_draft(args.draft, dev)
    t, _ = evaluate(d0, 0.0, True)
    rows.append({"condition": "dflash_orig, no sketch", "q": "-", "tau": t, "tau_net": t})
    print(f"[eval] dflash_orig {t:.3f}", flush=True)
    for q in args.q_eval:                                       # untrained drafter given anchors (sanity)
        t, tn = evaluate(d0, q, False)
        rows.append({"condition": "dflash_orig + sketch (untrained for it)", "q": q, "tau": t, "tau_net": tn})
    del d0
    torch.cuda.empty_cache()

    if not args.skip_control:
        dc = finetune(False)
        t, _ = evaluate(dc, 0.0, True)
        rows.append({"condition": "control: fine-tuned, no sketch", "q": "-", "tau": t, "tau_net": t})
        print(f"[eval] control {t:.3f}", flush=True)
        del dc
        torch.cuda.empty_cache()

    ds = finetune(True)
    t, _ = evaluate(ds, 0.0, True)
    rows.append({"condition": "sketch drafter, no sketch", "q": "-", "tau": t, "tau_net": t})
    for q in args.q_eval:
        t, tn = evaluate(ds, q, False)
        rows.append({"condition": "sketch drafter + sketch", "q": q, "tau": t, "tau_net": tn})
        print(f"[eval] sketch q={q}: tau {t:.3f} net {tn:.3f}", flush=True)

    print_table(rows, list(rows[0]), f"exp109 sketch oracle: layout {args.layout}, P {P}"
                + (f", {args.n_right} anchors from s+B+{gap}" if args.layout == "beyond" else f", lookahead {look}")
                + f"; tau out of {bs}; q = anchor corruption rate")
    save_json(args.save, {"args": vars(args), "rows": rows})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
