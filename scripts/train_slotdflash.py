"""DFlash with lookahead-slot context: does the target's trained lookahead help the real drafter?

The official DFlash drafter reads target features (layers 1/9/17/25/33) for positions [0, s) and drafts
s+1 .. s+15 from [x_s, MASK x 15]. Arms (all fine-tune the official drafter, same data / steps / lr):
  control   context = features [0, s)                                   (fine-tuning alone: 5.83 -> 6.16)
  slots     context = features [0, s) + the 8 lookahead-slot features at positions s .. s+7, computed by
            the target with the slot-gated LoRA of exp40 (in deployment: in the verify pass, lossless)
  oracle    context = features [0, s) + the TRUE features at s .. s+7    (upper bound: real context)
The drafter needs no new modules: extra context positions enter through its existing K/V injection.
Validation: tau (leading correct + 1, greedy truth) and acc@k on held-out sequences the LoRA never saw.

  python scripts/train_slotdflash.py --arm slots --lora .../lookahead_lora.pt --data .../data --out ...
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

from dflash.model import _draft_value, _output_head, _raw_input_embeddings  # noqa: E402
from exp40_lookahead_lora import FEAT_LAYERS, GatedLoRA, add_lora, sliced_cache  # noqa: E402

from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402


def lead(hits):
    n = 0
    while n < len(hits) and hits[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["control", "slots", "oracle"])
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--lora", default="", help="exp40 --save file (needed for --arm slots)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-seqs", type=int, default=2000)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--val-anchors", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=300, help="held-out size when no --lora file is given")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rng = random.Random(0)

    target = load_target(args.target, dev)
    target.requires_grad_(False)
    draft = load_draft(args.draft, dev).float()
    draft.train()
    bs = draft.block_size
    lids = list(draft.target_layer_ids)
    m = 8
    if args.arm == "slots":
        assert lids == FEAT_LAYERS, "the LoRA was trained on DFlash's feature layers"
        sv = torch.load(args.lora)
        la = sv["args"]
        m = la["m"]
        loras = add_lora(target, la["rank"], la["alpha"])
        with torch.no_grad():
            for mod, A, B in zip(loras, sv["A"], sv["B"]):
                mod.A.copy_(A)
                mod.B.copy_(B)
        for mod in loras:
            mod.A.requires_grad_(False)
            mod.B.requires_grad_(False)
        mid = load_tokenizer(args.target).convert_tokens_to_ids(la["mask_token"])
        val_keys = {tuple(x.tolist()) for x in sv["val_ids"]}
    else:
        val_keys = None
    GatedLoRA.on = False

    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
        if len(data) >= 4 * args.train_seqs:
            break
    data = [d for d in data if len(d["ids"]) <= args.max_len and len(d["ids"]) - d["n_prompt"] > bs + 2]
    if val_keys is None and args.lora:                       # same held-out set for every arm
        val_keys = {tuple(x.tolist()) for x in torch.load(args.lora)["val_ids"]}
    if val_keys:
        val = [d for d in data if tuple(d["ids"].tolist()) in val_keys]
        train = [d for d in data if tuple(d["ids"].tolist()) not in val_keys]
    else:
        rng.shuffle(data)
        val, train = data[: args.val_seqs], data[args.val_seqs :]
    rng.shuffle(train)
    train = train[: args.train_seqs]
    print(f"[{args.arm}] {len(train)} train / {len(val)} val sequences, {m} lookahead positions", flush=True)

    @torch.no_grad()
    def target_ctx(ids, anchors):
        """features [T, W] and, per anchor, the extra context [m, W] (or None)."""
        out = target.model(ids[None], use_cache=args.arm == "slots", output_hidden_states=True)
        feats = torch.cat([out.hidden_states[l + 1][0] for l in lids], -1)
        extra = []
        for s in anchors:
            if args.arm == "control":
                extra.append(None)
            elif args.arm == "oracle":
                extra.append(feats[s : s + m])
            else:
                GatedLoRA.on = True
                x = torch.full((1, m), mid, device=dev)
                so = target.model(x, past_key_values=sliced_cache(out.past_key_values, s), use_cache=True,
                                  output_hidden_states=True)
                GatedLoRA.on = False
                extra.append(torch.cat([so.hidden_states[l + 1][0] for l in FEAT_LAYERS], -1))
        return feats, extra

    head = _output_head(target)
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))

    def forward(ids, feats, extra, anchors):
        T = feats.shape[0]
        ctx = [feats] + [e for e in extra if e is not None]
        cpos = [torch.arange(T, device=dev)] + [torch.arange(s, s + len(e), device=dev)
                                               for s, e in zip(anchors, extra) if e is not None]
        C = sum(len(c) for c in ctx)
        toks, bpos = [], []
        for s in anchors:
            t = torch.full((bs,), draft.mask_token_id, dtype=torch.long, device=dev)
            t[0] = ids[s]
            toks.append(t)
            bpos.append(torch.arange(s, s + bs, device=dev))
        Q = bs * len(anchors)
        mask = torch.zeros(Q, C + Q, dtype=torch.bool, device=dev)
        off = T
        for b, (s, e) in enumerate(zip(anchors, extra)):
            r = slice(b * bs, (b + 1) * bs)
            mask[r, :s] = True
            if e is not None:
                mask[r, off : off + len(e)] = True
                off += len(e)
            mask[r, C + b * bs : C + (b + 1) * bs] = True
        noise = _raw_input_embeddings(target, torch.cat(toks)[None], scale).float()
        pos = torch.cat(cpos + bpos)[None]
        h = draft(target_hidden=torch.cat(ctx)[None].float(), noise_embedding=noise, position_ids=pos,
                  attention_mask=mask[None, None], past_key_values=None, use_cache=False)[0]
        h = h.view(len(anchors), bs, -1)[:, 1:]
        return draft.compute_logits(h.to(head.weight.dtype), head).float()          # [nb, bs-1, V]

    def anchors_of(d, n, r):
        T = len(d["ids"])
        lo, hi = max(d["n_prompt"], 1), T - bs
        return r.sample(range(lo, hi), min(n, hi - lo)) if hi > lo else []

    history = []

    @torch.no_grad()
    def evaluate(step):
        draft.eval()
        r = random.Random(123)
        taus, acc, cnt = [], torch.zeros(bs - 1), 0
        for d in val:
            ids = d["ids"].long().to(dev)
            anc = anchors_of(d, args.val_anchors, r)
            if not anc:
                continue
            feats, extra = target_ctx(ids, anc)
            pred = forward(ids, feats, extra, anc).argmax(-1)
            for b, s in enumerate(anc):
                hits = (pred[b] == ids[s + 1 : s + bs]).cpu()
                taus.append(1 + lead(hits.tolist()))
                acc += hits.float()
                cnt += 1
        draft.train()
        a = (acc / cnt).tolist()
        rec = {"step": step, "tau": sum(taus) / len(taus), "acc": a}
        history.append(rec)
        print(f"[{args.arm} step {step}] tau {rec['tau']:.3f} acc@1 {a[0]:.3f} acc@4 {a[3]:.3f} acc@8 {a[min(7, len(a) - 1)]:.3f} "
              f"acc@15 {a[-1]:.3f}", flush=True)

    if dev == "cpu":
        opt = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(draft.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(st, total) / total)))
    w = args.gamma ** torch.arange(bs - 1, device=dev, dtype=torch.float32)
    evaluate(0)
    t0 = time.time()
    for step, d in enumerate(train, 1):
        ids = d["ids"].long().to(dev)
        anc = anchors_of(d, args.anchors, rng)
        if not anc:
            continue
        feats, extra = target_ctx(ids, anc)
        logits = forward(ids, feats, extra, anc)
        lab = torch.stack([ids[s + 1 : s + bs] for s in anc])
        ce = F.cross_entropy(logits.flatten(0, 1), lab.flatten(), reduction="none").view(len(anc), -1)
        loss = (ce * w).sum() / (w.sum() * len(anc))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"[{args.arm} step {step}/{total}] loss {float(loss):.3f} {(time.time() - t0) / step:.2f}s/seq",
                  flush=True)
        if step % args.eval_every == 0:
            evaluate(step)
    evaluate(total)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import json
    (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    print(f"[{args.arm}] done", flush=True)


if __name__ == "__main__":
    main()
