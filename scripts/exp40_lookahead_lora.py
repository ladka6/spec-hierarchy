"""Experiment 40: lookahead adapters. Can a slot-gated LoRA make the target compute the missing context?

exp39: a frozen target computes no lookahead at mask slots. Here a LoRA on every linear layer of the
target is active ONLY on m mask slots appended after the verified prefix (the normal positions run
the untouched target, so verification stays lossless). It is trained so that the slot states match the
ORACLE states: the target's own hidden states at positions s .. s+m-1 computed on the TRUE tokens
(DFlash's feature layers 1/9/17/25/33 + the final layer), plus CE of the slot logits on the next tokens.
That is the context whose absence causes most drafter stops.

Training is cheap: the prefix runs once without grad (KV cache); only the m slot tokens run with grad.
Evaluation on held-out sequences (never seen by the LoRA):
  1. cosine(slot state, oracle state) per offset, final layer, before vs after training
  2. direct drafting: top-1 of the target's LM head at slot s+k-1 vs x_(s+k) (cf. official DFlash acc@k)
  3. the exp39 probe: A (base) / B (+ trained slot states) / O (oracle); share of the A->O gap closed

  python scripts/exp40_lookahead_lora.py --data .../hspec_lagft/data
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
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp39_lookahead import Probe  # noqa: E402

from hspec.models import load_target, load_tokenizer  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

FEAT_LAYERS = [1, 9, 17, 25, 33]           # DFlash's target layers (hidden_states index = layer + 1)


class GatedLoRA(nn.Module):
    """base(x) + gate * B(A(x)); gate is a module-level switch (on only for the slot pass)."""
    on = False

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        if GatedLoRA.on:
            y = y + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(y.dtype)
        return y


def add_lora(model, r, alpha):
    mods = []
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for n in names:
                m = GatedLoRA(getattr(parent, n), r, alpha)
                setattr(parent, n, m)
                mods.append(m)
    return mods


def sliced_cache(full, s):
    return DynamicCache(ddp_cache_data=[(l.keys[:, :, :s], l.values[:, :, :s]) for l in full.layers])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-seqs", type=int, default=1600)
    ap.add_argument("--probe-seqs", type=int, default=600)
    ap.add_argument("--val-seqs", type=int, default=300)
    ap.add_argument("--anchors", type=int, default=6)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--mask-token", default="<|fim_pad|>")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=128.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ce", type=float, default=0.3, help="weight of the next-token CE at the slots")
    ap.add_argument("--probe-epochs", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="exp40_lookahead_lora")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer(args.target)
    target = load_target(args.target, dev)
    target.requires_grad_(False)
    H = target.config.hidden_size
    mid = tok.convert_tokens_to_ids(args.mask_token)
    loras = add_lora(target, args.rank, args.alpha)
    params = [p for m in loras for p in (m.A, m.B)]
    print(f"gated LoRA r{args.rank}: {sum(p.numel() for p in params) / 1e6:.0f} M params, mask {args.mask_token}={mid}",
          flush=True)
    head = target.lm_head
    emb = target.model.embed_tokens

    data = []
    need = args.train_seqs + args.probe_seqs + args.val_seqs
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
        if len(data) >= 3 * need:
            break
    data = [d for d in data if len(d["ids"]) <= args.max_len and len(d["ids"]) - d["n_prompt"] > args.m + args.K + 4]
    rng.shuffle(data)
    val = data[: args.val_seqs]
    pro = data[args.val_seqs : args.val_seqs + args.probe_seqs]
    tr = data[args.val_seqs + args.probe_seqs : need]
    print(f"{len(tr)} LoRA-train / {len(pro)} probe-train / {len(val)} val sequences", flush=True)

    def anchors_of(d, r):
        T = len(d["ids"])
        lo, hi = max(d["n_prompt"], 1), T - max(args.m, args.K) - 1
        return r.sample(range(lo, hi), min(args.anchors, hi - lo))

    @torch.no_grad()
    def prefix_pass(ids):
        GatedLoRA.on = False
        out = target.model(ids[None], use_cache=True, output_hidden_states=True)
        hs = out.hidden_states
        oracle = torch.stack([hs[l + 1][0] for l in FEAT_LAYERS] + [out.last_hidden_state[0]])   # [6, T, H]
        return out.past_key_values, oracle

    def slot_pass(cache, s, grad):
        GatedLoRA.on = True
        x = torch.full((1, args.m), mid, device=dev)
        with torch.set_grad_enabled(grad):
            out = target.model(x, past_key_values=sliced_cache(cache, s), use_cache=True, output_hidden_states=True)
            hs = out.hidden_states
            st = torch.stack([hs[l + 1][0] for l in FEAT_LAYERS] + [out.last_hidden_state[0]])   # [6, m, H]
        GatedLoRA.on = False
        return st

    def slot_loss(st, oracle, ids, s):
        o = oracle[:, s : s + args.m].float()
        p = st.float()
        cos = 1 - F.cosine_similarity(p, o, dim=-1).mean()
        mse = ((p - o) ** 2).mean() / (o ** 2).mean().clamp_min(1e-6)
        logits = head(st[-1]).float()                                   # slot at s+k-1 -> x_(s+k)
        ce = F.cross_entropy(logits, ids[s + 1 : s + 1 + args.m])
        return cos + 0.1 * mse + args.ce * ce, float(cos.detach()), float(ce.detach())

    @torch.no_grad()
    def evaluate(tag):
        r = random.Random(7)
        cos_k = torch.zeros(args.m)
        hit = torch.zeros(args.m)
        n = 0
        for d in val:
            ids = d["ids"].long().to(dev)
            cache, oracle = prefix_pass(ids)
            for s in anchors_of(d, r):
                st = slot_pass(cache, s, False)
                cos_k += F.cosine_similarity(st[-1].float(), oracle[-1, s : s + args.m].float(), dim=-1).cpu()
                hit += (head(st[-1]).argmax(-1) == ids[s + 1 : s + 1 + args.m]).float().cpu()
                n += 1
        row = {"model": tag, **{f"cos@{k + 1}": float(cos_k[k] / n) for k in (0, 1, 3, args.m - 1)},
               **{f"top1@{k + 1}": float(hit[k] / n) for k in range(args.m)}}
        print(f"[eval {tag}] " + " ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "model"), flush=True)
        return row

    rows = [evaluate("frozen (exp39)")]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(st, total) / total)))
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        rng.shuffle(tr)
        for d in tr:
            ids = d["ids"].long().to(dev)
            cache, oracle = prefix_pass(ids)
            opt.zero_grad(set_to_none=True)
            cs, ces, na = 0.0, 0.0, 0
            for s in anchors_of(d, rng):
                loss, c, ce = slot_loss(slot_pass(cache, s, True), oracle, ids, s)
                (loss / args.anchors).backward()
                cs += c
                ces += ce
                na += 1
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"[train ep {ep + 1} step {step}/{total}] 1-cos {cs / max(na, 1):.3f} ce {ces / max(na, 1):.3f} "
                      f"{(time.time() - t0) / step:.2f}s/seq", flush=True)
        rows.append(evaluate(f"lora epoch {ep + 1}"))
    print_table(rows, list(rows[0]), "slots after the verified prefix: cosine to the oracle state (final layer) and "
                                     "direct top-1 for x_(s+k); official DFlash acc@1/4/8 = 0.818/0.585/0.397")

    # probe (exp39 protocol) with the trained slots, final layer, on sequences unseen by the LoRA
    feats = {"tr": {"A": [], "S": [], "O": [], "y": []}, "va": {"A": [], "S": [], "O": [], "y": []}}
    with torch.no_grad():
        for split, seqs in (("tr", pro), ("va", val)):
            r = random.Random(11)
            for d in seqs:
                ids = d["ids"].long().to(dev)
                cache, oracle = prefix_pass(ids)
                for s in anchors_of(d, r):
                    st = slot_pass(cache, s, False)
                    f = feats[split]
                    f["A"].append(torch.stack([oracle[-1, s - 1], emb(ids[s])]).to(torch.bfloat16).cpu())
                    f["S"].append(st[-1].to(torch.bfloat16).cpu())
                    f["O"].append(oracle[-1, s : s + args.m].to(torch.bfloat16).cpu())
                    f["y"].append(ids[s + 1 : s + 1 + args.K].cpu())
    W = head.weight
    res = {}
    for name in ("A_base", "B_slots", "O_oracle"):
        def Z(split):
            f = feats[split]
            A = torch.stack(f["A"])
            if name == "A_base":
                return A
            return torch.cat([A, torch.stack(f["S"] if name == "B_slots" else f["O"])], 1)
        Ztr, Zva = Z("tr"), Z("va")
        Ytr, Yva = torch.stack(feats["tr"]["y"]), torch.stack(feats["va"]["y"])
        torch.manual_seed(0)
        probe = Probe(Ztr.shape[1], H, args.K).to(dev)
        popt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=0.05)
        best = None
        for ep in range(args.probe_epochs):
            perm = torch.randperm(len(Ztr))
            for b in range(0, len(perm), 256):
                bi = perm[b : b + 256]
                logits = probe(Ztr[bi].to(dev).float()).to(W.dtype) @ W.T
                loss = F.cross_entropy(logits.float().flatten(0, 1), Ytr[bi].to(dev).flatten())
                popt.zero_grad(set_to_none=True)
                loss.backward()
                popt.step()
            with torch.no_grad():
                hits, lead = torch.zeros(args.K), 0.0
                for b in range(0, len(Zva), 256):
                    ok = (probe(Zva[b : b + 256].to(dev).float()).to(W.dtype) @ W.T).argmax(-1) == Yva[b : b + 256].to(dev)
                    hits += ok.float().sum(0).cpu()
                    lead += float(ok.long().cumprod(1).sum())
                acc, lead = hits / len(Zva), lead / len(Zva)
            print(f"[probe {name} epoch {ep + 1}] acc@1 {acc[0]:.3f} acc@4 {acc[3]:.3f} lead {lead:.3f}", flush=True)
            if best is None or lead > best[1]:
                best = (acc, lead)
        res[name] = best
        del probe, popt
        torch.cuda.empty_cache()
    prow = []
    for k in range(args.K):
        a, b, o = (float(res[n][0][k]) for n in ("A_base", "B_slots", "O_oracle"))
        prow.append({"offset": k + 1, "A_base": a, "B_slots": b, "O_oracle": o,
                     "gap_closed": (b - a) / (o - a) if o - a > 1e-3 else float("nan")})
    prow.append({"offset": "lead", **{n: res[n][1] for n in res},
                 "gap_closed": (res["B_slots"][1] - res["A_base"][1]) / max(res["O_oracle"][1] - res["A_base"][1], 1e-6)})
    print_table(prow, ["offset", "A_base", "B_slots", "O_oracle", "gap_closed"],
                "probe top-1 for x_(s+k), held-out: base / + trained lookahead slots / + oracle states")
    save_json(args.out, {"args": vars(args), "slots": rows, "probe": prow})


if __name__ == "__main__":
    main()
