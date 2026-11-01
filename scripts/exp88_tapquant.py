"""Experiment 88: tap-aware quantization. Learn the quantization parameters of a low-bit target copy so that what the
DFlash drafter sees, c = hidden_norm(fc([L1, L9, L17, L25, L33])), matches the target's, instead of a generic objective.

Why: the drafter needs the target's full depth but (we expect) not its full precision; only the taps it reads matter
(exp71: L25 / L33, threshold ~ cosine 0.95). The LoRA copy (exp55) cannot be merged into a low-bit format; here only
the quantizer is learned, so the result IS a low-bit model (deployable with real low-bit kernels).

Quantizer (per decoder Linear, groups of 64 input weights, asymmetric, STE through the rounding; LSQ+ style):
  W_q = clamp(round((W - lo) / s), 0, 2^b - 1) * s + lo,   s = s0 * exp(a),  lo = lo0 + b_off * s0
  initialised at round-to-nearest (a = b_off = 0); only a, b_off are trained. Embeddings and lm_head stay bf16.
  --hi-layers / --hi-bits give chosen layers more bits (mixed precision).
Objectives (weights on the command line):
  --w-feat   normalised MSE between the copy's and the target's fused drafter input c, at every position (tap-aware)
  --w-agree  cross-entropy of the copy's next-token logits on the target's greedy text (LM objective, the usual QAT)
Data: the lag training shards; same split rule as exp55 --shards (seed-0 shuffle, first --n-val = validation).

Outputs: validation fused-feature cosine and normalised MSE, copy-target next-token agreement; the trained copy is
saved as a bf16 HF checkpoint holding the quantized values (load with hf:<dir> in exp27 / exp55 --epochs 0).

  python scripts/exp88_tapquant.py --shards /scratch-shared/$USER/hspec_lagft/data --out $HOME/hspec_ckpt/tapquant_feat
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

from dflash.model import extract_context_feature  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402


class QLinear(nn.Module):
    """Fake-quantized Linear with learnable per-group step size and offset (STE rounding)."""

    def __init__(self, lin: nn.Linear, bits: int, group: int = 64):
        super().__init__()
        w = lin.weight.data
        out_f, in_f = w.shape
        self.out_f, self.in_f, self.group, self.qmax = out_f, in_f, group, 2 ** bits - 1
        self.register_buffer("w", w)                                         # bf16, frozen
        self.bias = lin.bias
        g = w.float().reshape(out_f, in_f // group, group)
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        self.register_buffer("s0", ((hi - lo).clamp_min(1e-8) / self.qmax))
        self.register_buffer("lo0", lo)
        self.a = nn.Parameter(torch.zeros_like(lo))
        self.b = nn.Parameter(torch.zeros_like(lo))

    def qweight(self):
        g = self.w.float().reshape(self.out_f, self.in_f // self.group, self.group)
        s = self.s0 * torch.exp(self.a)
        lo = self.lo0 + self.b * self.s0
        x = (g - lo) / s
        q = (x.round() - x).detach() + x                                     # STE
        q = q.clamp(0, self.qmax)
        return (q * s + lo).reshape(self.out_f, self.in_f).to(self.w.dtype)

    def forward(self, x):
        return F.linear(x, self.qweight(), self.bias)


def quantize_model_(model, bits, hi_layers, hi_bits, group):
    n = 0
    for li, layer in enumerate(model.model.layers):
        b = hi_bits if li in hi_layers else bits
        for name, mod in list(layer.named_modules()):
            if isinstance(mod, nn.Linear):
                parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
                setattr(parent, name.rsplit(".", 1)[-1], QLinear(mod, b, group))
                n += 1
    return n


@torch.no_grad()
def materialize_(model):
    """Replace every QLinear by a plain Linear holding the quantized values (for saving / loading as hf:)."""
    for layer in model.model.layers:
        for name, mod in list(layer.named_modules()):
            if isinstance(mod, QLinear):
                lin = nn.Linear(mod.in_f, mod.out_f, bias=mod.bias is not None, device=mod.w.device,
                                dtype=mod.w.dtype)
                lin.weight.copy_(mod.qweight())
                if mod.bias is not None:
                    lin.bias.copy_(mod.bias)
                parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
                setattr(parent, name.rsplit(".", 1)[-1], lin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--shards", required=True)
    ap.add_argument("--n-train", type=int, default=2400)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--hi-layers", type=int, nargs="*", default=[])
    ap.add_argument("--hi-bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--w-feat", type=float, default=1.0)
    ap.add_argument("--w-agree", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--teacher", default="", help="cache file for the target's fused features (reused if present)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    draft = load_draft(args.draft, dev)
    draft.requires_grad_(False)
    lids = list(draft.target_layer_ids)

    def fused(hs):
        return draft.hidden_norm(draft.fc(extract_context_feature(hs, lids)))[0]          # [T, H]

    pool = []
    for f in sorted(Path(args.shards).expanduser().glob("shard*.pt")):
        pool += [d for d in torch.load(f) if len(d["ids"]) <= args.max_len]
    random.Random(0).shuffle(pool)
    val, train = pool[: args.n_val], pool[args.n_val : args.n_val + args.n_train]

    tpath = Path(args.teacher).expanduser() if args.teacher else None
    if tpath and tpath.exists():
        teach = torch.load(tpath)
        print(f"[teacher] loaded {tpath}", flush=True)
    else:
        target = load_target(args.target, dev)
        teach = {}
        with torch.inference_mode():
            for split, data in (("val", val), ("train", train)):
                teach[split] = [fused(target(d["ids"].long().to(dev)[None], output_hidden_states=True,
                                             logits_to_keep=1).hidden_states).to(torch.bfloat16).cpu() for d in data]
        del target
        torch.cuda.empty_cache()
        if tpath:
            torch.save(teach, tpath)
        print(f"[teacher] fused features for {len(train)} train / {len(val)} val sequences, {time.time() - t0:.0f}s",
              flush=True)

    copy = load_target(args.target, dev)
    copy.requires_grad_(False)
    n = quantize_model_(copy, args.bits, set(args.hi_layers), args.hi_bits, args.group)
    params = [p for m in copy.modules() if isinstance(m, QLinear) for p in (m.a, m.b)]
    for p in params:
        p.requires_grad_(True)
    print(f"[copy] {n} linears at {args.bits} bit (layers {args.hi_layers} at {args.hi_bits}), "
          f"{sum(p.numel() for p in params) / 1e6:.0f}M quantizer params, objective feat {args.w_feat} "
          f"agree {args.w_agree}", flush=True)
    copy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    copy.config.use_cache = False

    def run(d, c_t, grad):
        ids = d["ids"].long().to(dev)
        with torch.set_grad_enabled(grad):
            out = copy(ids[None], output_hidden_states=True, use_cache=False)
            c = fused(out.hidden_states)[1:].float()                                    # skip the sink position
            ct = c_t[1:].to(dev).float()
            l_feat = ((c - ct) ** 2).sum(-1).mean() / (ct ** 2).sum(-1).mean()
            r = slice(d["n_prompt"] - 1, len(ids) - 1)
            logits = out.logits[0, r].float()
            l_agree = F.cross_entropy(logits, ids[r.start + 1 : r.stop + 1])
            st = {"cos": float(F.cosine_similarity(c, ct, dim=-1).mean()),
                  "agree": float((logits.argmax(-1) == ids[r.start + 1 : r.stop + 1]).float().mean())}
        return l_feat, l_agree, st

    def evaluate(step):
        copy.eval()
        tot = {"l_feat": 0.0, "cos": 0.0, "agree": 0.0}
        for d, ct in zip(val, teach["val"]):
            lf, _, st = run(d, ct, False)
            tot["l_feat"] += float(lf)
            tot["cos"] += st["cos"]
            tot["agree"] += st["agree"]
        copy.train()
        k = len(val)
        print(f"[eval step {step}] fused-feature cosine {tot['cos'] / k:.4f}  normalised mse {tot['l_feat'] / k:.4f}  "
              f"copy-target next-token agreement {tot['agree'] / k:.4f}", flush=True)

    evaluate(0)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    copy.train()
    step, rng = 0, random.Random(1)
    for _ in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for j in order:
            step += 1
            lf, la, _ = run(train[j], teach["train"][j], True)
            loss = args.w_feat * lf + args.w_agree * la
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 50 == 0:
                print(f"[train {step}/{total}] feat {float(lf):.4f} agree {float(la):.3f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 and step < total:
                evaluate(step)
    evaluate(total)
    if args.out:
        materialize_(copy)
        copy.config.use_cache = True
        copy.save_pretrained(args.out)
        print(f"[save] quantized copy (bf16 storage) -> {args.out}", flush=True)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
