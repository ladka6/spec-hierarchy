"""Experiment 78: a mixed-precision "feature copy". Which layers can be 2-bit, which must stay precise, so that the two
taps the drafter really needs (L25, L33) stay at real-target quality? (offline, simulated quantization, Colab-ready)

Facts this builds on:
  exp71  DFlash reads the outputs of layers 1, 9, 17, 25, 33; only L25 and L33 matter (threshold ~ cosine 0.95)
  exp73  the target's upper layers shrink errors in their inputs
So a copy only has to be accurate at L25 / L33 (layers 34-35 only for its own token guesses), and errors made in the
lower layers may be corrected by more precise upper layers. At batch 1 time ~ bytes of weights read, so the cost of a
configuration is its weight bytes (printed relative to the 4-bit copy).

Simulated round-to-nearest quantization (group 64) per layer, applied in place to the bf16 target and restored after
each configuration (2-bit RTN is crude: real 2-bit methods such as HQQ / GPTQ would do better).

  part 1  sensitivity: one block of 4 layers at 2-bit, everything else bf16 -> cosine at L25 / L33
  part 2  allocations (lower layers / upper layers 25-35 at given bits): cosine at L25 / L33, greedy agreement with the
          target, and tau when the configuration's features replace the last m = 15 context positions (exp71 protocol:
          real 8.44, removed 1.60, 4-bit copy 8.30)

  python scripts/exp78_mixedprec.py --cache /content/seqs.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

# (name, bits for layers 0-24, bits for layers 25-35); 16 = bf16
ALLOCS = [
    ("all 16 (real target)", 16, 16),
    ("all 4 (= 4-bit copy)", 4, 4),
    ("all 3", 3, 3),
    ("all 2", 2, 2),
    ("lower 2 / upper 16", 2, 16),
    ("lower 2 / upper 8", 2, 8),
    ("lower 2 / upper 4", 2, 4),
    ("lower 3 / upper 4", 3, 4),
    ("lower 3 / upper 8", 3, 8),
    ("lower 4 / upper 8", 4, 8),
]


@torch.no_grad()
def quantize_layer_(layer, bits, group=64):
    if bits >= 16:
        return
    qmax = 2 ** bits - 1
    for mod in layer.modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight
            out_f, in_f = w.shape
            g = w.float().reshape(out_f, in_f // group, group)
            lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
            scale = (hi - lo).clamp_min(1e-8) / qmax
            q = ((g - lo) / scale).round().clamp(0, qmax)
            w.copy_((q * scale + lo).reshape(out_f, in_f).to(w.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", type=int, default=25, help="first 'upper' layer")
    ap.add_argument("--m", type=int, default=15)
    ap.add_argument("--anchors", type=int, default=400)
    ap.add_argument("--sens-seqs", type=int, default=20, help="sequences for the sensitivity scan")
    ap.add_argument("--save", default="exp78_mixedprec")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    val = seqs[: int(len(seqs) * 0.2)]
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    layers = target.model.layers
    L, H, nT = len(layers), target.config.hidden_size, len(lids)
    deep = [nT - 2, nT - 1]                                                  # tap indices of L25, L33
    orig = [{k: v.detach().to("cpu", copy=True) for k, v in layer.state_dict().items()} for layer in layers]
    params_per_layer = sum(p.numel() for p in layers[0].parameters())

    def restore():
        for layer, sd in zip(layers, orig):
            layer.load_state_dict({k: v.to(dev) for k, v in sd.items()})

    def apply(bits_per_layer):
        restore()
        for layer, b in zip(layers, bits_per_layer):
            quantize_layer_(layer, b)

    def cost(bits_per_layer, upto):
        """weight bytes of layers 0..upto relative to a 4-bit copy of layers 0..35"""
        return sum(min(b, 16) for b in bits_per_layer[: upto + 1]) / (4 * L)

    # fixed reference: real features, tokens
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + args.m + 8, len(ids) - bs)]
    random.Random(0).shuffle(cand)
    by_seq = {}
    for i, s in sorted(cand[: args.anchors]):
        by_seq.setdefault(i, []).append(s)
    real = {}
    restore()
    with torch.inference_mode():
        for i in by_seq:
            x = val[i][0].to(dev)
            out = target(x[None], output_hidden_states=True, use_cache=False)
            real[i] = (extract_context_feature(out.hidden_states, lids), out.logits[0].argmax(-1))

    def run_config(bits_per_layer, seq_ids, with_tau):
        apply(bits_per_layer)
        cos = torch.zeros(nT, device=dev)
        n_pos, agree, n_tok, taus = 0, 0.0, 0, []
        with torch.inference_mode():
            for i in seq_ids:
                ids, lo = val[i]
                x = ids.to(dev)
                out = target(x[None], output_hidden_states=True, use_cache=False)
                f = extract_context_feature(out.hidden_states, lids)
                rf, rtok = real[i]
                a = F.cosine_similarity(f[0, lo:].float().view(-1, nT, H), rf[0, lo:].float().view(-1, nT, H), dim=-1)
                cos += a.sum(0)
                n_pos += a.shape[0]
                agree += float((out.logits[0, lo - 1 : -1].argmax(-1) == rtok[lo - 1 : -1]).float().sum())
                n_tok += len(x) - lo
                if with_tau:
                    for s in by_seq[i]:
                        fz = rf[:, :s].clone()
                        fz[:, s - args.m : s] = f[:, s - args.m : s].to(fz.dtype)
                        Lg = blocks_logits(draft, target, fz, x, [(s, 0)], bs)[0]   # embeddings / LM head are never quantized
                        taus.append(matched(Lg.argmax(-1).tolist(), x[s + 1 :].tolist()) + 1)
        c = (cos / max(n_pos, 1)).tolist()
        return {"cos_L25": c[deep[0]], "cos_L33": c[deep[1]], "agree": agree / max(n_tok, 1),
                "tau": sum(taus) / len(taus) if taus else float("nan")}

    # ---------------- part 1: sensitivity of L25 / L33 to 2-bit blocks of 4 layers ----------------
    sens_ids = list(by_seq)[: args.sens_seqs]
    srows = []
    for start in range(0, L, 4):
        bits = [16] * L
        for j in range(start, min(start + 4, L)):
            bits[j] = 2
        r = run_config(bits, sens_ids, False)
        srows.append({"2-bit layers": f"{start}-{min(start + 3, L - 1)}", "cos_L25": r["cos_L25"],
                      "cos_L33": r["cos_L33"], "agree": r["agree"]})
        print(f"[sens] {srows[-1]}  {time.time()-t0:.0f}s", flush=True)
    print_table(srows, list(srows[0]), "part 1: one block of 4 layers at 2-bit (RTN), everything else bf16")

    # ---------------- part 2: allocations ----------------
    rows = []
    for name, lo_b, up_b in ALLOCS:
        bits = [lo_b if j < args.split else up_b for j in range(L)]
        r = run_config(bits, list(by_seq), True)
        rows.append({"config": name, "bytes_vs_4bit_copy (feat, to L33)": cost(bits, lids[-1]),
                     "bytes_vs_4bit_copy (tokens, all)": cost(bits, L - 1), **r})
        print(f"[alloc] {rows[-1]}  {time.time()-t0:.0f}s", flush=True)
    restore()
    print_table(rows, list(rows[0]), f"part 2: lower layers 0-{args.split - 1} / upper {args.split}-{L - 1}; tau with "
                                     f"the configuration's features for the last {args.m} context positions "
                                     f"(real 8.44, removed 1.60, 4-bit copy 8.30 in exp71)")
    save_json(args.save, {"args": vars(args), "sensitivity": srows, "allocations": rows,
                          "params_per_layer": params_per_layer})
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
