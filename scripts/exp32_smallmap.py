"""Experiment 32: can a small model of the same family predict the verifier's states on the frontier?

Proposal 1, first test. A small Qwen3 (0.6B / 1.7B, same tokenizer) runs over the sequence; a
closed-form linear map (ridge regression, as in cross-model KV transfer) turns its hidden states
at a few layers into the 8B target's DFlash features (target layers 1, 9, 17, 25, 33). The drafter
reads these predicted features for the last g drafted positions (depth-lagged blocks, as in
train_shallow) while everything before stays real. No drafter training here (original drafter).

  fit       ridge W on response tokens of training sequences: X = [small layers sel] (+ bias),
            Y = target feature vector; lambda picked on held-out fit tokens by cosine
  evaluate  on the same 48 validation sequences / blocks as train_shallow: tau at lags 4 / 8 / 16
            with real features vs mapped features, cosine per feature slice, and the small model's
            own greedy-token agreement with the target (the correction signal)

Reference points (lag 16, original drafter unless noted): real ~5.9; 3-bit copy 5.30 (5.73 after
drafter training); target's first 18 layers + adapter 5.23 (5.29 joint); first 10 layers ~4.0-4.4.

  python scripts/exp32_smallmap.py --small Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.fusion import accepted  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.shallow import drafter_block_loss  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def small_layers(n_layers: int, n_sel: int) -> list[int]:
    """hidden_states indices (1 .. n_layers) spread evenly, last layer included."""
    return sorted({round(1 + i * (n_layers - 1) / max(n_sel - 1, 1)) for i in range(n_sel)})


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--small", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-sel", type=int, default=6, help="small-model layers used as regressors")
    ap.add_argument("--n-fit", type=int, default=400, help="training sequences for the ridge fit")
    ap.add_argument("--alphas", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    tag = args.small.split("/")[-1]
    t0 = time.time()

    def log(m):
        print(f"[{tag} {time.time() - t0:6.0f}s] {m}", flush=True)

    # same split as train_shallow (seed 0 shuffle, first val_seqs = validation)
    rng = random.Random(args.seed)
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
    data = [d for d in data if len(d["ids"]) <= args.max_len]
    rng.shuffle(data)
    val, train = data[: args.val_seqs], data[args.val_seqs :]
    fit = train[: args.n_fit]

    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    small = load_target(args.small, dev)
    lids = list(draft.target_layer_ids)
    sel = small_layers(small.config.num_hidden_layers, args.n_sel)
    bs = draft.block_size
    p_small = sum(p.numel() for p in small.parameters()) / 1e9
    p_target = sum(p.numel() for p in target.parameters()) / 1e9
    log(f"small model {p_small:.2f}B params ({p_small / p_target:.3f} of the target), regressor layers {sel}")

    def passes(ids):
        to = target(ids[None], output_hidden_states=True)
        real = extract_context_feature(to.hidden_states, lids)[0].double()
        so = small(ids[None], output_hidden_states=True)
        x = torch.cat([so.hidden_states[i][0] for i in sel], -1).double()
        x = torch.cat([x, torch.ones_like(x[:, :1])], -1)
        return real, x, to.logits[0].argmax(-1), so.logits[0].argmax(-1)

    # ---- fit: accumulate normal equations; hold out every 10th sequence for lambda selection
    Dx = small.config.hidden_size * len(sel) + 1
    Dy = len(lids) * target.config.hidden_size
    XtX = torch.zeros(Dx, Dx, dtype=torch.float64, device=dev)
    XtY = torch.zeros(Dx, Dy, dtype=torch.float64, device=dev)
    hold = []
    n_tok = 0
    for i, d in enumerate(fit):
        ids = d["ids"].long().to(dev)
        real, x, _, _ = passes(ids)
        r = slice(max(d["n_prompt"], 4), len(ids))
        if i % 10 == 0 and len(hold) < 20:
            hold.append((x[r].clone(), real[r].clone()))
            continue
        XtX += x[r].T @ x[r]
        XtY += x[r].T @ real[r]
        n_tok += r.stop - r.start
        if (i + 1) % 100 == 0:
            log(f"fit: {i + 1}/{len(fit)} sequences, {n_tok} tokens")
    scale = torch.trace(XtX) / Dx
    best = None
    for a in args.alphas:
        W = torch.linalg.solve(XtX + a * scale * torch.eye(Dx, dtype=XtX.dtype, device=dev), XtY)
        cs = []
        for xh, yh in hold:
            p = xh @ W
            cs.append(torch.nn.functional.cosine_similarity(p, yh, dim=-1).mean().item())
        c = sum(cs) / len(cs)
        log(f"alpha {a:g}: held-out cosine {c:.4f}")
        if best is None or c > best[0]:
            best = (c, a, W)
    _, alpha, W = best
    W = W.float()
    del XtX, XtY
    log(f"chosen alpha {alpha:g}")

    # ---- evaluate on the validation blocks (same sampling as train_shallow.evaluate)
    H = target.config.hidden_size
    rows_out = []
    for lag in (4, 8, 16):
        vrng = random.Random(123)
        taus = {"real": [], "mapped": []}
        cos_slices = [[] for _ in lids]
        agree = []
        for d in val:
            ids = d["ids"].long().to(dev)
            T = len(ids)
            blocks = []
            for _ in range(8):
                lo, hi = max(d["n_prompt"], lag + 1), T - 2
                if hi > lo:
                    blocks.append((vrng.randrange(lo, hi), lag))
            if not blocks:
                continue
            real, x, tp, sp = passes(ids)
            mapped = (x.float() @ W)[None].to(torch.bfloat16)
            real_b = real[None].to(torch.bfloat16)
            rows = torch.tensor(sorted({t for s, g in blocks for t in range(s - g, s)}), device=dev)
            for j in range(len(lids)):
                sl = slice(j * H, (j + 1) * H)
                cos_slices[j].append(torch.nn.functional.cosine_similarity(
                    mapped[0, rows, sl].float(), real_b[0, rows, sl].float(), dim=-1).mean().item())
            agree.append((sp[rows] == tp[rows]).float().mean().item())
            for name, src in (("real", real_b), ("mapped", mapped)):
                _, correct = drafter_block_loss(draft, target, real_b, src, ids, blocks, bs)
                taus[name] += [accepted(c) for c in correct if None not in c]
        row = {"small": tag, "lag": lag, "tau_real": sum(taus["real"]) / len(taus["real"]),
               "tau_mapped": sum(taus["mapped"]) / len(taus["mapped"]), "agree_small": sum(agree) / len(agree)}
        for j, l in enumerate(lids):
            row[f"cos_L{l}"] = sum(cos_slices[j]) / len(cos_slices[j])
        rows_out.append(row)
        log(f"lag {lag}: tau real {row['tau_real']:.2f} mapped {row['tau_mapped']:.2f} agree {row['agree_small']:.3f}")
    print_table(rows_out, ["small", "lag", "tau_real", "tau_mapped", "agree_small"] + [f"cos_L{l}" for l in lids],
                f"small model -> target features (ridge, alpha {alpha:g}), original drafter; "
                f"small = {p_small / p_target:.3f} of the target's parameters")
    save_json(args.out or f"exp32_smallmap_{tag}", {"args": vars(args), "alpha": alpha, "layers": sel,
                                                    "rows": rows_out})


if __name__ == "__main__":
    main()
