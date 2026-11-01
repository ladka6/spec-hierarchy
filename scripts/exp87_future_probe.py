"""Experiment 87: which target layer holds the most information about tokens further ahead?

Question: DFlash reads the target's deep taps (L25, L33), which are tuned for the NEXT token. A block drafter also needs
tokens t+2 .. t+16. Is information about those further tokens better in the middle layers (Future Lens, Pal et al. 2023)?
If so, block drafters read the wrong layers for far positions, and partial target computation (first k layers) for
in-flight tokens is cheap AND informative.

Method (linear probes, closed form, nothing else trained):
  For a layer l (hs[l] = residual stream after l layers; hs[0] = token embeddings, hs[36] = final normed state) and an
  offset j, fit ridge regression  hs[l](t) -> hs[36](t+j-1)  (the final state that the target's lm_head turns into
  token t+j), then predict token t+j as argmax lm_head(prediction). j = 1 at l = 36 is the identity (target's own next
  token). Features are standardised with per-dimension mean / std from the first sequences; the ridge strength is chosen
  per (layer, offset) on a validation split and accuracy is reported on a separate test split.
  The data are the target's own greedy responses, so token t+j is what the target would emit there.

Output: top-1 accuracy per layer x offset, the best layer per offset, a heatmap.

  python scripts/exp87_future_probe.py --data /scratch-shared/$USER/hspec_lagft/data
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True, help="directory with shard*.pt ({'ids', 'n_prompt'} records)")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[0, 3, 6, 9, 12, 15, 18, 21, 24, 26, 27, 30, 33, 34, 36],
                    help="hs indices; DFlash's taps L25 / L33 are hs[26] / hs[34]")
    ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    ap.add_argument("--train-tokens", type=int, default=200_000)
    ap.add_argument("--eval-tokens", type=int, default=24_000, help="split half validation, half test")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--plot", default="exp87_future_probe.png")
    ap.add_argument("--save", default="exp87_future_probe")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    torch.manual_seed(0)
    rng = random.Random(0)

    data = []
    for f in sorted(Path(args.data).expanduser().glob("shard*.pt")):
        data += torch.load(f)
    rng.shuffle(data)
    target = load_target(args.target, dev)
    head = target.lm_head
    Ls, Js, J = args.layers, args.offsets, max(args.offsets)
    D = target.config.hidden_size

    def states(d):
        """standardisable features hs[l](t) for every probed layer, final states, tokens; t in the response."""
        ids = d["ids"][: args.max_len].long()
        lo, T = max(d["n_prompt"], 1), len(ids)
        if T - 1 - J <= lo:
            return None
        hs = target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        t = torch.arange(lo, T - J, device=dev)
        X = {l: hs[l][0, t].float() for l in Ls}
        Y = {j: hs[-1][0, t + j - 1].float() for j in Js}
        tok = {j: ids.to(dev)[t + j] for j in Js}
        return X, Y, tok

    # standardisation statistics from the first sequences
    mx = {l: torch.zeros(D, device=dev) for l in Ls}
    sx = {l: torch.zeros(D, device=dev) for l in Ls}
    my, sy, n0, i = torch.zeros(D, device=dev), torch.zeros(D, device=dev), 0, 0
    while n0 < 20_000:
        s = states(data[i])
        i += 1
        if s is None:
            continue
        X, Y, _ = s
        for l in Ls:
            mx[l] += X[l].sum(0)
            sx[l] += (X[l] ** 2).sum(0)
        my += Y[1].sum(0)
        sy += (Y[1] ** 2).sum(0)
        n0 += len(Y[1])
    for l in Ls:
        mx[l] /= n0
        sx[l] = (sx[l] / n0 - mx[l] ** 2).clamp_min(1e-6).sqrt()
    my /= n0
    sy = (sy / n0 - my ** 2).clamp_min(1e-6).sqrt()
    print(f"[stats] {n0} tokens, {time.time() - t0:.0f}s", flush=True)

    # sufficient statistics on the training split (standardised features)
    XtX = {l: torch.zeros(D, D, device=dev) for l in Ls}
    XtY = {(l, j): torch.zeros(D, D, device=dev) for l in Ls for j in Js}
    sumx = {l: torch.zeros(D, device=dev) for l in Ls}
    sumy = {j: torch.zeros(D, device=dev) for j in Js}
    n = 0
    while n < args.train_tokens:
        s = states(data[i])
        i += 1
        if s is None:
            continue
        X, Y, _ = s
        Xs = {l: (X[l] - mx[l]) / sx[l] for l in Ls}
        Ys = {j: (Y[j] - my) / sy for j in Js}
        for l in Ls:
            XtX[l] += Xs[l].T @ Xs[l]
            sumx[l] += Xs[l].sum(0)
            for j in Js:
                XtY[(l, j)] += Xs[l].T @ Ys[j]
        for j in Js:
            sumy[j] += Ys[j].sum(0)
        n += len(Ys[1])
        if i % 100 == 0:
            print(f"[train] {n} tokens from {i} sequences, {time.time() - t0:.0f}s", flush=True)

    # held-out evaluation features (CPU, bf16)
    EX = {l: [] for l in Ls}
    Etok = {j: [] for j in Js}
    ne = 0
    while ne < args.eval_tokens:
        s = states(data[i])
        i += 1
        if s is None:
            continue
        X, _, tok = s
        for l in Ls:
            EX[l].append(((X[l] - mx[l]) / sx[l]).to(torch.bfloat16).cpu())
        for j in Js:
            Etok[j].append(tok[j].cpu())
        ne += len(tok[1])
    EX = {l: torch.cat(v) for l, v in EX.items()}
    Etok = {j: torch.cat(v) for j, v in Etok.items()}
    half = ne // 2
    print(f"[data] train {n} tokens, eval {ne} tokens ({half} validation / {ne - half} test), "
          f"{time.time() - t0:.0f}s", flush=True)

    def accuracy(Xe, W, b, tok):
        hits = 0
        for a in range(0, len(Xe), 2048):
            pred = (Xe[a : a + 2048].to(dev).float() @ W + b) * sy + my
            logits = head(pred.to(torch.bfloat16))
            hits += int((logits.argmax(-1) == tok[a : a + 2048].to(dev)).sum())
        return hits / len(Xe)

    acc = {}
    if True:
        for l in Ls:
            mu = sumx[l] / n
            C = (XtX[l] / n - torch.outer(mu, mu)).double()
            ev, Q = torch.linalg.eigh(C)
            for j in Js:
                muy = sumy[j] / n
                Cxy = (XtY[(l, j)] / n - torch.outer(mu, muy)).double()
                QtC = Q.T @ Cxy
                best = (-1.0, None)
                for lam in args.lams:
                    W = (Q @ (QtC / (ev.clamp_min(0) + lam)[:, None])).float()
                    b = muy - mu @ W
                    v = accuracy(EX[l][:half], W, b, Etok[j][:half])
                    if v > best[0]:
                        best = (v, (W, b, lam))
                W, b, lam = best[1]
                acc[(l, j)] = {"val": best[0], "test": accuracy(EX[l][half:], W, b, Etok[j][half:]), "lam": lam}
            print(f"[probe] hs[{l}] done: " + " ".join(f"j{j} {acc[(l, j)]['test']:.3f}" for j in Js)
                  + f"  {time.time() - t0:.0f}s", flush=True)

    rows = []
    for l in Ls:
        row = {"layer (hs index)": f"{l}" + (" (DFlash L25)" if l == 26 else " (DFlash L33)" if l == 34 else "")}
        for j in Js:
            row[f"t+{j}"] = acc[(l, j)]["test"]
        rows.append(row)
    print_table(rows, list(rows[0]), "1. top-1 accuracy of token t+j from a linear probe on hs[l](t) (test split)")
    best = []
    for j in Js:
        lb = max(Ls, key=lambda l: acc[(l, j)]["test"])
        best.append({"offset": f"t+{j}", "best layer": lb, "best acc": acc[(lb, j)]["test"],
                     "acc at hs[34] (L33)": acc.get((34, j), {}).get("test"),
                     "acc at hs[36] (final)": acc.get((36, j), {}).get("test")})
    print_table(best, list(best[0]), "2. best layer per offset")
    save_json(args.save, {"args": vars(args), "acc": {f"{l},{j}": v for (l, j), v in acc.items()}, "best": best})

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
        M = torch.tensor([[acc[(l, j)]["test"] for j in Js] for l in Ls])
        im = a1.imshow(M, aspect="auto", cmap="viridis")
        a1.set_xticks(range(len(Js)), [f"t+{j}" for j in Js])
        a1.set_yticks(range(len(Ls)), [f"hs[{l}]" for l in Ls])
        a1.set_xlabel("predicted token")
        a1.set_ylabel("probed layer")
        a1.set_title("top-1 accuracy")
        fig.colorbar(im, ax=a1)
        for c, j in enumerate(Js):
            v = M[:, c] / M[:, c].max()
            a2.plot(Ls, v.tolist(), marker="o", label=f"t+{j}")
        for x in (26, 34):
            a2.axvline(x, color="grey", ls="--", lw=0.8)
        a2.set_xlabel("probed layer (hs index); dashed = DFlash L25 / L33")
        a2.set_ylabel("accuracy / best accuracy for that offset")
        a2.set_title("where the information about each future token peaks")
        a2.legend(fontsize=8)
        fig.savefig(args.plot, dpi=130, bbox_inches="tight")
        print(f"[plot] saved {args.plot}")
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
