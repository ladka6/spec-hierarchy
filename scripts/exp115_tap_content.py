"""Experiment 115: what is IN the target's hidden states at each depth, by how far ahead it reaches.

exp111 on three families (Qwen3-8B, LLaMA-3.1-8B, gemma4-12B, original z-lab drafters): for the most recent context
positions the drafter lives on the tap at about 3/4 of the depth (68-92% of tau from that tap alone); the deepest
tap alone gives 50% everywhere, the first quarter of the depth nothing. Why 3/4, and why not the last layer?

Hypothesis: the last layer is narrowed to the NEXT token, and that token is the one the drafter needs least (for
the newest context position p = s - 1 it is the anchor s, which the drafter is given). What a block drafter needs
from position p is the tokens p + 2, p + 3, ... ; that information should peak before the last layer.

Measurement (closed form, nothing trained by SGD): for a layer l and an offset j, a ridge map from the state of
layer l at position p to the target's FINAL state at position p + j - 1, i.e. the state whose lm_head output is
token p + j. Read out with the target's own lm_head:
  top-1 / top-5   the target's token at p + j is the first / among the first five of lm_head(map(h_l(p)))
  cos             cosine between the mapped state and the true final state at p + j - 1
j = 1 is the usual "which layer knows the next token" (tuned-lens style); j >= 2 is what the drafter cannot get
from the anchor. Layers: the embedding (0), the drafter's taps, a grid over the second half of the depth, the last.
Maps are fitted on --fit-seqs sequences and scored on held-out ones (target-greedy text, response positions only).

  python scripts/exp115_tap_content.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
      --data /scratch-shared/$USER/hspec_taps/qwen3_8b
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

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default="", help="only to mark its tap layers")
    ap.add_argument("--data", required=True, help="directory with shard*.pt (ids, n_prompt), target-greedy")
    ap.add_argument("--offsets", nargs="+", type=int, default=[1, 2, 3, 4, 8, 12])
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--fit-seqs", type=int, default=350)
    ap.add_argument("--eval-seqs", type=int, default=120)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--min-resp", type=int, default=48)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--save", default="exp115_tap_content")
    args = ap.parse_args()
    dev, t0 = "cuda", time.time()
    J = sorted(args.offsets)

    recs = []
    for f in sorted(Path(args.data).expanduser().glob("shard*.pt")):
        recs += torch.load(f)
    seqs = [(d["ids"].long()[: args.max_len], int(d["n_prompt"])) for d in recs]
    seqs = [(ids, lo) for ids, lo in seqs if len(ids) - lo >= args.min_resp]
    assert len(seqs) >= 30, f"only {len(seqs)} usable sequences in {args.data}"
    random.Random(0).shuffle(seqs)
    n_eval = min(args.eval_seqs, len(seqs) // 3)
    ev, fit = seqs[:n_eval], seqs[n_eval : n_eval + args.fit_seqs]

    taps = []
    if args.draft:
        d = load_draft(args.draft, dev)
        taps = [int(l) + 1 for l in d.target_layer_ids]      # hidden_states index of a tap
        del d
        torch.cuda.empty_cache()
    target = load_target(args.target, dev)
    head = target.get_output_embeddings()
    hdt = head.weight.dtype

    def states(ids):
        return target(ids[None].to(dev), output_hidden_states=True, use_cache=False).hidden_states

    hs = states(fit[0][0])
    L, H = len(hs) - 1, hs[0].shape[-1]
    layers = sorted(set([0, L] + taps + [round(f * L) for f in args.fracs]))
    print(f"[setup] {args.target}: {L} layers, width {H}, taps (hidden_states index) {taps}, layers {layers}, "
          f"offsets {J}, fit {len(fit)} / eval {len(ev)} sequences", flush=True)
    del hs

    def rows(h, lo, T):
        """states of positions lo .. T-2 with a constant column"""
        x = h[0, lo : T - 1]
        return torch.cat([x, torch.ones_like(x[:, :1])], -1)

    # ---- fit: weighted ridge, rows weighted by 1 / rms(x) (attention-sink tokens have huge norms) ----
    xtx = {l: torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev) for l in layers}
    xty = {(l, j): torch.zeros(H + 1, H, dtype=torch.float32, device=dev) for l in layers for j in J}
    n_rows = 0
    for i, (ids, lo) in enumerate(fit):
        hs = states(ids)
        T = len(ids)
        fin = hs[L][0].float()
        for l in layers:
            xa = rows(hs[l], lo, T).double()
            w = torch.rsqrt(xa[:, :H].pow(2).mean(-1, keepdim=True) + 1e-6)
            xa = xa * w
            xtx[l] += xa.T @ xa
            xa32, w32 = xa.float(), w.float()
            for j in J:
                n = T - j - lo                               # positions p = lo .. T-1-j
                if n > 0:
                    xty[(l, j)] += xa32[:n].T @ (fin[lo + j - 1 : T - 1] * w32[:n])
        n_rows += T - 1 - lo
        del hs
        if i % 50 == 0:
            print(f"[fit] {i}/{len(fit)} sequences, {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
    W = {}
    eye = torch.eye(H + 1, dtype=torch.float64, device=dev)
    for l in layers:
        A = xtx[l] + args.ridge * xtx[l].diagonal().mean() * eye
        for j in J:
            W[(l, j)] = torch.linalg.solve(A, xty.pop((l, j)).double()).float()
        del xtx[l]
    print(f"[fit] maps solved, {time.time() - t0:.0f}s", flush=True)

    # ---- held-out readout ----
    st = {(l, j): [0.0, 0.0, 0.0, 0] for l in layers for j in J}
    for i, (ids, lo) in enumerate(ev):
        hs = states(ids)
        T = len(ids)
        x = ids.to(dev)
        fin = hs[L][0].float()
        for l in layers:
            xa = rows(hs[l], lo, T).float()
            for j in J:
                n = T - j - lo
                if n <= 0:
                    continue
                pred = xa[:n] @ W[(l, j)]
                top = head(pred.to(hdt)).topk(5, -1).indices
                tgt = x[lo + j : lo + j + n]
                s = st[(l, j)]
                s[0] += (top[:, 0] == tgt).sum().item()
                s[1] += (top == tgt[:, None]).any(1).sum().item()
                s[2] += F.cosine_similarity(pred, fin[lo + j - 1 : T - 1], dim=-1).sum().item()
                s[3] += n
        del hs
        if i % 40 == 0:
            print(f"[eval] {i}/{len(ev)} sequences, {time.time() - t0:.0f}s", flush=True)

    def table(k, title):
        out = []
        for l in layers:
            r = {"layer": l, "depth": l / L, "tap": "tap" if l in taps else ""}
            for j in J:
                s = st[(l, j)]
                r[f"+{j}"] = s[k] / max(s[3], 1)
            out.append(r)
        print_table(out, list(out[0]), title)
        return out

    print(f"\n{args.target}: token p + j read from the state of layer l at position p "
          f"(held-out, {st[(layers[0], J[0])][3]} positions at +{J[0]})")
    t1 = table(0, "top-1 of the target's token at p + j")
    t5 = table(1, "top-5 of the target's token at p + j")
    cs = table(2, "cosine to the true final state at p + j - 1")
    best = {f"+{j}": max(layers, key=lambda l: st[(l, j)][0] / max(st[(l, j)][3], 1)) for j in J}
    print("\nbest layer per offset (top-1): " + ", ".join(f"{k}: layer {v} ({v / L:.0%})" for k, v in best.items()))
    save_json(args.save, {"args": vars(args), "layers": layers, "taps": taps, "n_layers": L, "top1": t1, "top5": t5,
                          "cos": cs, "best": best})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
