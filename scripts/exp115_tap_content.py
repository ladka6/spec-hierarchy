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

--anchor: the forward readout with the ANCHOR given. The drafter is handed the token after its newest context
position, so what a state is worth to it is what it adds ON TOP of that token. Regressors = [state of layer l at p,
input embedding of token p + 1] (each row-normalised), offsets j >= 2; the row "anchor only" uses the embedding
alone, and "gain" = top-1 minus that row. If the last layer's lead at p + 2 is mostly "it knows token p + 1", the
lead should shrink or vanish here. Layers: the drafter's taps, half depth, the last.

All offsets of one run use the same rows (positions for which the farthest target exists), so the normal equations
of every offset come from exactly the rows of its targets.

--past: the same readout BACKWARDS. The drafter never sees the context tokens, only these features, so a feature
also has to say what is at the position and just before it. Offsets are then k = 0, 1, 2, ... tokens back: a ridge
map from the state of layer l at position p to the INPUT EMBEDDING of token p - k, read out as the nearest token
by cosine over the input embedding matrix (top-1 / top-5), plus the cosine to the true embedding. If the last layers
drop this while the layers before them still hold it, a tap at 3/4 depth is the one place that has both.

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
    ap.add_argument("--past", action="store_true", help="read tokens p - k (offsets = k) instead of p + j")
    ap.add_argument("--anchor", action="store_true", help="forward readout with the token p + 1 given")
    ap.add_argument("--offsets", nargs="+", type=int, default=None, help="default 1 2 3 4 8 12, with --past 0 1 2 4 8")
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--fit-seqs", type=int, default=350)
    ap.add_argument("--eval-seqs", type=int, default=120)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--min-resp", type=int, default=48)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--save", default="exp115_tap_content")
    args = ap.parse_args()
    dev, t0 = "cuda", time.time()
    assert not (args.past and args.anchor)
    J = sorted(args.offsets or ([0, 1, 2, 4, 8] if args.past else [2, 3, 4, 8] if args.anchor else [1, 2, 3, 4, 8, 12]))
    K = max(J) if args.past else 0                           # rows start K positions into the response

    recs = []
    for f in sorted(Path(args.data).expanduser().glob("shard*.pt")):
        recs += torch.load(f)
    seqs = [(d["ids"].long()[: args.max_len], int(d["n_prompt"])) for d in recs]
    seqs = [(ids, lo + K) for ids, lo in seqs if len(ids) - lo - K >= args.min_resp]
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
    En = None
    if args.past:                                            # unit-norm input embeddings, for the nearest token
        E = target.get_input_embeddings().weight
        nrm = torch.cat([E[i : i + 65536].float().norm(dim=-1) for i in range(0, E.shape[0], 65536)])
        En = E / nrm.clamp_min(1e-6)[:, None].to(E.dtype)
        del nrm

    def target_rows(hs, lo, T, j, n):
        """regression target for offset j over the first n rows (positions lo .. lo + n - 1)"""
        if args.past:
            return hs[0][0, lo - j : lo - j + n].float()     # embedding of token p - j
        return hs[L][0, lo + j - 1 : lo + j - 1 + n].float() # final state whose output is token p + j

    def n_rows_of(lo, T, j=None):
        """rows = positions lo .. lo + n - 1, the same for every offset"""
        return T - 1 - lo if args.past else T - max(J) - lo

    def states(ids):
        return target(ids[None].to(dev), output_hidden_states=True, use_cache=False).hidden_states

    hs = states(fit[0][0])
    L, H = len(hs) - 1, hs[0].shape[-1]
    layers = sorted(set([0, L] + taps + [round(f * L) for f in args.fracs]))
    if args.anchor:                                          # -1 = the anchor's embedding alone
        layers = [-1] + sorted(set(taps + [L // 2, L]))
    print(f"[setup] {args.target}: {L} layers, width {H}, taps (hidden_states index) {taps}, layers {layers}, "
          f"offsets {J}, fit {len(fit)} / eval {len(ev)} sequences", flush=True)
    del hs

    def unit(v):
        return v.float() * torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True) + 1e-6)

    def rows(hs, l, lo, n):
        """regressors of positions lo .. lo + n - 1 with a constant column. Plain: the state of layer l (weighted
        afterwards). --anchor: [row-normalised state, row-normalised embedding of token p + 1], l = -1: embedding only."""
        if not args.anchor:
            x = hs[l][0, lo : lo + n]
            return torch.cat([x, torch.ones_like(x[:, :1])], -1)
        e = unit(hs[0][0, lo + 1 : lo + 1 + n])
        parts = ([] if l < 0 else [unit(hs[l][0, lo : lo + n])]) + [e, torch.ones_like(e[:, :1])]
        return torch.cat(parts, -1)

    # ---- fit: ridge; plain mode weights the rows by 1 / rms(x) (attention-sink tokens have huge norms) ----
    xtx, xty = {}, {}
    n_rows = 0
    for i, (ids, lo) in enumerate(fit):
        hs = states(ids)
        T = len(ids)
        n = n_rows_of(lo, T)
        if n <= 0:
            continue
        for l in layers:
            xa = rows(hs, l, lo, n).double()
            if args.anchor:
                w = torch.ones_like(xa[:, :1])
            else:
                w = torch.rsqrt(xa[:, :H].pow(2).mean(-1, keepdim=True) + 1e-6)
            xa = xa * w
            if l not in xtx:
                xtx[l] = torch.zeros(xa.shape[1], xa.shape[1], dtype=torch.float64, device=dev)
                for j in J:
                    xty[(l, j)] = torch.zeros(xa.shape[1], H, dtype=torch.float32, device=dev)
            xtx[l] += xa.T @ xa
            xa32, w32 = xa.float(), w.float()
            for j in J:
                xty[(l, j)] += xa32.T @ (target_rows(hs, lo, T, j, n) * w32)
        n_rows += n
        del hs
        if i % 50 == 0:
            print(f"[fit] {i}/{len(fit)} sequences, {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
    W = {}
    for l in layers:
        eye = torch.eye(xtx[l].shape[0], dtype=torch.float64, device=dev)
        A = xtx[l] + args.ridge * xtx[l].diagonal().mean() * eye
        for j in J:
            W[(l, j)] = torch.linalg.solve(A, xty.pop((l, j)).double()).float()
        del xtx[l], eye, A
    print(f"[fit] maps solved, {time.time() - t0:.0f}s", flush=True)

    # ---- held-out readout ----
    st = {(l, j): [0.0, 0.0, 0.0, 0] for l in layers for j in J}
    for i, (ids, lo) in enumerate(ev):
        hs = states(ids)
        T = len(ids)
        x = ids.to(dev)
        n = n_rows_of(lo, T)
        if n <= 0:
            continue
        for l in layers:
            xa = rows(hs, l, lo, n).float()
            for j in J:
                pred = xa @ W[(l, j)]
                if args.past:
                    top = (F.normalize(pred, dim=-1).to(En.dtype) @ En.T).topk(5, -1).indices
                    tgt = x[lo - j : lo - j + n]
                else:
                    top = head(pred.to(hdt)).topk(5, -1).indices
                    tgt = x[lo + j : lo + j + n]
                s = st[(l, j)]
                s[0] += (top[:, 0] == tgt).sum().item()
                s[1] += (top == tgt[:, None]).any(1).sum().item()
                s[2] += F.cosine_similarity(pred, target_rows(hs, lo, T, j, n), dim=-1).sum().item()
                s[3] += n
        del hs
        if i % 40 == 0:
            print(f"[eval] {i}/{len(ev)} sequences, {time.time() - t0:.0f}s", flush=True)

    sign = "-" if args.past else "+"

    def table(k, title):
        out = []
        for l in layers:
            r = {"layer": "anchor only" if l < 0 else l, "depth": max(l, 0) / L, "tap": "tap" if l in taps else ""}
            for j in J:
                s = st[(l, j)]
                r[f"{sign}{j}"] = s[k] / max(s[3], 1)
            out.append(r)
        print_table(out, list(out[0]), title)
        return out

    what = "p - k" if args.past else "p + j, token p + 1 given" if args.anchor else "p + j"
    print(f"\n{args.target}: token {what} read from the state of layer l at position p "
          f"(held-out, {st[(layers[0], J[0])][3]} positions at {sign}{J[0]})")
    t1 = table(0, f"top-1 of the token at {what}")
    t5 = table(1, f"top-5 of the token at {what}")
    cs = table(2, "cosine to the true input embedding of that token" if args.past
               else "cosine to the true final state at p + j - 1")
    if args.anchor:
        base = {j: st[(-1, j)][0] / max(st[(-1, j)][3], 1) for j in J}
        gain = [{"layer": l, "depth": l / L, "tap": "tap" if l in taps else "",
                 **{f"+{j}": st[(l, j)][0] / max(st[(l, j)][3], 1) - base[j] for j in J}} for l in layers if l >= 0]
        print_table(gain, list(gain[0]), "gain in top-1 over the anchor alone (what the state adds to the known token)")
    best = {f"{sign}{j}": max(layers, key=lambda l: st[(l, j)][0] / max(st[(l, j)][3], 1)) for j in J}
    print("\nbest layer per offset (top-1): " + ", ".join(f"{k}: " + ("anchor only" if v < 0 else f"layer {v} ({v / L:.0%})") for k, v in best.items()))
    save_json(args.save, {"args": vars(args), "layers": layers, "taps": taps, "n_layers": L, "top1": t1, "top5": t5,
                          "cos": cs, "best": best})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
