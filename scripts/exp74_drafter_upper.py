"""Experiment 74: upper-layer completion with REALISTIC errors: DFlash's own states, mapped linearly to the target's
layer k, finished by the target's layers k..L-1 (offline, no drafter training, Colab-ready)

exp73 showed the target's upper layers absorb random errors in layer-k states. A drafter's errors are not random, so
here the layer-k states come from the drafter itself:

  1. DFlash drafts the block from anchor s with real context features (as in deployment). Its output state at mask
     i (i = 1..15) predicts x[s+i]; the target state that predicts x[s+i] sits at position s+i-1.
  2. A ridge map (fit on the train split, closed form) takes [drafter state at mask i ; target embedding of the token
     at position s+i-1 (the anchor for i = 1, else the drafter's own greedy guess for i-1)] to the target's layer-k
     state at position s+i-1.
  3. The mapped states for positions s .. s+14 run through the target's layers k..L-1 over the real prefix (causal,
     so later block positions see earlier ones), then norm + LM head.

Compared on held-out anchors, tau out of 16 and accuracy:
  dflash        DFlash's own tokens (the baseline to beat)
  map_final     the same ridge map straight to the target's last-layer state, then norm + head (no upper layers):
                separates "the upper layers help" from "the map alone helps"
  map_k         ridge to layer k, then the target's layers k..L-1   (k from --ks)
  cos_in        cosine of the mapped states to the real layer-k states (compare with exp73's noise table)

  python scripts/exp74_drafter_upper.py --cache /content/seqs.pt
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

from dflash.model import _output_head, extract_context_feature  # noqa: E402

from hspec.lagtrain import pack  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences, e.g. from exp69")
    ap.add_argument("--ks", nargs="+", type=int, default=[20, 28, 32])
    ap.add_argument("--fit-anchors", type=int, default=3000)
    ap.add_argument("--anchors", type=int, default=500, help="eval anchors")
    ap.add_argument("--lam", type=float, default=1e-2, help="ridge strength, relative to mean diagonal of X'X")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--save", default="exp74_drafter_upper")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    nv = int(len(seqs) * args.val_frac)
    val, train = seqs[:nv], seqs[nv:]
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    B = bs - 1
    layers = target.model.layers
    L = len(layers)
    ks = [k for k in args.ks if 0 < k < L]
    names = [*ks, "final"]
    emb = target.get_input_embeddings()
    head = _output_head(target)

    pre_norm = {}
    target.model.norm.register_forward_hook(lambda m, i, o: pre_norm.__setitem__("h", i[0]))

    def anchors_of(split, n, seed):
        cand = [(i, s) for i, (ids, lo) in enumerate(split) for s in range(lo + 4, len(ids) - bs - 1)]
        random.Random(seed).shuffle(cand)
        by = {}
        for i, s in sorted(cand[:n]):
            by.setdefault(i, []).append(s)
        return by

    @torch.inference_mode()
    def per_sequence(x, anchors):
        """Target pass (features, layer-k states, last pre-norm state) + one packed DFlash pass for all anchors.
        Returns dict with drafter inputs X [A*B, Din], targets per name [A*B, H], and per-anchor extras."""
        out = target(x[None], output_hidden_states=True, use_cache=False)
        feats = extract_context_feature(out.hidden_states, lids)
        hs = {k: out.hidden_states[k][0] for k in ks}
        hs["final"] = pre_norm["h"][0]
        del out
        hid_all, logit_all = [], []
        for c in range(0, len(anchors), 48):
            ch = anchors[c : c + 48]
            noise, pos, mask, starts = pack(draft, target, feats, x, [(s, 0) for s in ch], bs)
            hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                        past_key_values=None, use_cache=False)[0]
            hb = torch.stack([hid[st : st + B] for st in starts])                     # [a, B, Hd]
            hid_all.append(hb)
            logit_all.append(draft.compute_logits(hb, head))
        hd = torch.cat(hid_all)                                                       # [A, B, Hd]
        dtok = torch.cat(logit_all).argmax(-1)                                        # [A, B] drafter tokens
        prev = torch.cat([x[torch.tensor(anchors, device=dev)][:, None], dtok[:, :-1]], 1)   # token at s+i-1
        X = torch.cat([hd.float(), emb(prev).float()], -1)                           # [A, B, Hd+H]
        Y = {}
        for nme in names:
            Y[nme] = torch.stack([hs[nme][s : s + B] for s in anchors]).float()      # positions s .. s+B-1
        return X, Y, dtok, hs

    # ---------------- fit ridge maps on the train split ----------------
    fit = anchors_of(train, args.fit_anchors, 1)
    XtX, XtY, n_fit = None, {}, 0
    with torch.inference_mode():
        for n_done, (i, anchors) in enumerate(fit.items()):
            x = train[i][0].to(dev)
            X, Y, _, _ = per_sequence(x, anchors)
            X = torch.cat([X.reshape(-1, X.shape[-1]), torch.ones(X.shape[0] * X.shape[1], 1, device=dev)], 1).double()
            if XtX is None:
                XtX = torch.zeros(X.shape[1], X.shape[1], device=dev, dtype=torch.float64)
            XtX += X.T @ X
            for nme in names:
                yy = Y[nme].reshape(-1, Y[nme].shape[-1]).double()
                XtY[nme] = XtY.get(nme, 0) + X.T @ yy
            n_fit += X.shape[0]
            if n_done % 50 == 0:
                print(f"[fit] {n_done}/{len(fit)} sequences, {n_fit} positions, {time.time()-t0:.0f}s", flush=True)
        lam = args.lam * float(XtX.diagonal().mean())
        XtX += lam * torch.eye(XtX.shape[0], device=dev, dtype=torch.float64)
        W = {nme: torch.linalg.solve(XtX, XtY[nme]).float() for nme in names}
    del XtX, XtY
    print(f"[fit] ridge maps for {names} on {n_fit} positions, lambda {lam:.3g}, {time.time()-t0:.0f}s", flush=True)

    def run_upper(h, k):
        pos = torch.arange(h.shape[1], device=dev)[None]
        pe = target.model.rotary_emb(h, pos)
        for layer in layers[k:]:
            o = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe, use_cache=False,
                      cache_position=pos[0])
            h = o[0] if isinstance(o, tuple) else o
        return h

    # ---------------- evaluate on held-out anchors ----------------
    ev = anchors_of(val, args.anchors, 0)
    st = {nme: {"tau": 0.0, "acc": 0.0, "acc4": 0.0, "cos_in": 0.0} for nme in ["dflash", *[f"map_{n}" for n in names]]}
    n_eval = 0
    with torch.inference_mode():
        for n_done, (i, anchors) in enumerate(ev.items()):
            x = val[i][0].to(dev)
            X, Y, dtok, hs = per_sequence(x, anchors)
            Xb = torch.cat([X, torch.ones(*X.shape[:2], 1, device=dev)], -1)
            for a, s in enumerate(anchors):
                truth = x[s + 1 : s + B + 1]
                preds = {"dflash": dtok[a]}
                cos_in = {}
                for nme in names:
                    hm = Xb[a] @ W[nme]                                               # [B, H] mapped states
                    cos_in[nme] = float(F.cosine_similarity(hm, Y[nme][a], dim=-1).mean())
                    if nme == "final":
                        hout = hm.to(torch.bfloat16)
                    else:
                        h = torch.cat([hs[nme][None, :s], hm.to(hs[nme].dtype)[None]], 1)
                        hout = run_upper(h, nme)[0, s : s + B]
                    preds[f"map_{nme}"] = target.lm_head(target.model.norm(hout)).argmax(-1)
                for key, pr in preds.items():
                    right = (pr == truth).float()
                    st[key]["tau"] += matched(pr.tolist(), truth.tolist()) + 1
                    st[key]["acc"] += float(right.mean())
                    st[key]["acc4"] += float(right[:4].mean())
                    if key.startswith("map_"):
                        st[key]["cos_in"] += cos_in[key[4:]]
                n_eval += 1
            if n_done % 20 == 0:
                print(f"[eval] {n_done}/{len(ev)} sequences, {time.time()-t0:.0f}s", flush=True)

    rows = []
    for key, v in st.items():
        nm = key[4:] if key.startswith("map_") else None
        rows.append({"method": key, "upper_layers": 0 if nm in (None, "final") else L - int(nm),
                     "cos_in": v["cos_in"] / n_eval if nm else float("nan"),
                     "acc": v["acc"] / n_eval, "acc@1-4": v["acc4"] / n_eval, "tau": v["tau"] / n_eval})
    print_table(rows, list(rows[0]), f"DFlash vs its own states mapped (ridge) to target layer k and finished by the "
                                     f"target's layers k..{L-1}; tau out of {B + 1}, {n_eval} held-out anchors")
    save_json(args.save, {"args": vars(args), "rows": rows, "n_fit_positions": n_fit, "n_eval": n_eval})
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
