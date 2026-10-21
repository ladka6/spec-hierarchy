"""Experiment 65: how cheap can a KV refresh be? (offline, T=0, closed-form fits, no training)

The DFlash drafter reads, for every context position, the target's 5 tap features (layers 1, 9, 17, 25, 33). A refresh
has to supply them for drafted positions. Two near-free ingredients exist:
  - the drafter's own final hidden state at the row that predicts token p+1 is trained (shared LM head) to look like
    the target's final hidden state at p, so it already estimates the DEEP taps of position p;
  - the shallow tap (layer 1) is the output of the target's first two layers: exact and cheap to compute.
Sources of features for the refreshed positions:
  oracle  the target's real features                                 (upper bound; what verification would give)
  copy    the 4-bit copy over the drafted tokens                      (what we have, 3-6 ms)
  tier0   ridge map [drafter final hidden, token embedding] -> 5 taps  (one matmul)
  tier1   tier0 inputs + the exact layer-1 tap -> the other 4 taps     (+ the target's first 2 layers)
  none    no features: the chunk enters as plain tokens (frozen drafter in lag mode, lower bound)
M1  cosine of each source to the real features, per tap, on held-out positions.
M2  chunk-and-restart: keep the first n drafted tokens (chunk), refresh positions s .. s+n-1 with a source, draft a
    new block from s+n. Accepted length of chunk + continuation (block up to n+15 tokens) vs block 1.
Ridge maps are fitted on --n-fit training sequences (teacher.pt), evaluated on the held-out ones.

  python scripts/exp65_cheaprefresh.py --teacher .../teacher.pt --copy bnb4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from hspec.lagtrain import draft_logits_lagged, pack  # noqa: E402
from hspec.models import load_draft, load_mid, load_target  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

SOURCES = ("oracle", "copy", "tier0", "tier1", "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--n-fit", type=int, default=600, help="training sequences for the ridge fits")
    ap.add_argument("--n-val", type=int, default=0, help="0: all held-out sequences")
    ap.add_argument("--chunks", nargs="+", type=int, default=[4, 8, 12])
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--save", default="exp65_cheaprefresh")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tdata = torch.load(args.teacher)
    fit, val = tdata["train"][: args.n_fit], tdata["val"]
    if args.n_val:
        val = val[: args.n_val]
    target = load_target(args.target, dev)
    copy = load_mid(args.copy, dev)
    draft = load_draft(args.draft, dev)
    for m in (target, copy, draft):
        m.requires_grad_(False)
    bs, lids, D = draft.block_size, list(draft.target_layer_ids), draft.block_size - 1
    H = target.config.hidden_size
    nT = len(lids)
    emb = target.get_input_embeddings()

    @torch.inference_mode()
    def first_pass(feats, x, anchors):
        """Drafter block 1 at every anchor: top-1 tokens [A, 15] and final hidden states [A, 15, H] (normed)."""
        noise, pos, mask, starts = pack(draft, target, feats, x, [(s, 0) for s in anchors], bs)
        hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                    past_key_values=None, use_cache=False)[0]
        rows = torch.stack([hid[st : st + D] for st in starts])                       # [A, 15, H]
        y = draft.compute_logits(rows, target.lm_head).argmax(-1)
        return y, rows

    @torch.inference_mode()
    def seq_data(d):
        x = d["ids"].long().to(dev)
        tout = target(x[None], output_hidden_states=True, logits_to_keep=1, use_cache=True)
        feats = extract_context_feature(tout.hidden_states, lids)                      # [1, T, 5H]
        cache = tout.past_key_values
        del tout
        anchors = list(d["anchors"])
        y, rows = first_pass(feats, x, anchors)
        return x, feats, cache, anchors, y, rows

    def inputs(rows_k, toks, shallow=None):
        """Regression inputs for positions s .. s+14: drafter row i (predicting s+i+1), embedding of token at s+i."""
        parts = [rows_k.float(), emb(toks).float()]
        if shallow is not None:
            parts.append(shallow.float())
        X = torch.cat(parts, -1)
        return torch.cat([X, torch.ones_like(X[:, :1])], -1)

    # ---------------------------------------------------------------- fit tier0 / tier1 (ridge, closed form)
    t0 = time.time()
    dims = {"tier0": 2 * H + 1, "tier1": 3 * H + 1}
    acc = {k: (torch.zeros(n, n, device=dev, dtype=torch.float32),
               torch.zeros(n, nT * H, device=dev, dtype=torch.float32)) for k, n in dims.items()}
    with torch.inference_mode():
        for d in fit:
            x, feats, _, anchors, y, rows = seq_data(d)
            for k, s in enumerate(anchors):
                toks = torch.cat([x[s : s + 1], y[k, : D - 1]])                       # tokens at s .. s+14
                Y = feats[0, s : s + D].float()                                        # real taps at s .. s+14
                for name in dims:
                    X = inputs(rows[k], toks, Y[:, :H] if name == "tier1" else None)
                    acc[name][0].add_(X.T @ X)
                    acc[name][1].add_(X.T @ Y)
    Wmap = {}
    for name, (xtx, xty) in acc.items():
        xtx64 = xtx.double()
        lam = args.ridge * xtx64.diagonal().mean()
        Wmap[name] = torch.linalg.solve(xtx64 + lam * torch.eye(xtx64.shape[0], device=dev, dtype=torch.float64),
                                        xty.double()).float()
    del acc
    print(f"[fit] ridge maps from {len(fit)} sequences in {time.time() - t0:.0f}s", flush=True)

    def predict(name, rows_k, toks, real):
        X = inputs(rows_k, toks, real[:, :H] if name == "tier1" else None)
        out = X @ Wmap[name]
        if name == "tier1":
            out[:, :H] = real[:, :H].float()                                            # exact shallow tap
        return out

    # ---------------------------------------------------------------- evaluate
    cos = defaultdict(lambda: torch.zeros(nT))
    ncos = 0
    tau = defaultdict(list)
    with torch.inference_mode():
        for d in val:
            x, feats, tcache, anchors, y, rows = seq_data(d)
            for k, s in enumerate(anchors):
                truth = x[s + 1 :].tolist()
                yk = y[k]
                toks = torch.cat([x[s : s + 1], yk[: D - 1]])
                real = feats[0, s : s + D].float()
                z = torch.cat([x[: s + 1], yk])
                c = DynamicCache(ddp_cache_data=[(l.keys[:, :, :s], l.values[:, :, :s]) for l in tcache.layers])
                po = copy(z[None, s:], past_key_values=c, use_cache=True, output_hidden_states=True,
                          logits_to_keep=1)
                src = {"oracle": real, "copy": extract_context_feature(po.hidden_states, lids)[0, :D].float(),
                       "tier0": predict("tier0", rows[k], toks, real), "tier1": predict("tier1", rows[k], toks, real)}
                del po, c
                # M1: cosine per tap on positions whose drafted prefix is still correct (where features matter)
                ok = matched(yk.tolist(), truth)                                       # positions s .. s+ok valid
                n_ok = min(ok + 1, D)
                for name in ("copy", "tier0", "tier1"):
                    a = src[name][:n_ok].view(n_ok, nT, H)
                    b = real[:n_ok].view(n_ok, nT, H)
                    cos[name] += F.cosine_similarity(a, b, dim=-1).sum(0).cpu()
                ncos += n_ok
                tau["block1"].append(ok + 1)
                # M2: chunk-and-restart
                for n in args.chunks:
                    ids = torch.cat([x[: s + 1], yk[:n]])                               # anchor of restart: s+n
                    for name in SOURCES:
                        if name == "none":
                            L = draft_logits_lagged(draft, target, feats, ids, s + n, n, bs)
                        else:
                            Fz = torch.cat([feats[:, :s], src[name][:n][None].to(feats.dtype)], 1)
                            L = blocks_logits(draft, target, Fz, ids, [(s + n, 0)], bs)[0]
                        chain = yk[:n].tolist() + L.argmax(-1).tolist()
                        a = matched(chain, truth) + 1
                        tau[f"{name}|{n}"].append(a)
                        tau[f"{name}|{n}|best"].append(max(a, ok + 1))
            print(f"[val] {len(tau['block1'])} anchors", flush=True)

    crow = [{"source": n, **{f"cos_L{lids[i]}": float(cos[n][i] / ncos) for i in range(nT)}}
            for n in ("copy", "tier0", "tier1")]
    print_table(crow, list(crow[0]), "M1: cosine to the target's real features, per tap layer (held-out positions "
                                     "with a correct drafted prefix); tier1's layer-1 tap is exact by construction")
    rows_out = []
    for n in args.chunks:
        for name in SOURCES:
            rows_out.append({"chunk": n, "source": name, "tau": mean(tau[f"{name}|{n}"]),
                             "tau_best_of_two": mean(tau[f"{name}|{n}|best"]), "block1": mean(tau["block1"])})
    print_table(rows_out, list(rows_out[0]), "M2: chunk of n drafted tokens + restart from its end with refreshed "
                                             "features (block up to n+15); best_of_two = max with block 1")
    save_json(args.save, {"args": vars(args), "cos": crow, "restart": rows_out})


if __name__ == "__main__":
    main()
