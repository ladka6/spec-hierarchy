"""Experiment 126: what is in the code the drafter asks for? (no training; reads the exp125 checkpoints)

exp125: an r-number code of an in-flight token's real feature, trained with the drafter on the drafter's loss,
closes half (r = 64) to three quarters (r = 256) of the gap between tokens-only and real features, while the PCA
projection of the same size is worse than nothing before training. So the drafter does not want the target's state;
it wants something much smaller. This script looks at what that is, three ways.

A. Content (closed-form readouts on held-out text). From a set of regressors, a ridge map to the target's final state
   and the target's own head give the token at p + j (j = 0: the token itself, 1: the next, 2, 3, 4, 8); ridge to the
   target's entropy and top-1 probability at p gives R2. Regressors: the trained code, the PCA code of the same size
   (what the code was before training), the full fused feature, and each of them ON TOP OF the tokens the drafter
   already has (embeddings of x_p and x_{p+1}). The last group is the one that matters: what the code adds to the
   known tokens, and what training kept or threw away relative to PCA.
B. Use (interventions on the trained drafter, lag 8 and 16). tau when the in-flight positions show the code at all
   of them / only the last 1, 2, 4 / all but the last 4 / shuffled inside the span / taken from other positions of
   the sequence / hidden. And accuracy by block position given a correct prefix (the hazard curve): does the code
   repair the first drafted token (the fork) or the tokens after it?
C. Geometry. Overlap of the trained encoder's linear subspace with the PCA subspace it started from; effective
   dimension of the code (participation ratio).

  python scripts/exp126_code_content.py --data /scratch-shared/$USER/hspec_lagft/data \
      --ckpts /scratch-shared/$USER/hspec_code/r0 .../r16 .../r64 .../r256 .../rreal
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import _output_head, extract_context_feature  # noqa: E402
from exp125_code_bottleneck import Code  # noqa: E402

from hspec.lagtrain import pack, sample_blocks  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

BF = torch.bfloat16
JS = (0, 1, 2, 3, 4, 8)


def unit(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="exp125 output directories (state.pt)")
    ap.add_argument("--fit-seqs", type=int, default=300)
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--val-blocks", type=int, default=16)
    ap.add_argument("--lags", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="exp126_code_content")
    args = ap.parse_args()
    dev, t0 = "cuda", time.time()

    rng = random.Random(args.seed)                           # the split of train_lag.py / exp125
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
    data = [d for d in data if len(d["ids"]) <= args.max_len]
    rng.shuffle(data)
    val, train = data[: args.val_seqs], data[args.val_seqs :]
    fit = train[: args.fit_seqs]

    target = load_target(args.target, dev)
    H = target.config.hidden_size
    emb = target.get_input_embeddings()
    out = {"args": vars(args), "content": {}, "use": {}, "hazard": {}, "geometry": {}}

    def seq_of(d, lids):
        ids = d["ids"].long().to(dev)
        hs = target(ids[None], output_hidden_states=True, logits_to_keep=1).hidden_states
        return ids, hs, extract_context_feature(hs, lids)

    lm = target.get_output_embeddings()

    def head(state):                                         # hidden_states[-1] (already normed, as exp115) -> logits
        return lm(state.to(lm.weight.dtype)).float()

    def block_logits(draft, real, extra, ids, s, g, bs, rows):
        """one block at anchor s, lag g; rows = positions of `extra` it may read (None: all of s-g .. s-1)"""
        T = ids.shape[0]
        noise, pos, mask, starts = pack(draft, target, real, ids, [(s, g)], bs)
        ctx = real
        if extra is not None:
            ctx = torch.cat([real, extra.to(real.dtype)], 1)
            q = noise.shape[1]
            m = torch.zeros((q, 2 * T + q), dtype=torch.bool, device=dev)
            m[:, :T] = mask[0, 0, :, :T]
            m[:, 2 * T :] = mask[0, 0, :, T:]
            vis = list(range(s - g, s)) if rows is None else rows
            if vis:
                m[:, torch.tensor(vis, device=dev) + T] = True
            mask = m[None, None]
            pos = torch.cat([torch.arange(T, device=dev)[None], pos], 1)
        with torch.autocast("cuda", dtype=BF):
            hid = draft(target_hidden=ctx, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                        past_key_values=None, use_cache=False)[0]
            return draft.compute_logits(hid[starts[0] : starts[0] + bs - 1], _output_head(target)).float()

    for ck in args.ckpts:
        st = torch.load(Path(ck) / "state.pt", weights_only=False)
        a = st["args"]
        arm = str(a["code"])
        if st.get("proj") is not None:
            print(f"[skip] {ck}: a path / state arm, not a code arm", flush=True)
            continue
        draft = load_draft(a["draft"], dev).float()
        fc = draft.fc
        draft.fc = nn.Identity()
        draft.load_state_dict(st["draft"])
        fc.load_state_dict(st["fc"])
        draft.eval()
        lids, bs = list(draft.target_layer_ids), draft.block_size
        Hd = fc.out_features
        code = None
        if st.get("code") is not None:
            r = st["code"]["enc.weight"].shape[0]
            code = Code(Hd, r, torch.zeros(Hd), torch.zeros(Hd, r)).to(dev)
            code.load_state_dict(st["code"])
        mode = "real" if arm == "real" else ("code" if code is not None else "none")
        print(f"\n===== {ck}: arm {arm} ({mode}), {time.time() - t0:.0f}s", flush=True)

        def fused(feats):
            return fc(feats.float())

        # ---------------- C + the PCA code of the same size ----------------
        U = mu = cstd = pstd = None
        if mode == "code":
            sm = torch.zeros(Hd, dtype=torch.float64, device=dev)
            xx = torch.zeros(Hd, Hd, dtype=torch.float64, device=dev)
            cs, cc, n = torch.zeros(r, dtype=torch.float64, device=dev), torch.zeros(r, r, dtype=torch.float64, device=dev), 0
            for d in fit[:150]:
                ids, hs, feats = seq_of(d, lids)
                x = fused(feats)[0, d["n_prompt"] :]
                c = (code.enc(x) + code.enc_res(x)).double()
                x = x.double()
                sm += x.sum(0)
                xx += x.T @ x
                cs += c.sum(0)
                cc += c.T @ c
                n += x.shape[0]
                del hs
            mu = sm / n
            ev, evec = torch.linalg.eigh(xx / n - torch.outer(mu, mu))
            U = evec[:, -r:].flip(1).float()
            mu = mu.float()
            ccov = cc / n - torch.outer(cs / n, cs / n)
            cev = torch.linalg.eigvalsh(ccov).clamp_min(0)
            Q = torch.linalg.qr(code.enc.weight.T.float())[0]            # basis of the encoder's linear row space
            cstd = ccov.diagonal().mean().sqrt().float()
            pstd = (ev[-r:].mean()).sqrt().float()
            geo = {"arm": arm, "r": r, "overlap with PCA-r subspace": float((U.T @ Q).pow(2).sum() / r),
                   "PCA-r share of feature variance": float(ev[-r:].sum() / ev.sum()),
                   "code participation ratio": float(cev.sum() ** 2 / cev.pow(2).sum()),
                   "nonlinear share of the code": 0.0}
            lin = res = 0.0
            for d in fit[:20]:
                ids, hs, feats = seq_of(d, lids)
                x = fused(feats)[0, d["n_prompt"] :]
                lin += float(code.enc(x).var(0).sum())
                res += float(code.enc_res(x).var(0).sum())
                del hs
            geo["nonlinear share of the code"] = res / max(lin + res, 1e-9)
            out["geometry"][arm] = geo
            print_table([geo], list(geo), f"C. Geometry of the trained code, arm {arm}")

        # ---------------- A. content ----------------
        if mode == "code":
            def regs(ids, hs, feats, lo, hi):
                """regressor sets for positions lo .. hi-1"""
                p = torch.arange(lo, hi, device=dev)
                x = fused(feats)[0, p]
                c = (code.enc(x) + code.enc_res(x)) / cstd
                pc = ((x - mu) @ U) / pstd
                full = unit(x)
                tok = torch.cat([unit(emb(ids[p]).float()), unit(emb(ids[p + 1]).float())], -1)
                one = torch.ones(len(p), 1, device=dev)
                return {"code": torch.cat([c, one], -1), "pca": torch.cat([pc, one], -1),
                        "full": torch.cat([full, one], -1), "tokens": torch.cat([tok, one], -1),
                        "tokens + code": torch.cat([tok, c, one], -1), "tokens + pca": torch.cat([tok, pc, one], -1),
                        "tokens + full": torch.cat([tok, full, one], -1)}

            def targets(ids, hs, lo, hi):
                p = torch.arange(lo, hi, device=dev)
                fin = hs[-1][0].float()
                Y = {j: fin[p + j - 1] for j in JS}
                pr = torch.softmax(head(fin[p]), -1)
                ent = -(pr * pr.clamp_min(1e-9).log()).sum(-1)
                return Y, torch.stack([ent, pr.max(-1).values], -1)

            acc = None
            for d in fit:
                ids, hs, feats = seq_of(d, lids)
                lo, hi = max(d["n_prompt"], 1), len(ids) - max(JS)
                if hi - lo < 4:
                    continue
                R = regs(ids, hs, feats, lo, hi)
                Y, S = targets(ids, hs, lo, hi)
                if acc is None:
                    acc = {k: {"xx": torch.zeros(v.shape[1], v.shape[1], dtype=torch.float64, device=dev),
                               "xy": {j: torch.zeros(v.shape[1], H, dtype=torch.float32, device=dev) for j in JS},
                               "xs": torch.zeros(v.shape[1], 2, dtype=torch.float64, device=dev)} for k, v in R.items()}
                for k, v in R.items():
                    vd = v.double()
                    acc[k]["xx"] += vd.T @ vd
                    for j in JS:
                        acc[k]["xy"][j] += (vd.T @ Y[j].double()).float()
                    acc[k]["xs"] += vd.T @ S.double()
                del hs
            W = {}
            for k, v in acc.items():
                A = v["xx"] + args.ridge * v["xx"].diagonal().mean() * torch.eye(v["xx"].shape[0], dtype=torch.float64, device=dev)
                W[k] = ({j: torch.linalg.solve(A, v["xy"][j].double()).float() for j in JS},
                        torch.linalg.solve(A, v["xs"]).float())
                v["xy"].clear()
            del acc
            print(f"[content] readouts fitted on {len(fit)} sequences, {time.time() - t0:.0f}s", flush=True)
            hit = {k: {j: [0, 0] for j in JS} for k in W}    # top-1, top-5
            sse = {k: torch.zeros(2, device=dev) for k in W}
            ssum, ssq, nrow = torch.zeros(2, device=dev), torch.zeros(2, device=dev), 0
            for d in val:
                ids, hs, feats = seq_of(d, lids)
                lo, hi = max(d["n_prompt"], 1), len(ids) - max(JS)
                if hi - lo < 4:
                    continue
                R = regs(ids, hs, feats, lo, hi)
                _, S = targets(ids, hs, lo, hi)
                p = torch.arange(lo, hi, device=dev)
                nrow += len(p)
                ssum += S.sum(0)
                ssq += S.pow(2).sum(0)
                for k in W:
                    for j in JS:
                        top = head(R[k] @ W[k][0][j]).topk(5, -1).indices
                        tru = ids[p + j]
                        hit[k][j][0] += int((top[:, 0] == tru).sum())
                        hit[k][j][1] += int((top == tru[:, None]).any(-1).sum())
                    sse[k] += (R[k] @ W[k][1] - S).pow(2).sum(0)
                del hs
            var = ssq / nrow - (ssum / nrow) ** 2
            rows1, rows5 = [], []
            for k in W:
                r2 = 1 - sse[k] / nrow / var
                rows1.append({"regressors": k, **{f"p+{j}": hit[k][j][0] / nrow for j in JS},
                              "R2 entropy": float(r2[0]), "R2 top-1 prob": float(r2[1])})
                rows5.append({"regressors": k, **{f"p+{j}": hit[k][j][1] / nrow for j in JS}})
            print_table(rows1, list(rows1[0]), f"A. Content of the code, arm {arm}: top-1 of the token at p + j read "
                                               f"through the target's head, and R2 of the target's uncertainty at p "
                                               f"({nrow} held-out positions)")
            print_table(rows5, list(rows5[0]), f"A. top-5, arm {arm}")
            out["content"][arm] = {"top1": rows1, "top5": rows5}
            del W

        # ---------------- B. use ----------------
        vrng = random.Random(123)
        conds = ["all"] if mode != "code" else ["all", "last 1", "last 2", "last 4", "all but last 4", "shuffled",
                                                 "other positions", "hidden"]
        if mode == "none":
            conds = ["hidden"]
        taus = {(g, c): [] for g in args.lags + [0] for c in conds}
        hz = {g: [[0, 0] for _ in range(bs - 1)] for g in args.lags + [0]}
        for d in val:
            ids, hs, feats = seq_of(d, lids)
            real = fused(feats)
            base = real if mode == "real" else (code(real) if mode == "code" else None)
            T = len(ids)
            anchors = [s for s, _ in sample_blocks(d["n_prompt"], T, args.val_blocks, bs, 0, 1.0, vrng)]
            for s in anchors:
                tru = ids[s + 1 : s + bs]
                for g0 in args.lags + [0]:
                    g = min(g0, s - 1)
                    span = list(range(s - g, s))
                    for c in conds:
                        extra, rows = base, None
                        if c == "hidden" or g == 0:
                            rows = []
                        elif c.startswith("last "):
                            rows = span[-int(c.split()[1]) :]
                        elif c == "all but last 4":
                            rows = span[:-4]
                        elif c == "shuffled":
                            perm = span[:]
                            vrng.shuffle(perm)
                            extra = base.clone()
                            extra[0, span] = base[0, perm]
                        elif c == "other positions":
                            pool = [i for i in range(d["n_prompt"], T) if i < s - g or i >= s + bs]
                            if len(pool) < g:
                                continue
                            extra = base.clone()
                            extra[0, span] = base[0, vrng.sample(pool, g)]
                        lg = block_logits(draft, real, extra if mode != "none" else None, ids, s, g, bs, rows)
                        ok = (lg.argmax(-1)[: len(tru)] == tru).tolist()
                        taus[(g0, c)].append(1 + (ok.index(False) if False in ok else len(ok)))
                        if c == conds[0]:
                            for k, o in enumerate(ok):
                                hz[g0][k][0] += 1
                                hz[g0][k][1] += o
                                if not o:
                                    break
            del hs
        rows = [{"in-flight positions show": c, **{f"lag {g}": sum(taus[(g, c)]) / max(len(taus[(g, c)]), 1)
                                                  for g in [0] + args.lags}} for c in conds]
        print_table(rows, list(rows[0]), f"B. Use of the code, arm {arm}: tau under interventions "
                                         f"({len(taus[(0, conds[0])])} blocks)")
        hrows = [{"lag": g, **{f"k={k + 1}": hz[g][k][1] / max(hz[g][k][0], 1) for k in range(bs - 1)}}
                 for g in [0] + args.lags]
        print_table(hrows, list(hrows[0]), f"B. Accuracy at block position k given a correct prefix, arm {arm} "
                                           f"(the in-flight positions show: {conds[0]})")
        out["use"][arm], out["hazard"][arm] = rows, hrows
        del draft, fc, code
        torch.cuda.empty_cache()

    save_json(args.save, out)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
