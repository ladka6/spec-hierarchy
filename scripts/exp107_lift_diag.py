"""Experiment 107: where does the "partial pass + mapping" idea break, and why? (no gradient training)

Setting as exp106: in-flight positions only get the target's first k layers (h_k, exact, over the real KV cache);
DFlash's missing deep taps for the last m context positions are filled from a closed-form map of the partial pass.
Four diagnostics, all with the frozen drafter and maps fitted by ridge regression (standardised normal equations):

A1 chain length   share of real vs m (number of provisional positions: 4 / 8 / 16 / 32 ~ one to three chained blocks)
A2 map inputs     which information closes the gap (per exit k, per missing tap):
                    ridge  [h_k]                                  (exp106)
                    stack  [h_{k/2}, h_k]                         (several shallow layers, ADEPT-style)
                    ctx    [h_k, real tap of the last verified position]   (verified deep context)
                    both   [h_{k/2}, h_k, real tap of the last verified position]
                  context maps are fitted with a random distance d in 1..max(m) to the "last verified" position, so
                  they see the same situation as in a chain; their quality is also reported per distance d.
A3 where it fails per-tap ablation at m = 8 (only that tap estimated, the others real), centred cosine per distance d,
                  and the share of real split by the target's confidence at the anchor and by the lifted exit
                  confidence over the provisional positions (is there a signal a reliability gate could use?)
A4 forks          at the drafter's first error in a block drafted from real features: top-1 / top-3 of the raw
                  exit head (final norm + LM head on h_k) and of the lifted final state (each map variant -> final
                  state -> LM head), split into target near-ties (margin < --tie) and clear forks

Data: --data (shard*.pt from gen_traindata.py) or --cache (list of (ids, n_prompt)); the target's own greedy text.

  python scripts/exp107_lift_diag.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 --cache ~/seqs.pt
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

VARIANTS = ("ridge", "stack", "ctx", "both")


def load_seqs(args):
    seqs = []
    if args.cache:
        for r in torch.load(args.cache):
            ids, lo = (r["ids"], r["n_prompt"]) if isinstance(r, dict) else r
            seqs.append((torch.as_tensor(ids).long(), int(lo)))
    else:
        for f in sorted(Path(args.data).expanduser().glob("shard*.pt")):
            for d in torch.load(f):
                seqs.append((d["ids"].long(), int(d["n_prompt"])))
    random.Random(0).shuffle(seqs)
    return [(ids[: args.max_len], lo) for ids, lo in seqs if len(ids[: args.max_len]) - lo > 48]


class Ridge:
    """Accumulates centred, standardised normal equations for one input space and several targets."""

    def __init__(self, din, targets, H, dev):
        self.n = 0
        self.sx = torch.zeros(din, device=dev)
        self.xx = torch.zeros(din, din, device=dev)
        self.sy = {t: torch.zeros(H, device=dev) for t in targets}
        self.xy = {t: torch.zeros(din, H, device=dev) for t in targets}

    def add(self, X, Ys):
        self.n += len(X)
        self.sx += X.sum(0)
        self.xx += X.T @ X
        for t, Y in Ys.items():
            self.sy[t] += Y.sum(0)
            self.xy[t] += X.T @ Y

    def solve(self, lam):
        n = self.n
        mx = self.sx / n
        C = self.xx / n - torch.outer(mx, mx)
        sd = C.diagonal().clamp_min(1e-8).sqrt()
        Cs = C / torch.outer(sd, sd)
        A = Cs + lam * torch.eye(len(sd), device=sd.device)
        out = {}
        for t in self.xy:
            my = self.sy[t] / n
            Cxy = (self.xy[t] / n - torch.outer(mx, my)) / sd[:, None]
            out[t] = (torch.linalg.solve(A, Cxy) / sd[:, None], mx, my)
        return out


def apply_map(Wm, X):
    W, mx, my = Wm
    return (X - mx) @ W + my


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--exits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--ms", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--m-ablate", type=int, default=8, help="m for the per-tap ablation and the A3 splits")
    ap.add_argument("--fit-tokens", type=int, default=35_000)
    ap.add_argument("--val-tokens", type=int, default=5_000)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--tie", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--save", default="exp107_lift_diag")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    assert args.m_ablate in args.ms, "--m-ablate must be one of --ms"
    dev, t0 = "cuda", time.time()
    gen = torch.Generator().manual_seed(0)

    seqs = load_seqs(args)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    norm, head = target.model.norm, target.lm_head
    lids, bs = list(draft.target_layer_ids), draft.block_size
    hidx = [l + 1 for l in lids]
    H, nT = target.config.hidden_size, len(lids)
    M = max(args.ms)
    missing = {k: [j for j, h in enumerate(hidx) if h > k] for k in args.exits}
    tgts = {k: [f"tap{j}" for j in missing[k]] + ["final"] for k in args.exits}
    print(f"[setup] {len(seqs)} sequences, taps hs{hidx}, missing "
          + ", ".join(f"k={k}: hs{[hidx[j] for j in v]}" for k, v in missing.items()), flush=True)

    def hidden(ids):
        return target(ids[None].to(dev), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states

    def tgt_layer(t):
        return -1 if t == "final" else hidx[int(t[3:])]

    def inputs(v, k, t, hs, idx, ctx_idx):
        """map inputs for positions idx; ctx_idx = position of the last verified token (real deep tap)."""
        hk = hs[k][0, idx].float()
        if v == "ridge":
            return hk
        hm = hs[k // 2][0, idx].float()
        if v == "stack":
            return torch.cat([hm, hk], -1)
        c = hs[tgt_layer(t)][0, ctx_idx].float()
        return torch.cat([hk, c], -1) if v == "ctx" else torch.cat([hm, hk, c], -1)

    # ---- fit, one exit at a time (accumulators are large): one accumulator per (variant, k) for
    # target-independent inputs, per (variant, k, target) for the context maps
    def ccos(a, b, mu):
        return F.cosine_similarity(a - mu, b - mu, dim=-1)

    fit_seqs, val, n, i = [], [], 0, 0
    max_fit = int(0.6 * len(seqs))
    while n < args.fit_tokens + args.val_tokens and i < max_fit:
        ids, lo = seqs[i]
        i += 1
        T = len(ids)
        idx = torch.arange(max(lo, M + 1), T)
        d = torch.randint(1, M + 1, (len(idx),), generator=gen)
        if n >= args.fit_tokens:
            hs = hidden(ids)
            val.append((tuple(h.clone() for h in hs), idx.to(dev), (idx - d).to(dev)))
            del hs
        else:
            fit_seqs.append((ids, idx, idx - d))
        n += len(idx)
    assert val, "no validation tokens left: lower --fit-tokens or add data"
    fit_end = i
    maps, lam_used = {}, {}
    for k in args.exits:
        acc = {("ridge", k): Ridge(H, tgts[k], H, dev), ("stack", k): Ridge(2 * H, tgts[k], H, dev)}
        for t in tgts[k]:
            acc[("ctx", k, t)] = Ridge(2 * H, [t], H, dev)
            acc[("both", k, t)] = Ridge(3 * H, [t], H, dev)
        for ids, idx, cidx in fit_seqs:
            hs = hidden(ids)
            idx, cidx = idx.to(dev), cidx.to(dev)
            Ys = {t: hs[tgt_layer(t)][0, idx].float() for t in tgts[k]}
            acc[("ridge", k)].add(inputs("ridge", k, None, hs, idx, cidx), Ys)
            acc[("stack", k)].add(inputs("stack", k, None, hs, idx, cidx), Ys)
            for t in tgts[k]:
                acc[("ctx", k, t)].add(inputs("ctx", k, t, hs, idx, cidx), {t: Ys[t]})
                acc[("both", k, t)].add(inputs("both", k, t, hs, idx, cidx), {t: Ys[t]})
            del hs
        for key in list(acc):
            a = acc.pop(key)
            best = {}
            for lam in args.lams:
                for t, Wm in a.solve(lam).items():
                    sc = torch.cat([ccos(apply_map(Wm, inputs(key[0], k, t, hsv, vidx, vcidx)),
                                         hsv[tgt_layer(t)][0, vidx].float(), Wm[2]) for hsv, vidx, vcidx in val])
                    sc = sc.mean().item()
                    if t not in best or sc > best[t][0]:
                        best[t] = (sc, lam, Wm)
            for t, (sc, lam, Wm) in best.items():
                maps[(key[0], k, t)] = Wm
                lam_used[(key[0], k, t)] = lam
            del a
            torch.cuda.empty_cache()
        print(f"[fit] k={k} done, {time.time() - t0:.0f}s", flush=True)
    del val
    print(f"[fit] {min(n, args.fit_tokens)} tokens from {len(fit_seqs)} sequences, {len(maps)} maps", flush=True)

    def predict(v, k, t, hs, idx, cidx):
        return apply_map(maps[(v, k, t)], inputs(v, k, t, hs, idx, cidx))

    # ---- evaluation on held-out sequences ----
    rest = seqs[fit_end:]
    cand = [(si, s) for si, (ids, lo) in enumerate(rest) for s in range(max(lo, M + 2), len(ids) - bs)]
    random.Random(1).shuffle(cand)
    cand = sorted(cand[: args.anchors])
    by_seq = {}
    for si, s in cand:
        by_seq.setdefault(si, []).append(s)

    tau = {}                                   # condition -> list of per-anchor tau (anchor order)
    meta = {"tconf": [], "lconf": {}}          # per-anchor target confidence; lifted exit confidence per (v, k)

    def rec(name, val_):
        tau.setdefault(name, []).append(val_)

    close = {}                                 # (v, k, t, dbucket) -> list of centred cos
    dbuckets = [(a, min(b, M)) for a, b in ((1, 4), (5, 8), (9, 16), (17, 32), (33, 64)) if a <= M]
    forks = {}                                 # (resolver, tie) -> [top1, top3]

    def run_block(fz, x, s, lag=0):
        L = blocks_logits(draft, target, fz, x, [(s, lag)], bs)[0]
        return L.argmax(-1).tolist()

    for n_done, (si, anchors) in enumerate(by_seq.items()):
        ids, lo = rest[si]
        x = ids.to(dev)
        T = len(ids)
        hs = hidden(ids)
        feats = extract_context_feature(hs, lids)
        truth_all = x.tolist()
        logits_real = head(hs[-1][0]).float()                     # hs[-1] is already normed in HF
        # closeness per distance d (response positions, random d)
        idx = torch.arange(max(lo, M + 1), T, device=dev)
        d = torch.randint(1, M + 1, (len(idx),), generator=gen).to(dev)
        for k in args.exits:
            for t in tgts[k]:
                y = hs[tgt_layer(t)][0, idx].float()
                for v in VARIANTS:
                    Wm = maps[(v, k, t)]
                    c = ccos(apply_map(Wm, inputs(v, k, t, hs, idx, idx - d)), y, Wm[2])
                    for a, b in dbuckets:
                        sel = (d >= a) & (d <= b)
                        close.setdefault((v, k, t, f"{a}-{b}"), []).append(c[sel].cpu())
        for s in anchors:
            truth = truth_all[s + 1 :]
            # target confidence at the anchor (predicting x[s+1])
            meta["tconf"].append(torch.softmax(logits_real[s], -1).max().item())
            chain = run_block(feats[:, :s], x, s)
            rec("real", matched(chain, truth) + 1)
            # A4: forks of the real-feature block
            e = matched(chain, truth)
            if e < len(chain):
                p = s + e                                          # state that predicts the forked token
                tok = truth_all[p + 1]
                pr = torch.softmax(logits_real[p], -1).topk(2).values
                tie = "near-tie" if (pr[0] - pr[1]).item() < args.tie else "clear"
                pidx = torch.tensor([p], device=dev)
                cidx = torch.tensor([s - 1], device=dev)
                for k in args.exits:
                    res = {"raw_exit": head(norm(hs[k][0, p : p + 1])).float()[0]}
                    for v in VARIANTS:
                        res[v] = head(predict(v, k, "final", hs, pidx, cidx).to(hs[-1].dtype)).float()[0]
                    for r, lg in res.items():
                        top = lg.topk(3).indices.tolist()
                        f = forks.setdefault((f"{r}_k{k}", tie), [0, 0, 0])
                        f[0] += top[0] == tok
                        f[1] += tok in top
                        f[2] += 1
            for m in args.ms:
                rec(f"removed_m{m}", matched(run_block(feats[:, : s - m], x, s, m), truth) + 1)
                pos = torch.arange(s - m, s, device=dev)
                cidx = torch.full_like(pos, s - m - 1)
                for k in args.exits:
                    for v in VARIANTS:
                        fz = feats[:, :s].clone()
                        blk = fz[0, s - m : s].view(m, nT, H)
                        for j in missing[k]:
                            blk[:, j] = predict(v, k, f"tap{j}", hs, pos, cidx).to(blk.dtype)
                        fz[0, s - m : s] = blk.view(m, nT * H)
                        rec(f"{v}_k{k}_m{m}", matched(run_block(fz, x, s), truth) + 1)
                        if m == args.m_ablate:
                            fin = predict(v, k, "final", hs, pos, cidx)
                            conf = torch.softmax(head(fin.to(hs[-1].dtype)).float(), -1).max(-1).values.mean().item()
                            meta["lconf"].setdefault((v, k), []).append(conf)
                            if v in ("ridge", "both"):
                                for j in missing[k]:
                                    fz = feats[:, :s].clone()
                                    blk = fz[0, s - m : s].view(m, nT, H)
                                    blk[:, j] = predict(v, k, f"tap{j}", hs, pos, cidx).to(blk.dtype)
                                    fz[0, s - m : s] = blk.view(m, nT * H)
                                    rec(f"{v}_k{k}_only{j}", matched(run_block(fz, x, s), truth) + 1)
        del hs, feats
        if n_done % 20 == 0:
            print(f"[run] {n_done}/{len(by_seq)} sequences, {time.time() - t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    real = mean(tau["real"])

    def share(name, m):
        rem = mean(tau[f"removed_m{m}"])
        return (mean(tau[name]) - rem) / max(real - rem, 1e-9)

    # A1 + A2
    rows = [{"m": m, "real": real, "removed": mean(tau[f"removed_m{m}"]),
             **{f"{v}_k{k}": share(f"{v}_k{k}_m{m}", m) for k in args.exits for v in VARIANTS}} for m in args.ms]
    print_table(rows, list(rows[0]), "A1/A2. share of real (tau - removed) / (real - removed) by m (rows) and map "
                                     "(columns); real / removed are tau")
    out = {"args": vars(args), "a1a2": rows}

    # A2: closeness per distance
    rows = []
    for k in args.exits:
        for t in tgts[k]:
            r = {"k": k, "target": t if t == "final" else f"L{lids[int(t[3:])]}"}
            for v in VARIANTS:
                for a, b in dbuckets:
                    r[f"{v} d{a}-{b}"] = torch.cat(close[(v, k, t, f"{a}-{b}")]).mean().item()
            rows.append(r)
    print_table(rows, list(rows[0]), "A2. centred cosine of the mapped state to the real one, by distance d to the "
                                     "last verified position")
    out["closeness"] = rows

    # A3: per-tap ablation
    m = args.m_ablate
    rows = []
    for k in args.exits:
        for v in ("ridge", "both"):
            rows.append({"k": k, "map": v, "all missing": share(f"{v}_k{k}_m{m}", m),
                         **{f"only L{lids[j]}": share(f"{v}_k{k}_only{j}", m) for j in missing[k]}})
    print_table(rows, list(rows[0]), f"A3. per-tap ablation, m = {m}: share of real when only that tap is estimated "
                                     "(others real)")
    out["ablation"] = rows

    # A3: splits by target confidence and by lifted confidence (terciles over anchors)
    rem = tau[f"removed_m{m}"]

    def tercile_rows(conf, label):
        order = sorted(range(len(conf)), key=lambda q: conf[q])
        thirds = [order[: len(order) // 3], order[len(order) // 3 : 2 * len(order) // 3], order[2 * len(order) // 3 :]]
        rws = []
        for name, sel in zip(("low", "mid", "high"), thirds):
            r = {"split": label, "tercile": name, "conf range": f"{conf[sel[0]]:.2f}-{conf[sel[-1]]:.2f}",
                 "real tau": mean([tau['real'][q] for q in sel]), "removed tau": mean([rem[q] for q in sel])}
            for k in args.exits:
                for v in ("ridge", "both"):
                    sub = [tau[f"{v}_k{k}_m{m}"][q] for q in sel]
                    rr, rm = r["real tau"], r["removed tau"]
                    r[f"{v}_k{k}"] = (mean(sub) - rm) / max(rr - rm, 1e-9)
            rws.append(r)
        return rws

    rows = tercile_rows(meta["tconf"], "target conf at anchor")
    for k in args.exits:
        rows += tercile_rows(meta["lconf"][("both", k)], f"lifted conf (both, k={k})")
    print_table(rows, list(rows[0]), f"A3. share of real by confidence tercile, m = {m}")
    out["splits"] = rows

    # A4: forks
    rows = []
    for (r, tie), (t1, t3, cnt) in sorted(forks.items()):
        rows.append({"resolver": r, "fork type": tie, "forks": cnt, "top-1": t1 / cnt, "top-3": t3 / cnt})
    print_table(rows, list(rows[0]), "A4. resolving the drafter's first error (real-feature block)")
    out["forks"] = rows
    out["lams"] = {f"{v}_k{k}_{t}": lam for (v, k, t), lam in lam_used.items()}
    save_json(args.save, out)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
