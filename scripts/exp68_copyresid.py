"""Experiment 68: can the target's verified states correct the cheap copy online? (offline, no training, Colab-ready)

At every verify the target gives, for free, its exact states at the positions it just checked; the copy computed the
same positions earlier. Their difference r_t = target(t) - copy(t) is the copy's error. If r_t is predictable from the
errors at already-verified positions of the SAME text, then copy(t) + r_hat(t) is closer to the target, and the copy's
greedy agreement q rises without training the copy.

On greedy target responses (generated here), teacher forcing (an accepted prefix is the same text for both models):
  1. autocorrelation of r_t: cosine(r_t, r_{t-k}) along the sequence vs the same index in another sequence (baseline)
  2. causal estimators r_hat(t) that only use r at positions <= t - lag (lag = how far verification trails the copy):
       glob_bias     mean residual over the TRAIN sequences (static, no adaptation)
       glob_affine   per-dimension affine map a*h_copy + b fitted on the train sequences (static)
       ema_b         exponential moving average of this sequence's past residuals (decay b)
       win_w         mean of the last w verified residuals
       seq_mean      mean of all verified residuals so far in this sequence
       glob+ema_b    glob_affine, then an EMA of what it leaves (static + online)
  3. metrics on held-out sequences, per estimator and lag:
       q             argmax(LM head(corrected final state)) == target argmax   (and fixed / broken counts)
       explained     1 - |r - r_hat|^2 / |r|^2 on the final state
       cos_taps      cosine to the target's 5 tap features (what the drafter reads), corrected vs plain

  python scripts/exp68_copyresid.py --n 150 --copy bnb4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp67_probe import generate  # noqa: E402

from hspec.data import load_prompts  # noqa: E402
from hspec.models import load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

LAGS = (1, 8, 32)
KS = (1, 2, 4, 8, 16, 32, 64)


def causal_estimates(R, lag, kind, base=None):
    """R [N, D] residuals in order. Returns r_hat [N, D] using only rows <= t - lag (zeros before any is verified)."""
    N, D = R.shape
    out = torch.zeros_like(R)
    if kind.startswith("ema"):
        b = float(kind.split("_")[1])
        e, w = torch.zeros(D, device=R.device), 0.0
        hist = torch.zeros_like(R)
        for t in range(N):
            e = b * e + (1 - b) * R[t]
            w = b * w + (1 - b)
            hist[t] = e / w
        out[lag:] = hist[: N - lag]
    elif kind.startswith("win"):
        W = int(kind.split("_")[1])
        cs = torch.cat([torch.zeros(1, D, device=R.device), R.cumsum(0)])
        for t in range(lag, N):
            hi = t - lag + 1
            lo = max(0, hi - W)
            out[t] = (cs[hi] - cs[lo]) / (hi - lo)
    elif kind == "seq_mean":
        cs = R.cumsum(0)
        cnt = torch.arange(1, N + 1, device=R.device, dtype=R.dtype)[:, None]
        out[lag:] = (cs / cnt)[: N - lag]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--taps", nargs="+", type=int, default=[1, 9, 17, 25, 33])
    ap.add_argument("--save", default="exp68_copyresid")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    tok = load_tokenizer(args.target)
    target = load_target(args.target, dev)
    prompts = []
    for ds in args.datasets:
        try:
            prompts += load_prompts(ds, args.n)
        except Exception as e:  # noqa: BLE001
            print(f"[data] {ds} skipped: {type(e).__name__}: {str(e)[:120]}", flush=True)
    seqs = generate(target, tok, prompts, args.max_new, args.gen_batch, dev)
    random.Random(0).shuffle(seqs)
    half = len(seqs) // 2
    train, val = seqs[:half], seqs[half:]
    print(f"[data] {len(train)} train / {len(val)} val sequences, {time.time() - t0:.0f}s", flush=True)
    copy = load_mid(args.copy, dev)
    norm, head = target.model.norm, target.lm_head
    H = target.config.hidden_size

    @torch.inference_mode()
    def states(model, x, lo):
        out = model(x[None], output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        taps = torch.cat([hs[l + 1][0, lo - 1 : -1] for l in args.taps], -1).float()    # states at positions lo-1..T-2
        last = hs[-1][0, lo - 1 : -1].float()
        am = out.logits[0, lo - 1 : -1].argmax(-1)                                        # predicts tokens lo..T-1
        return taps, last, am

    # decide whether hidden_states[-1] is already normed (logits = head(last)) or not (logits = head(norm(last)))
    with torch.inference_mode():
        x = train[0][0].to(dev)
        _, last, am = states(target, x, train[0][1])
        m_raw = float((head(last.to(head.weight.dtype)).argmax(-1) == am).float().mean())
        m_norm = float((head(norm(last.to(head.weight.dtype))).argmax(-1) == am).float().mean())
    pre_norm = m_norm > m_raw
    print(f"[check] head(last) match {m_raw:.3f}, head(norm(last)) match {m_norm:.3f} -> "
          f"{'pre-norm' if pre_norm else 'post-norm'} last state", flush=True)

    def logits_of(h):
        h = h.to(head.weight.dtype)
        return head(norm(h) if pre_norm else h)

    # ---- static correctors on the train split: global mean residual and per-dimension affine r = a*h + b
    S = {k: torch.zeros(H, device=dev, dtype=torch.float64) for k in ("h", "hh", "r", "hr")}
    n_fit = 0
    with torch.inference_mode():
        for ids, lo in train:
            x = ids.to(dev)
            if len(x) - lo < 4:
                continue
            _, lt, _ = states(target, x, lo)
            _, lc, _ = states(copy, x, lo)
            r = (lt - lc).double()
            h = lc.double()
            S["h"] += h.sum(0); S["hh"] += (h * h).sum(0); S["r"] += r.sum(0); S["hr"] += (h * r).sum(0)
            n_fit += len(h)
    mh, mr = S["h"] / n_fit, S["r"] / n_fit
    var = S["hh"] / n_fit - mh ** 2
    cov = S["hr"] / n_fit - mh * mr
    a = (cov / (var + 1e-6)).float()
    b = (mr - a.double() * mh).float()
    gbias = mr.float()
    print(f"[fit] static correctors on {n_fit} train positions, {time.time() - t0:.0f}s", flush=True)

    kinds = ["none", "glob_bias", "glob_affine", "ema_0.5", "ema_0.9", "ema_0.98", "win_16", "seq_mean", "glob+ema_0.9"]
    acc = defaultdict(lambda: defaultdict(float))     # acc[(kind, lag)][metric]
    ac_same, ac_cross = defaultdict(list), defaultdict(list)
    prev_R = None
    with torch.inference_mode():
        for si, (ids, lo) in enumerate(val):
            x = ids.to(dev)
            if len(x) - lo < 70:
                continue
            tt, lt, amt = states(target, x, lo)
            tc, lc, amc = states(copy, x, lo)
            R = lt - lc                                       # final-state residual [N, H]
            RT = tt - tc                                      # tap residual [N, 5H]
            N = len(R)
            # autocorrelation along the sequence vs another sequence at the same index
            for k in KS:
                if k < N:
                    ac_same[k].append(float(F.cosine_similarity(R[k:], R[:-k], dim=-1).mean()))
            if prev_R is not None:
                m = min(N, len(prev_R))
                ac_cross["other_seq"].append(float(F.cosine_similarity(R[:m], prev_R[:m], dim=-1).mean()))
            prev_R = R
            base_fit = lc * a + b
            for lag in LAGS:
                for kind in kinds:
                    if kind == "none":
                        rh, rht = torch.zeros_like(R), torch.zeros_like(RT)
                    elif kind == "glob_bias":
                        rh, rht = gbias.expand_as(R), None
                    elif kind == "glob_affine":
                        rh, rht = base_fit, None
                    elif kind.startswith("glob+"):
                        rh = base_fit + causal_estimates(R - base_fit, lag, kind[5:])
                        rht = None
                    else:
                        rh = causal_estimates(R, lag, kind)
                        rht = causal_estimates(RT, lag, kind)
                    pred = logits_of(lc + rh).argmax(-1)
                    ok, ok0 = pred == amt, amc == amt
                    d = acc[(kind, lag)]
                    d["n"] += N
                    d["agree"] += float(ok.sum())
                    d["fixed"] += float((ok & ~ok0).sum())
                    d["broken"] += float((~ok & ok0).sum())
                    d["res2"] += float(((R - rh) ** 2).sum())
                    d["tot2"] += float((R ** 2).sum())
                    if rht is not None:
                        a5 = (tc + rht).view(N, len(args.taps), H)
                        d["cos"] += float(F.cosine_similarity(a5, tt.view(N, len(args.taps), H), dim=-1).mean(-1).sum())
                        d["ncos"] += N
            if si % 10 == 0:
                print(f"[val] {si}/{len(val)} sequences, {time.time() - t0:.0f}s", flush=True)

    rows = []
    for (kind, lag), d in acc.items():
        if kind == "none" and lag != LAGS[0]:
            continue
        rows.append({"estimator": kind, "lag": lag if kind not in ("none", "glob_bias", "glob_affine") else "-",
                     "q": d["agree"] / d["n"], "fixed": int(d["fixed"]), "broken": int(d["broken"]),
                     "explained": 1 - d["res2"] / d["tot2"],
                     "cos_taps": (d["cos"] / d["ncos"]) if d.get("ncos") else float("nan")})
    rows = [r for i, r in enumerate(rows) if not (r["lag"] == "-" and any(
        p["estimator"] == r["estimator"] for p in rows[:i]))]
    print_table([{"lag k": k, "cos(r_t, r_t-k)": sum(v) / len(v)} for k, v in sorted(ac_same.items())]
                + [{"lag k": "other sequence", "cos(r_t, r_t-k)": sum(ac_cross["other_seq"]) / max(len(ac_cross["other_seq"]), 1)}],
                ["lag k", "cos(r_t, r_t-k)"], "autocorrelation of the copy's final-state error along the sequence")
    print_table(rows, list(rows[0]), f"online correction of the copy ({args.copy}) from verified target states; "
                                     f"q = greedy agreement with the target, lag = verification delay in tokens")
    save_json(args.save, {"args": vars(args), "rows": rows,
                          "autocorr": {k: sum(v) / len(v) for k, v in ac_same.items()},
                          "autocorr_other_seq": sum(ac_cross["other_seq"]) / max(len(ac_cross["other_seq"]), 1)})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
