"""Experiment 33: when to stop refining and send the chain to the verifier (proposal 2, offline).

Input: exp29's saved per-iteration states (results/iterate/exp29_rules_t*.json, key "traj"): for
every anchor, iteration k = 0 .. K with (chain_len, accepted, full_agree, mean_conf, j, min_conf,
copy_prob_at_correction, copy_mean_prob, dataset). Stopping at k costs
  t(k) = (k+1) * drafter + k * copy + verify(chain_len_k + 1)
and yields accepted_k tokens. Goal: maximise total tokens / total time (throughput).

Policies (all may stop at k = 0, i.e. plain DFlash, which exp29's rules could not):
  fixed k        always k iterations
  conf c         continue while the new block's mean drafter confidence >= c (exp29's best rule family)
  myopic-learned continue while the predicted gain of one more iteration, E[acc_{k+1} - acc_k],
                 exceeds lambda * (its extra time), lambda = the throughput being targeted. The gain
                 model is fitted on three datasets and tested on the fourth (no leakage).
  oracle         per anchor the k maximising acc_k - lambda * t_k (knows the future: upper bound)

lambda is found by fixed-point iteration (Dinkelbach): lambda <- throughput achieved with it.

  python scripts/exp33_stopping.py results/iterate/exp29_rules_t0.0.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

VLLM_V = [(1, 12.7), (16, 13.3), (32, 14.1), (64, 14.2), (128, 15.8), (256, 19.0)]
HF_V = [(1, 20.8), (16, 22.0), (32, 22.8), (48, 23.3), (64, 23.6), (128, 25.5), (256, 29.0)]
ENGINES = {"vLLM": (2.6, [3.0, 4.5, 6.0], VLLM_V), "HF+graphs": (3.7, [15.5, 17.7], HF_V)}


def interp(tab, q):
    for (q0, t0), (q1, t1) in zip(tab, tab[1:]):
        if q <= q1:
            return t0 + (t1 - t0) * (q - q0) / (q1 - q0)
    return tab[-1][1] * q / tab[-1][0]


def feats(st, k):
    """Features of state k (decision: continue to k+1?)."""
    ln, _acc, full, conf, j, mn, cpj, cpm, _d = st[k]
    cpj = 1.0 if cpj != cpj else cpj          # nan at k = 0
    cpm = 1.0 if cpm != cpm else cpm
    j = 15.0 if j < 0 else float(j)
    x = [1.0, k, ln / 16, float(full), conf, mn, cpj, cpm, j / 16, conf * conf, conf * k, mn * conf]
    return x


def run(traj, dd, p, vt, policy, lam):
    tok = tim = ks = 0.0
    for st in traj:
        K = len(st) - 1
        t = [(k + 1) * dd + k * p + interp(vt, st[k][0] + 1) for k in range(K + 1)]
        k = policy(st, t, lam, K)
        tok += st[k][1]
        tim += t[k]
        ks += k
    return tok / tim, ks / len(traj), tok / len(traj)


def solve(traj, dd, p, vt, policy, iters=8):
    lam = sum(st[0][1] for st in traj) / sum(dd + interp(vt, st[0][0] + 1) for st in traj)
    for _ in range(iters):
        thr, mk, tau = run(traj, dd, p, vt, policy, lam)
        lam = thr
    return thr, mk, tau


def main():
    path = sys.argv[1]
    data = json.load(open(path))
    traj = [[tuple(s) for s in st] for st in data["traj"]]
    K = len(traj[0]) - 1
    dsets = sorted({st[0][-1] for st in traj})
    print(f"{len(traj)} anchors, up to {K} iterations, datasets {dsets}")

    def fit_gain(train):
        X, y = [], []
        for st in train:
            for k in range(len(st) - 1):
                X.append(feats(st, k))
                y.append(st[k + 1][1] - st[k][1])
        X, y = np.array(X), np.array(y)
        reg = 1e-2 * np.eye(X.shape[1])
        return np.linalg.solve(X.T @ X + reg, X.T @ y)

    models = {d: fit_gain([st for st in traj if st[0][-1] != d]) for d in dsets}   # leave one dataset out

    def p_fixed(kf):
        return lambda st, t, lam, K_: min(kf, K_)

    def p_conf(c):
        def f(st, t, lam, K_):
            k = 0
            while k < K_ and st[k][3] >= c:
                k += 1
            return k
        return f

    def p_learned(st, t, lam, K_):
        w = models[st[0][-1]]
        k = 0
        while k < K_:
            gain = float(np.dot(w, feats(st, k)))
            if gain <= lam * (t[k + 1] - t[k]):
                break
            k += 1
        return k

    def p_oracle(st, t, lam, K_):
        return max(range(K_ + 1), key=lambda k: st[k][1] - lam * t[k])

    pols = [(f"fixed {k}", p_fixed(k)) for k in range(K + 1)] + \
        [(f"conf {c}", p_conf(c)) for c in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)] + \
        [("myopic-learned", p_learned), ("oracle", p_oracle)]
    for eng, (dd, ps, vt) in ENGINES.items():
        for p in ps:
            base = solve(traj, dd, p, vt, p_fixed(0))[0]
            print(f"\n== {eng}, copy {p} ms (x = throughput / plain DFlash) ==")
            print(f"{'policy':16s} {'x':>6s} {'mean_k':>7s} {'tau':>7s}")
            best = None
            for name, pol in pols:
                thr, mk, tau = solve(traj, dd, p, vt, pol)
                print(f"{name:16s} {thr / base:6.3f} {mk:7.2f} {tau:7.2f}")
                if name not in ("oracle",) and (best is None or thr > best[1]):
                    best = (name, thr)
            print(f"best non-oracle: {best[0]} ({best[1] / base:.3f}x)")


if __name__ == "__main__":
    main()
