"""Wall-clock estimate for feature-refreshed drafting from exp27 results and measured vLLM costs.

Per round (batch 1):
  plain DFlash chain   d + v(16)
  refreshed            d + v(nodes + 1) + ran * (p + d2)
    d    drafter pass (block 1)
    p    cheap target copy over the 16 drafted tokens (4-bit AWQ in vLLM: 3-6 ms measured)
    d2   second, batched drafter pass for the restarts; d2 = d separately, smaller if fused into one
         graph with the copy's pass (option 3)
    v(q) target verify of q tokens, interpolated from the vLLM table
Speedup = (tau / round) / (tau_plain / round_plain).

  python scripts/costmodel_refresh.py results/refresh/exp27_*.json
"""

from __future__ import annotations

import argparse
import json

# vLLM, A100, Qwen3-8B bf16, 1024-token context (results/bench_vllm_cost.json), made monotone
VTAB = [(1, 12.7), (16, 13.3), (32, 14.1), (64, 14.2), (128, 15.8), (256, 19.0)]


def v(q):
    for (q0, t0), (q1, t1) in zip(VTAB, VTAB[1:]):
        if q <= q1:
            return t0 + (t1 - t0) * (q - q0) / (q1 - q0)
    return VTAB[-1][1] * q / VTAB[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--d", type=float, default=2.6, help="drafter pass ms (vLLM)")
    ap.add_argument("--ps", nargs="+", type=float, default=[3.0, 4.5, 6.0], help="copy pass ms")
    ap.add_argument("--d2s", nargs="+", type=float, default=[2.6, 1.0], help="second drafter pass ms")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    for f in args.files:
        data = json.load(open(f))
        rows = data["rows"]
        base = next(r for r in rows if r["policy"] == "block1 only")
        t0 = args.d + v(16)
        r0 = base["tau"] / t0
        print(f"\n== {f}  (T={data['args'].get('temp')}) plain DFlash: tau {base['tau']:.2f}, "
              f"{t0:.1f} ms/round, {r0 * 1000:.0f} tok/s")
        out = []
        for r in rows:
            if r["policy"] == "block1 only":
                continue
            for p in args.ps:
                for d2 in args.d2s:
                    t = args.d + v(r["nodes"] + 1) + r.get("ran", 1.0) * (p + d2)
                    out.append((r["tau"] / t / r0, r["source"], r["policy"], r["tau"], r["nodes"],
                                r.get("ran", 1.0), p, d2, t))
        out.sort(reverse=True)
        print(f"{'speedup':>8} {'source':>7} {'policy':>16} {'tau':>6} {'nodes':>6} {'ran':>5} {'p':>4} {'d2':>4} {'ms':>6}")
        for x in out[: args.top]:
            print(f"{x[0]:8.3f} {x[1]:>7} {x[2]:>16} {x[3]:6.2f} {x[4]:6.1f} {x[5]:5.2f} {x[6]:4.1f} {x[7]:4.1f} {x[8]:6.1f}")
        for p in args.ps:
            best = max((x for x in out if x[6] == p and x[7] == args.d), default=None)
            if best:
                print(f"best at p={p} ms, separate second pass: {best[0]:.3f}x ({best[1]} {best[2]})")


if __name__ == "__main__":
    main()
