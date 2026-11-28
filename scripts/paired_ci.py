"""Paired difference of tau between two exp125 runs, with a bootstrap interval.

exp125 evaluates every arm on the same held-out anchors (fixed seed), so two arms can be compared block by block.
The json of a run made after 28 November holds the taus of every block, one list per validation sequence
("end" -> lag -> "taus"). The interval is a bootstrap over the SEQUENCES (blocks of one sequence are not
independent).

  python scripts/paired_ci.py results/exp125_A/exp125_x.json results/exp125_B/exp125_y.json
"""

from __future__ import annotations

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    A, B = (json.load(open(f))["end"] for f in (args.a, args.b))
    rng = random.Random(0)
    print(f"{'lag':>4} {'tau a':>7} {'tau b':>7} {'b - a':>7} {'95% interval':>18} {'blocks':>7}")
    for ra, rb in zip(A, B):
        ta, tb = ra["taus"], rb["taus"]
        assert ra["lag"] == rb["lag"] and [len(x) for x in ta] == [len(x) for x in tb], \
            "the two runs did not use the same anchors"
        groups = [[y - x for x, y in zip(sa, sb)] for sa, sb in zip(ta, tb)]
        d = [v for g in groups for v in g]
        ta, tb = [v for g in ta for v in g], [v for g in tb for v in g]
        means = []
        for _ in range(args.boot):
            pick = [g for g in rng.choices(groups, k=len(groups))]
            means.append(sum(map(sum, pick)) / sum(map(len, pick)))
        means.sort()
        lo, hi = means[int(0.025 * args.boot)], means[int(0.975 * args.boot)]
        print(f"{ra['lag']:>4} {sum(ta) / len(ta):>7.3f} {sum(tb) / len(tb):>7.3f} {sum(d) / len(d):>7.3f} "
              f"{f'[{lo:.3f}, {hi:.3f}]':>18} {len(d):>7}")


if __name__ == "__main__":
    main()
