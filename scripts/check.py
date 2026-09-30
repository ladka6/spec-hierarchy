"""Summarize results/check: drafter x target cells over prompt seeds (mean and half-range),
plus GSM8K accuracy of the targets and the async ceiling table if present.

  python scripts/check.py results/check
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/check")
    cells = defaultdict(list)
    for f in sorted(d.glob("exp21_*@*_s*.json")):
        m = re.match(r"exp21_(.+)@(.+)_s(\d+)$", f.stem)
        t, dr, _ = m.groups()
        rows = {r["method"]: r for r in json.loads(f.read_text())["rows"]}
        cells[(dr, t)].append(rows)
    for method, key, title in [("ddtree_tb128", "speedup", "DDTree-128 speedup"), ("ddtree_tb128", "tau", "DDTree-128 tau"),
                               ("dflash", "tau", "DFlash tau")]:
        print(f"\n{title}: mean +- half-range over prompt seeds (40 prompts/dataset each)")
        for (dr, t), runs in sorted(cells.items()):
            v = [r[method][key] for r in runs]
            print(f"  drafter {dr:<8} target {t:<9} {sum(v) / len(v):6.2f} +- {(max(v) - min(v)) / 2:.2f}  (n_seeds={len(v)})")
    for name in ("exp23_taskacc.json", "exp24_ceiling.json"):
        f = d / name
        if f.exists():
            print(f"\n{name}:")
            for r in json.loads(f.read_text())["rows"]:
                print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
