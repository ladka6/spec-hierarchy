"""Drafter x target matrix from exp21 runs named TARGET@DRAFTER (snellius/job_matrix.sbatch).

  python scripts/matrix.py results/matrix
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TARGETS = ["base", "math200", "code200", "chat1000"]
DRAFTERS = ["base", "ctrl", "math200", "code200", "chat1000", "joint"]
NOTE = {"base": "no update", "ctrl": "base data, same recipe", "joint": "all targets' data"}


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/matrix")
    cells = {}
    for f in d.glob("exp21_*@*.json"):
        t, dr = f.stem[len("exp21_"):].split("@", 1)
        cells[(dr, t)] = {r["method"]: r for r in json.loads(f.read_text())["rows"]}
    drafters = [x for x in DRAFTERS if any((x, t) in cells for t in TARGETS)]
    drafters += sorted({dr for dr, _ in cells} - set(drafters))
    for metric, method, title in [("speedup", "ddtree_tb128", "DDTree-128 speedup over AR"),
                                  ("tau", "ddtree_tb128", "DDTree-128 acceptance length"),
                                  ("tau", "dflash", "DFlash acceptance length")]:
        print(f"\n{title} (rows: drafter, cols: target)")
        print(f"{'drafter':<10}" + "".join(f"{t:>10}" for t in TARGETS) + f"{'mean':>10}  note")
        for dr in drafters:
            vals = [cells.get((dr, t), {}).get(method, {}).get(metric) for t in TARGETS]
            got = [v for v in vals if v is not None]
            mean = sum(got) / len(got) if len(got) == len(TARGETS) else None
            fmt = lambda v: f"{v:>10.2f}" if v is not None else f"{'-':>10}"  # noqa: E731
            print(f"{dr:<10}" + "".join(fmt(v) for v in vals) + fmt(mean) + "  " + NOTE.get(dr, ""))
    print("\nspecialist = diagonal (drafter trained on that target's data); forgetting = its 'base' column")


if __name__ == "__main__":
    main()
