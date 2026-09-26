"""Merge exp8_*.json files and print the comparison.

  python scripts/aggregate_exp8.py results/r6_split [results/r7_split ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.utils import mean, print_table  # noqa: E402


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [Path("results")]
    records, seen = [], set()
    for root in roots:
        for f in sorted(root.glob("exp8_*.json")):
            for r in json.loads(f.read_text())["records"]:
                key = (r["config"], r["dataset"], r["i"], r["lat"])
                if key not in seen:          # first directory wins on duplicates
                    seen.add(key)
                    records.append(r)
    if not records:
        sys.exit(f"no exp8_*.json in {roots}")
    groups = defaultdict(list)
    for r in records:
        groups[(r["lat"], r["dataset"], r["config"])].append(r)
        groups[(r["lat"], "ALL", r["config"])].append(r)

    rows = []
    for (lat, d, cfg), rs in sorted(groups.items()):
        rows.append({"lat_ms": lat, "dataset": d, "config": cfg,
                     "tok/s": sum(r["tokens"] for r in rs) / (sum(r["ms"] for r in rs) / 1000),
                     "tgt/tok": mean(r["tgt_calls_per_tok"] for r in rs),
                     "mid/tok": mean(r.get("mid_calls_per_tok", float("nan")) for r in rs),
                     "tau": mean(r.get("tau", float("nan")) for r in rs),
                     "rollbacks": mean(r.get("rollbacks", float("nan")) for r in rs),
                     "caught": mean(r.get("caught", float("nan")) for r in rs),
                     "forks": mean(r.get("forks", float("nan")) for r in rs),
                     "n": len(rs)})
    by = {(r["lat_ms"], r["dataset"], r["config"]): r["tok/s"] for r in rows}
    for r in rows:
        k = (r["lat_ms"], r["dataset"])
        r["x_dflash"] = r["tok/s"] / by.get(k + ("dflash",), float("nan"))
        ddt = [v for (l, d, c), v in by.items() if (l, d) == k and c.startswith("ddtree")]
        r["x_ddtree"] = r["tok/s"] / max(ddt) if ddt else float("nan")

    cols = ["lat_ms", "dataset", "config", "tok/s", "x_dflash", "x_ddtree", "tgt/tok", "mid/tok", "tau",
            "rollbacks", "caught", "forks", "n"]
    print_table([r for r in rows if r["dataset"] == "ALL"], cols, "all datasets pooled")
    print_table([r for r in rows if r["dataset"] != "ALL"], cols, "per dataset")

    lats = sorted({r["lat_ms"] for r in rows})
    pivot = []
    for cfg in sorted({r["config"] for r in rows}):
        row = {"config": cfg}
        for lat in lats:
            m = [r for r in rows if r["config"] == cfg and r["lat_ms"] == lat and r["dataset"] == "ALL"]
            row[f"{lat:g}ms"] = m[0]["x_ddtree"] if m else float("nan")
        pivot.append(row)
    print_table(pivot, ["config"] + [f"{lat:g}ms" for lat in lats], "speedup over the best DDTree, all datasets")


if __name__ == "__main__":
    main()
