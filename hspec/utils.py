"""Small shared helpers for the experiment scripts."""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

RESULTS = Path(os.environ.get("HSPEC_RESULTS", "results"))


def env_info() -> dict:
    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_json(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    payload = {"env": env_info(), **payload}
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved {path}")
    return path


def mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def print_table(rows: list[dict], cols: list[str], title: str = "") -> None:
    if title:
        print(f"\n== {title} ==")
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def slug(spec: str) -> str:
    return spec.replace("/", "_").replace(":", "-")
