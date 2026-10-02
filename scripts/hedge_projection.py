"""Pipeline projection with hedged rollbacks: ms/token = m + e * (1 - h) * c_T, side branches at
flagged positions assumed free (the middle-model GPU is memory-bound at batch 1).

  python scripts/hedge_projection.py results/middle [Qwen3-8B-AWQ]
"""

import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results/middle")
name = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-8B-AWQ"
c_T = json.loads((R / "speed_target.json").read_text())["decode_ms_per_token"]
base = json.loads((R / "dflash_target.json").read_text())["decode_ms_per_token"]
m = json.loads((R / f"dflash_{name}.json").read_text())["decode_ms_per_token"]
hd = json.loads((R / f"hedge_{name}.json").read_text())
e = hd["e"]
print(f"{name}: e {e:.3f}, m {m:.2f} ms/tok, c_T {c_T:.2f}; DFlash on target {base:.2f} ms/tok; "
      f"coverage of disagreements by top-2/3/5: " + " ".join(f"{v:.2f}" for v in hd["coverage_of_disagreements"].values()))
print(f"{'margin<':>8}{'flagged':>9}{'h top2':>8}{'h top3':>8}{'ms/tok(2)':>11}{'vs DFlash':>11}{'vs TP2':>8}")
print(f"{'none':>8}{0:>9.3f}{0:>8.2f}{0:>8.2f}{m + e * c_T:>11.2f}{base / (m + e * c_T):>11.2f}{base / (m + e * c_T) / 1.55:>8.2f}")
for c in hd["curve"]:
    ms = m + e * (1 - c["h_top2"]) * c_T
    print(f"{c['margin_lt']:>8.2f}{c['flagged_share']:>9.3f}{c['h_top2']:>8.2f}{c['h_top3']:>8.2f}"
          f"{ms:>11.2f}{base / ms:>11.2f}{base / ms / 1.55:>8.2f}")
print("vs TP2 > 1 beats tensor parallel on 2 GPUs (if side branches really are free)")
