"""Projected throughput of the three-model PEARL pipeline from results/middle (vllm_middle + vllm_dflash).

Stages overlap on 2 GPUs (drafter + middle model on one, target on the other). The middle model's
stream advances at 1 / m ms per token (m = its DFlash ms/token, or plain ms/token without a drafter);
the target checks it in windows (one pass ~ one plain decode step c_T, flat up to ~128 tokens);
a disagreement (rate e per token) throws away about one target pass of work:

  tokens/ms = (1/e) / ((1/e) * m + c_T)

Compared with DFlash on the target on 1 GPU and with 2-GPU tensor parallel (measured: 1.55x).
"""

import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results/middle")


def load(name):
    f = R / name
    return json.loads(f.read_text()) if f.exists() else None


c_T = load("speed_target.json")["decode_ms_per_token"]
base = load("dflash_target.json")
base_ms = base["decode_ms_per_token"] if base else None
print(f"target plain {c_T:.2f} ms/tok; DFlash on target (1 GPU) "
      f"{base_ms:.2f} ms/tok" if base_ms else f"target plain {c_T:.2f} ms/tok")
print(f"{'middle':<14}{'e':>7}{'plain':>8}{'dflash':>8}{'tau':>6}{'pipe ms/tok':>13}{'vs DFlash 1GPU':>16}{'vs TP2 (1.55x)':>16}")
for f in sorted(R.glob("agree_*.json")):
    n = f.stem[len("agree_"):]
    e = json.loads(f.read_text())["e"]
    sp, df = load(f"speed_{n}.json"), load(f"dflash_{n}.json")
    m = df["decode_ms_per_token"] if df else (sp["decode_ms_per_token"] if sp else None)
    if m is None or e <= 0:
        continue
    L = 1 / e
    pipe = (L * m + c_T) / L                       # ms per token
    x = base_ms / pipe if base_ms else float("nan")
    print(f"{n:<14}{e:>7.3f}{(sp or {}).get('decode_ms_per_token', float('nan')):>8.2f}"
          f"{(df or {}).get('decode_ms_per_token', float('nan')):>8.2f}{(df or {}).get('tau', float('nan')):>6.2f}"
          f"{pipe:>13.2f}{x:>16.2f}{x / 1.55:>16.2f}")
print("pipe = projected ms/token of the pipeline; > 1 in the last column beats tensor parallel on 2 GPUs")
