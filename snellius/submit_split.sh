#!/usr/bin/env bash
# Submit round 6 as four single-GPU jobs (they start sooner than one 4-GPU job).
#   bash snellius/submit_split.sh
# When all four are done:
#   python scripts/aggregate_exp8.py results/r6_split   (activate the venv first)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$PWD/results/r6_split
mkdir -p "$OUT"
for p in 0 1 2 3; do sbatch snellius/job_r6_part.sbatch $p "$OUT"; done
echo "results -> $OUT"
