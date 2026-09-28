#!/usr/bin/env bash
# Round 9 as three single-GPU jobs.
#   bash snellius/submit_r9.sh
# When all three are done (squeue -u $USER is empty):
#   source ~/venvs/hspec/bin/activate
#   python scripts/aggregate_exp8.py results/r9_split results/r7_split results/r6_split | grep -E "lat_ms|ALL"
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$PWD/results/r9_split
mkdir -p "$OUT"
for p in 0 1 2; do sbatch snellius/job_r9_part.sbatch $p "$OUT"; done
echo "results -> $OUT"
