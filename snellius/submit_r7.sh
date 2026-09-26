#!/usr/bin/env bash
# Round 7 as four single-GPU jobs.
#   bash snellius/submit_r7.sh
# When all four are done (squeue -u $USER is empty):
#   source ~/venvs/hspec/bin/activate
#   python scripts/aggregate_exp8.py results/r7_split results/r6_split
#   sed -n '/==/,$p' results/r7_split/bench_graph.log
#   grep -v -i warn results/r7_split/calib_ao4.log | sed -n '/==/,$p' | head -8
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$PWD/results/r7_split
mkdir -p "$OUT"
for p in 0 1 2 3; do sbatch snellius/job_r7_part.sbatch $p "$OUT"; done
echo "results -> $OUT"
