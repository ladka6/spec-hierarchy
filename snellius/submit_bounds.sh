#!/usr/bin/env bash
# Drafter x target matrix for the continual-target study. Run after the recover-* jobs finished.
#   bash snellius/submit_bounds.sh
# When all done:  python scripts/matrix.py results/matrix
set -euo pipefail
cd "$(dirname "$0")/.."
W=${DRIFT_DIR:-/scratch-shared/$USER/hspec_drift}
for n in math200 code200 chat1000; do
  [ -f "$W/drafter_$n/config.json" ] || { echo "missing $W/drafter_$n (recover-$n not finished)"; exit 1; }
done
# rows that need no training: no update, and the three specialists
sbatch --job-name=mx-base snellius/job_matrix.sbatch base z-lab/Qwen3-8B-DFlash-b16
for n in math200 code200 chat1000; do
  sbatch --job-name=mx-$n snellius/job_matrix.sbatch $n "$W/drafter_$n"
done
# rows that need training first
c=$(sbatch --parsable --job-name=bounds-ctrl snellius/job_bounds.sbatch ctrl)
j=$(sbatch --parsable --job-name=bounds-joint snellius/job_bounds.sbatch joint)
sbatch --dependency=afterok:$c --job-name=mx-ctrl snellius/job_matrix.sbatch ctrl "$W/drafter_ctrl"
sbatch --dependency=afterok:$j --job-name=mx-joint snellius/job_matrix.sbatch joint "$W/drafter_joint"
