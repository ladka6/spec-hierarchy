#!/usr/bin/env bash
# Verify the drift numbers (error bars + target task accuracy) and measure the async ceiling.
#   bash snellius/submit_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
W=${DRIFT_DIR:-/scratch-shared/$USER/hspec_drift}
for T in base math200; do
  sbatch --job-name=check-base-$T snellius/job_check.sbatch base z-lab/Qwen3-8B-DFlash-b16 $T
  sbatch --job-name=check-strong-$T snellius/job_check.sbatch strong "$W/drafter_math200_strong" $T
done
sbatch snellius/job_taskacc.sbatch
sbatch snellius/job_ceiling.sbatch
