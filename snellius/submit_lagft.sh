#!/usr/bin/env bash
# Lag-tolerant drafter: generate data -> fine-tune (lag16 + control) -> evaluate, as a chain
# of dependent jobs. Run snellius/setup_lagft.sh on a login node first (once).
#   bash snellius/submit_lagft.sh
set -euo pipefail
cd "$(dirname "$0")/.."
gen=$(sbatch --parsable snellius/job_lag_gen.sbatch)
train=$(sbatch --parsable --dependency=afterok:$gen snellius/job_lag_train.sbatch)
ev=$(sbatch --parsable --dependency=afterok:$train snellius/job_lag_eval.sbatch)
echo "submitted: gen $gen -> train $train -> eval $ev"
echo "logs: lag-gen-$gen.out, lag-train-$train.out, lag-eval-$ev.out (training detail in /scratch-shared/$USER/hspec_lagft)"
