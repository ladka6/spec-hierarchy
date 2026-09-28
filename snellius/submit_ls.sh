#!/usr/bin/env bash
# LayerSkip Llama3-8B: generate data -> adapt the DFlash drafter -> evaluate, as a job chain.
# Run snellius/setup_layerskip.sh on a login node first (downloads the model and the drafter).
#   bash snellius/submit_ls.sh
set -euo pipefail
cd "$(dirname "$0")/.."
gen=$(sbatch --parsable snellius/job_ls_gen.sbatch)
train=$(sbatch --parsable --dependency=afterok:$gen snellius/job_ls_train.sbatch)
ev=$(sbatch --parsable --dependency=afterok:$train snellius/job_ls_eval.sbatch)
echo "submitted: gen $gen -> train $train -> eval $ev"
echo "logs: ls-gen-$gen.out, ls-train-$train.out, ls-eval-$ev.out (detail in /scratch-shared/$USER/hspec_ls)"
