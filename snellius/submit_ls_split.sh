#!/usr/bin/env bash
# Same chain as submit_ls.sh, but data generation as four single-GPU jobs (start sooner
# than one 4-GPU job). Training starts when all four shards are done.
#   bash snellius/submit_ls_split.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ids=()
for k in 0 1 2 3; do ids+=($(sbatch --parsable snellius/job_ls_gen_part.sbatch $k)); done
dep=$(IFS=:; echo "${ids[*]}")
train=$(sbatch --parsable --dependency=afterok:$dep snellius/job_ls_train.sbatch)
ev=$(sbatch --parsable --dependency=afterok:$train snellius/job_ls_eval.sbatch)
echo "submitted: gen ${ids[*]} -> train $train -> eval $ev"
