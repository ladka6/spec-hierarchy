#!/usr/bin/env bash
# Ideas A and B on Qwen3-8B: re-drafting (one job) and dependency heads (two training jobs in
# parallel, then the evaluation once both are done).
#   bash snellius/submit_ab.sh
set -euo pipefail
cd "$(dirname "$0")/.."
a=$(sbatch --parsable snellius/job_redraft.sbatch)
t1=$(sbatch --parsable snellius/job_dephead_train.sbatch tf)
t2=$(sbatch --parsable snellius/job_dephead_train.sbatch tree)
ev=$(sbatch --parsable --dependency=afterok:$t1:$t2 snellius/job_dephead_eval.sbatch)
echo "submitted: redraft $a | dephead train tf $t1, tree $t2 -> eval $ev"
echo "logs: redraft-$a.out, dephead-train-$t1.out, dephead-train-$t2.out, dephead-eval-$ev.out"
