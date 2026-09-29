#!/usr/bin/env bash
# Is "drafters under target change" worth a thesis? Five single-GPU jobs: the base target and
# four LoRA fine-tunes of Qwen3-8B (math light / math / code / chat), each evaluated with the
# drafter trained for the base target.
#   bash snellius/submit_drift.sh
# When all are done:
#   for f in $(ls -t drift-*.out | head -5); do sed -n '/base-target drafter/,$p' $f; done
set -euo pipefail
cd "$(dirname "$0")/.."
for spec in "base none 0" "math200 math 200" "math1000 math 1000" "code1000 code 1000" "chat1000 chat 1000"; do
  set -- $spec
  sbatch --job-name=drift-$1 snellius/job_drift.sbatch $1 $2 $3
done
