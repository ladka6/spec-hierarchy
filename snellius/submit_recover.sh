#!/usr/bin/env bash
# Re-alignment and forgetting for three fine-tuned targets (one GPU each).
#   bash snellius/submit_recover.sh
# When done:
#   for f in $(ls -t recover-*.out | head -3); do grep "\[final\]" $f; sed -n '/base-target drafter/,/^variant  kl/p' $f; done
set -euo pipefail
cd "$(dirname "$0")/.."
for spec in "math200 math 200" "chat1000 chat 1000" "code200 code 200"; do
  set -- $spec
  sbatch --job-name=recover-$1 snellius/job_recover.sbatch $1 $2 $3
done
