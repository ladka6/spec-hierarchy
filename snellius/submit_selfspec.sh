#!/usr/bin/env bash
# LoRA fine-tunes of the LayerSkip model, then exp22 on all of them (one job, after the LoRAs).
#   bash snellius/submit_selfspec.sh
# When done:  sed -n '/best setting/,$p' $(ls -t selfspec-*.out | head -1)
set -euo pipefail
cd "$(dirname "$0")/.."
deps=""
for spec in "math200 math 200" "math1000 math 1000" "code200 code 200" "chat1000 chat 1000"; do
  set -- $spec
  j=$(sbatch --parsable --job-name=ls-lora-$1 snellius/job_ls_lora.sbatch $1 $2 $3)
  deps="$deps:$j"
done
sbatch --dependency=afterany$deps snellius/job_selfspec.sbatch
