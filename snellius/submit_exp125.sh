#!/usr/bin/env bash
# one job per arm of exp125:  bash snellius/submit_exp125.sh [arms...]
for c in ${@:-0 16 64 256 real}; do
  sbatch --export=ALL,CODE=$c snellius/job_exp125.sbatch
done
