#!/usr/bin/env bash
# Everything queued while Snellius was down (25 November). Short, no-training jobs first.
#   bash snellius/submit_after_maintenance.sh
set -u
C=/scratch-shared/$USER/hspec_code
L=$HOME/hspec_ckpt/drafter_lag16_fp32
sub() { CODE="$1" NAME="$2" DRAFT="${3:-$L}" EPOCHS="${4:-1}" EXTRA="${5:-}" \
        sbatch --time=${6:-10:00:00} --export=ALL snellius/job_exp125.sbatch; }

# 1. no training
sbatch snellius/job_exp126.sbatch                                   # what the code carries; can the drafter produce it
sbatch --export=ALL,DRAFT=$L,TAG=lag_fork snellius/job_exp123.sbatch   # fork top-k as a baseline number

# 2. producers that run no target layer (same protocol as the code arms: from the lag drafter, 1 epoch)
sub self      self64          "" 1 "--from-ckpt $C/r64 --block-chunk 8"
sub self      self256         "" 1 "--from-ckpt $C/r256 --block-chunk 8"
sub self      self64_nodistill "" 1 "--from-ckpt $C/r64 --block-chunk 8 --distill 0"
sub xattn:29  xattn29         "" 1 "--block-chunk 8"

# 3. adapted (LoRA) path layers, one stage from the original drafter
bash snellius/submit_exp125_lora.sh
