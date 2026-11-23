#!/usr/bin/env bash
# exp114 (Qwen3-8B): how far the target has to move to become draftable. Same data and anchors as exp112 / exp113.
#   frozen            readers only, backbone untouched (zero drift, the reference)
#   lam10 / lam1 / lam0.1   backbone LoRA r32, exit loss at 4 and 8 layers, drift weight 10 / 1 / 0.1
#   lam1long          lam 1 with three times the training
set -euo pipefail
J=snellius/job_exp114.sbatch
sbatch --export=ALL,TAG=frozen,RANK=0 $J
for LAM in 10 1 0.1; do
  sbatch --export=ALL,TAG=lam$LAM,LAM=$LAM $J
done
sbatch --time=08:00:00 --export=ALL,TAG=lam1long,LAM=1,STEPS=12000 $J
