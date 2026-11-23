#!/usr/bin/env bash
# exp113 (Qwen3-8B): shape of the shallow verify at a budget of 4 and 8 layers, same data and anchors as exp112.
#   first / skip / sandwich / attn   frozen layers, closed-form bridges ("fit") and trained bridges ("train")
#   firstlora / sandwichlora         + LoRA r64 on the kept layers (draft mode only)
#   CHECK=1 bash snellius/submit_exp113.sh    short smoke job over all four shapes
set -euo pipefail
J=snellius/job_exp113.sbatch
if [ "${CHECK:-0}" = 1 ]; then
  sbatch --time=00:40:00 --export=ALL,TAG=check,BUDGETS="4",STEPS=20,EXTRA="--fit-seqs 20 --anchors 60 --eval-every 10" $J
  exit 0
fi
for S in first skip sandwich attn; do
  sbatch --export=ALL,TAG=$S,SHAPES="$S" $J
done
sbatch --export=ALL,TAG=firstlora,SHAPES="first",METHODS="train",RANK=64 $J
sbatch --export=ALL,TAG=sandwichlora,SHAPES="sandwich",METHODS="train",RANK=64 $J
