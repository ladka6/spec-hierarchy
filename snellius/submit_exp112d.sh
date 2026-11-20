#!/usr/bin/env bash
# exp112, fourth round (Qwen3-8B): chain of two blocks without a fork head. The cut point comes from the drafter's
# own confidence, the drafter redrafts from the cut with the branch's features (second table of the output).
#   chainread   ridge / mlp references and the reader trained through the drafter
#   chainlora   draft-only LoRA inside the k layers (+ KL, the best feature setting so far)
#   CHECK=1 bash snellius/submit_exp112d.sh    short smoke job
set -euo pipefail
J=snellius/job_exp112.sbatch
if [ "${CHECK:-0}" = 1 ]; then
  sbatch --time=00:40:00 --export=ALL,TAG=check4,EXITS="4",METHODS="ridge read",STEPS=20,EXTRA="--fit-tokens 60000 --mlp-steps 500 --anchors 60" $J
  exit 0
fi
sbatch --export=ALL,TAG=chainread,METHODS="ridge mlp read",EXITS="4 8" $J
sbatch --export=ALL,TAG=chainlora,METHODS="lora",EXITS="4 8",EXTRA="--kl-w 1.0 --lr-branch 5e-5" $J
