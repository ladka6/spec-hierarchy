#!/usr/bin/env bash
# exp112, third round (Qwen3-8B): fork reranker over the drafter's top-16 candidates on the draft branch.
#   rankread    ridge / mlp references (new fork columns, untrained) and the reader trained with the ranking loss
#   ranklora    draft-only LoRA inside the k layers + ranking loss
#   ranklora10k the same at k = 4 with 10k steps
#   CHECK=1 bash snellius/submit_exp112c.sh    short smoke job of the new code path
set -euo pipefail
J=snellius/job_exp112.sbatch
if [ "${CHECK:-0}" = 1 ]; then
  sbatch --time=00:40:00 --export=ALL,TAG=check3,EXITS="4",METHODS="ridge read lora",STEPS=20,EXTRA="--rank-w 1.0 --fit-tokens 60000 --mlp-steps 500 --anchors 60" $J
  exit 0
fi
sbatch --export=ALL,TAG=rankread,METHODS="ridge mlp read",EXITS="4 8",EXTRA="--rank-w 1.0" $J
sbatch --export=ALL,TAG=ranklora,METHODS="lora",EXITS="4 8",EXTRA="--rank-w 1.0 --lr-branch 5e-5" $J
sbatch --time=08:00:00 --export=ALL,TAG=ranklora10k,METHODS="lora",EXITS="4",STEPS=10000,EXTRA="--rank-w 1.0 --lr-branch 5e-5 --eval-every 250 --patience 12" $J
