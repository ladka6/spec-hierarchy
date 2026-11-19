#!/usr/bin/env bash
# exp112, second round (Qwen3-8B): the two levers the first round did not touch, at k = 4, 8 and 16.
#   draft    the drafter is adapted too (LoRA r32 on its linears) together with the reader
#   klread   fork resolver: the reader's final state trained with KL to the target's next-token distribution
#   kllora   the same with a draft-only LoRA inside the k layers (LayerSkip-style exit training, draft branch only)
#   k16      frozen-drafter references at k = 16 (k = 4 / 8 are in the first round)
#   CHECK=1 bash snellius/submit_exp112b.sh    short smoke job of the new code paths
set -euo pipefail
J=snellius/job_exp112.sbatch
if [ "${CHECK:-0}" = 1 ]; then
  sbatch --time=00:40:00 --export=ALL,TAG=check2,EXITS="4",METHODS="ridge read lora",STEPS=20,EXTRA="--draft-rank 8 --kl-w 1.0 --fit-tokens 60000 --mlp-steps 500 --anchors 60" $J
  exit 0
fi
sbatch --export=ALL,TAG=draft,METHODS="read",EXITS="4 8 16",EXTRA="--draft-rank 32" $J
sbatch --export=ALL,TAG=klread,METHODS="read",EXITS="4 8 16",EXTRA="--kl-w 1.0" $J
sbatch --export=ALL,TAG=kllora,METHODS="lora",EXITS="4 8 16",EXTRA="--kl-w 1.0 --lr-branch 5e-5" $J
sbatch --export=ALL,TAG=k16,METHODS="ridge mlp read",EXITS="16" $J
