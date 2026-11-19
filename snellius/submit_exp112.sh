#!/usr/bin/env bash
# exp112, all standard configurations (Qwen3-8B, frozen drafter). Run from the repository root.
#   CHECK=1 bash snellius/submit_exp112.sh    only the short smoke job (start-up check + a few steps)
# The full copy at k = 8 needs ~45 GB (fp32 weights + Adam): it goes to H100_PART (default gpu_h100).
set -euo pipefail
J=snellius/job_exp112.sbatch
if [ "${CHECK:-0}" = 1 ]; then
  sbatch --time=00:40:00 --export=ALL,TAG=check,EXITS="4",METHODS="ridge read lora copy",STEPS=20,EXTRA="--fit-tokens 60000 --mlp-steps 500 --anchors 60" $J
  exit 0
fi
sbatch --export=ALL,TAG=base,METHODS="ridge mlp read lora" $J                       # reader level vs LoRA r64
sbatch --export=ALL,TAG=r256,METHODS="lora",RANK=256 $J                             # capacity: rank
sbatch --export=ALL,TAG=copy4,METHODS="copy",EXITS="4" $J                           # capacity: full layers
# copy at k = 8 only on request (COPY8=1): decide after copy4 vs lora at k = 4
[ "${COPY8:-0}" = 1 ] && sbatch --partition="${H100_PART:-gpu_h100}" --export=ALL,TAG=copy8,METHODS="copy",EXITS="8" $J || true
sbatch --export=ALL,TAG=half,METHODS="lora",KVMAP=half $J                           # deep verified cache
sbatch --export=ALL,TAG=spread,METHODS="lora",KVMAP=spread $J
sbatch --export=ALL,TAG=ceonly,METHODS="lora",MSEW=0 $J                             # objective
sbatch --export=ALL,TAG=mseonly,METHODS="lora",CEW=0,MSEW=1 $J
