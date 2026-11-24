#!/usr/bin/env bash
# exp116 (Qwen3-8B, tap of layer 25, same data and anchors as exp112 / exp113).
#   bridge    which bridge over the first half: none / affine / + MLP / + self-attention / + attention to the real
#             prefix states, all on the band shape (layers 0,1 then 18..25)
#   slice     which 10 layers: the band (with 0, 2 or 4 shallow layers), the first 10, evenly spread, the plateau
#   small     the same question at 6 layers
set -euo pipefail
J=snellius/job_exp116.sbatch
sbatch --export=ALL,TAG=bridge,CONFIGS="band:2:none band:2:lin band:2:mlp band:2:attn band:2:ctx" $J
sbatch --export=ALL,TAG=slice,CONFIGS="band:0:mlp band:4:mlp first:0:mlp skip:0:mlp plateau:2:mlp" $J
sbatch --export=ALL,TAG=small,BUDGET=6,CONFIGS="band:2:mlp band:2:ctx first:0:mlp skip:0:mlp" $J
