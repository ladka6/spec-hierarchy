#!/usr/bin/env bash
# exp125 step 2: speculative features from a sparse path through the backbone, trained with the drafter on the
# block loss only. One job per arm; CODE goes through the environment because it contains commas.
#   bash snellius/submit_exp125_path.sh
sub() { CODE="$1" NAME="$2" DRAFT="${3:-$HOME/hspec_ckpt/drafter_lag16_fp32}" EPOCHS="${4:-1}" EXTRA="${5:-}" \
        sbatch --export=ALL snellius/job_exp125.sbatch; }
sub state:29          state29                 # ceiling: the real state the paths stand in for
sub path:0,1,28       path3                   # 3 layers, 1.6 ms
sub path:0,1,27,28    path4_band              # 4 layers, 2.2 ms: two shallow + two deep
sub path:0,9,19,28    path4_spread            # 4 layers, 2.2 ms: spread over the depth
sub path:0,1,21-28    path10                  # 10 layers, 5.5 ms: the exp116 band, for reference
# what joint training adds: the same 4-layer path with the drafter frozen
sub path:0,1,27,28    path4_band_frozen "" 1 "--freeze-draft"
# one stage instead of two: from the original DFlash drafter, the total training of both stages
sub path:0,1,27,28    path4_band_onestage z-lab/Qwen3-8B-DFlash-b16 2 "--lr 1e-4"
sub 0                 tokens_onestage     z-lab/Qwen3-8B-DFlash-b16 2 "--lr 1e-4"
