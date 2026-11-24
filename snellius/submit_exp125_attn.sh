#!/usr/bin/env bash
# exp125 step 2b: the band with its deep layers attention-only (MLP -> rank-256 stand-in), same protocol as the
# path arms (lag drafter, 1 epoch, block loss only). Compare with path10 (the same band with full layers, 5.5 ms).
#   bash snellius/submit_exp125_attn.sh
sub() { CODE="$1" NAME="$2" DRAFT="${3:-$HOME/hspec_ckpt/drafter_lag16_fp32}" EPOCHS="${4:-1}" EXTRA="${5:-}" \
        sbatch --export=ALL snellius/job_exp125.sbatch; }
sub path:0,1,21-28a   path10a                 # 2 full + 8 attention-only: about 4 layers of weights
sub path:0,1,13-28a   path18a "" 1 "--block-chunk 8"   # 2 full + 16 attention-only: about 6 layers of weights
