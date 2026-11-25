#!/usr/bin/env bash
# exp125 step 3: ADAPTED path layers (LoRA, in-flight pass only) + drafter, one stage from the original DFlash
# drafter, block loss only. Baselines of the same protocol already run: tokens 4.54, frozen path 0,1,27,28 4.86
# (tau at lag 8).   bash snellius/submit_exp125_lora.sh
sub() { CODE="$1" NAME="$2" DRAFT=z-lab/Qwen3-8B-DFlash-b16 EPOCHS=2 EXTRA="--lr 1e-4 --block-chunk 8 $3" \
        sbatch --time=${4:-10:00:00} --export=ALL snellius/job_exp125.sbatch; }
sub real              real_onestage      ""               04:00:00   # ceiling of the one-stage protocol
sub path:0-3          first4_lora        "--lora-rank 64"             # early-exit shape, 2.2 ms
sub path:0-7          first8_lora        "--lora-rank 64" 14:00:00    # early-exit shape, 4.3 ms
sub path:0,1,27,28    path4_band_lora    "--lora-rank 64"             # the frozen version gave 4.86
sub path:1,9,17,25,33 taps5_lora         "--lora-rank 64" 12:00:00    # the exp81 path, 2.7 ms
