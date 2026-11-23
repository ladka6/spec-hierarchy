#!/usr/bin/env bash
# Tap importance of the original DFlash drafters on their own targets, one job per pair (same 600 prompts).
#   bash snellius/submit_taps.sh            Qwen3-8B and LLaMA-3.1-8B-Instruct
#   GEMMA=1 bash snellius/submit_taps.sh    also gemma4-12B-it (target and drafter must be in the HF cache)
set -euo pipefail
J=snellius/job_taps.sbatch
sbatch --export=ALL,TARGET=Qwen/Qwen3-8B,DRAFT=z-lab/Qwen3-8B-DFlash-b16,TAG=qwen3_8b $J
sbatch --export=ALL,TARGET=meta-llama/Llama-3.1-8B-Instruct,DRAFT=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat,TAG=llama31_8b $J
if [ "${GEMMA:-0}" = 1 ]; then
  sbatch --export=ALL,TARGET=${GEMMA_TARGET:-google/gemma-4-12b-it},DRAFT=z-lab/gemma4-12B-it-DFlash,TAG=gemma4_12b $J
fi
