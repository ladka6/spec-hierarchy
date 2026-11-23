#!/usr/bin/env bash
# exp115 on the three original z-lab pairs (data from job_taps.sbatch).
set -euo pipefail
J=snellius/job_content.sbatch
sbatch --export=ALL,TARGET=Qwen/Qwen3-8B,DRAFT=z-lab/Qwen3-8B-DFlash-b16,TAG=qwen3_8b $J
sbatch --export=ALL,TARGET=meta-llama/Llama-3.1-8B-Instruct,DRAFT=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat,TAG=llama31_8b $J
sbatch --export=ALL,TARGET=google/gemma-4-12B-it,DRAFT=z-lab/gemma4-12B-it-DFlash,TAG=gemma4_12b $J
