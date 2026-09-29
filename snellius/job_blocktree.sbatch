#!/usr/bin/env bash
#SBATCH --job-name=blocktree
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=blocktree-%j.out
# exp15: block trees (DDTree + lag-drafted continuation blocks) vs DDTree at equal node budget,
# Qwen3-8B, original DFlash drafter + the lag16 drafter from the lag fine-tune.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/blocktree_$SLURM_JOB_ID"
python scripts/exp15_blocktree.py --n 10 --lag-draft "${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/drafter_lag16" \
  2>&1 | grep --line-buffered -v -i warn
