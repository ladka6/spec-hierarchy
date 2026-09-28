#!/usr/bin/env bash
#SBATCH --job-name=lag-eval
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=lag-eval-%j.out
# Acceptance vs lag: original, control fine-tune, lag-tolerant fine-tune.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/lagft_$SLURM_JOB_ID"
python scripts/exp10_lag_eval.py --n 10 \
  --drafters orig=z-lab/Qwen3-8B-DFlash-b16 ctrl="$W/drafter_ctrl" lag16="$W/drafter_lag16" \
  2>&1 | grep -v -i warn
