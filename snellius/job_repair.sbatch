#!/usr/bin/env bash
#SBATCH --job-name=repair
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=repair-%j.out
# exp20: how much of a draft survives a boundary correction (bounds verified-prefix repair), Qwen3-8B.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/repair_$SLURM_JOB_ID"
python scripts/exp20_repair.py --n 10 --lag-draft "$W/drafter_lag16" 2>&1 | grep --line-buffered -v -i warn
