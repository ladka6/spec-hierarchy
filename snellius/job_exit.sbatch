#!/usr/bin/env bash
#SBATCH --job-name=exit-heads
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=exit-heads-%j.out
# Early-exit heads on Qwen3-8B as the middle verifier (single GPU):
#   1) train lin / layer exit heads at k = 12, 18, 24, 30 (one epoch, same training data
#      as the lag fine-tune)
#   2) exp11: raw / lin / layer heads vs the bnb4 middle model on DFlash blocks
# Needs the training data from the lag fine-tune (hspec_lagft/data).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exit_$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ ! -f "$W/exit_heads.pt" ]; then
  python scripts/train_exit.py --data "$W/data" --out "$W/exit_heads.pt" 2>&1 | grep --line-buffered -v -i warn
fi
python scripts/exp11_exit.py --n 10 --heads "$W/exit_heads.pt" 2>&1 | grep --line-buffered -v -i warn
