#!/usr/bin/env bash
#SBATCH --job-name=lag-train
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=06:00:00
#SBATCH --output=lag-train-%j.out
# Two fine-tunes in parallel on the same data: lag-tolerant (lags 0..16) and control (lag 0).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}
CUDA_VISIBLE_DEVICES=0 python scripts/train_lag.py --data "$W/data" --out "$W/drafter_lag16" --max-lag 16 \
  > "$W/train_lag16.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_lag.py --data "$W/data" --out "$W/drafter_ctrl" --max-lag 0 \
  > "$W/train_ctrl.log" 2>&1 &
wait
grep -h "val\|saved" "$W"/train_*.log | grep -v -i warn
