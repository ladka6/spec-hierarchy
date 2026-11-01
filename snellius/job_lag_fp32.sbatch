#!/usr/bin/env bash
#SBATCH --job-name=lag-fp32
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:30:00
#SBATCH --output=lag-fp32-%j.out
# Lag-tolerant DFlash (lags 0..16) on the vLLM training set, fp32 master weights; saved to $HOME (scratch purges).
# Floor for chained drafting: drafted tokens as plain inputs, no features (FlexDraft / Packed-SS style).
#   sbatch snellius/job_lag_fp32.sbatch
#   depth lag (real shallow taps for lagged positions):
#   OUT=$HOME/hspec_ckpt/drafter_depth2 EXTRA="--depth-exit 2" sbatch --export=ALL snellius/job_lag_fp32.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
DATA=${DATA:-/scratch-shared/$USER/hspec_lagft/data}
OUT=${OUT:-$HOME/hspec_ckpt/drafter_lag16_fp32}
echo "[job] OUT=$OUT EXTRA=${EXTRA:-}"
python scripts/train_lag.py --data "$DATA" --out "$OUT" --max-lag 16 --fp32 --optim adamw --lr 1e-4 ${EXTRA:-} \
  --epochs ${EPOCHS:-1} --eval-every 1000 2>&1 | grep --line-buffered -v -i warn
