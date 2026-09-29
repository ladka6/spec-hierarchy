#!/usr/bin/env bash
#SBATCH --job-name=bounds
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=bounds-%j.out
# Reference drafters for the continual-target study (needs the recover-* runs' data):
#   sbatch snellius/job_bounds.sbatch ctrl    base drafter fine-tuned on base-target data, same size
#                                             and recipe as the re-aligned drafters (data-matched control)
#   sbatch snellius/job_bounds.sbatch joint   one drafter trained on all targets' data (base + 3 variants)
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
MODE=$1
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${DRIFT_DIR:-/scratch-shared/$USER/hspec_drift}
BASEDATA=${BASEDATA:-/scratch-shared/$USER/hspec_lagft/data}
F="grep --line-buffered -v -i warn"
BASE=Qwen/Qwen3-8B
DRAFT0=z-lab/Qwen3-8B-DFlash-b16
N=1968   # training sequences per target in the re-aligned drafters (2000 prompts - 32 val)

if [ "$MODE" = ctrl ]; then
  python scripts/train_lag.py --target $BASE --draft $DRAFT0 --data "$BASEDATA" --out "$W/drafter_ctrl" \
    --max-lag 0 --epochs 1 --val-seqs 32 --max-train $N 2>&1 | $F
elif [ "$MODE" = joint ]; then
  python scripts/train_joint.py --draft $DRAFT0 --out "$W/drafter_joint" --max-per-target $N --rounds 4 \
    --pairs $BASE="$BASEDATA" "$W/qwen3-8b-math200=$W/data_math200" \
            "$W/qwen3-8b-code200=$W/data_code200" "$W/qwen3-8b-chat1000=$W/data_chat1000" 2>&1 | $F
else
  echo "unknown mode $MODE"; exit 1
fi
