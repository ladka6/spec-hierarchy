#!/usr/bin/env bash
#SBATCH --job-name=strong
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=strong-%j.out
# Is the tau gap after target fine-tuning recoverable? Strong re-alignment of the base drafter:
# all ~5.6k training prompts (the recover run used 2k), more epochs, higher lr. Then speed on
# the fine-tuned target and on the base target (forgetting), as cells of the matrix.
#   sbatch snellius/job_strong.sbatch NAME EPOCHS LR LABEL      e.g. math200 3 1e-4 strong
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
NAME=$1; EPOCHS=$2; LR=$3; LABEL=$4
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${DRIFT_DIR:-/scratch-shared/$USER/hspec_drift}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/matrix"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
VAR="$W/qwen3-8b-$NAME"
D="$W/data_$NAME"
OUT="$W/drafter_${NAME}_$LABEL"

# the remaining prompts, added to the existing 2k-prompt data
[ -f "$D/shard0_ext.pt" ] || python scripts/gen_traindata.py --target "$VAR" --skip-prompts 2000 --tag _ext \
  --out "$D" 2>&1 | $F
[ -f "$OUT/config.json" ] || python scripts/train_lag.py --target "$VAR" --draft z-lab/Qwen3-8B-DFlash-b16 \
  --data "$D" --out "$OUT" --max-lag 0 --epochs $EPOCHS --lr $LR --val-seqs 32 2>&1 | $F
for T in $NAME base; do
  if [ $T = base ]; then V=Qwen/Qwen3-8B; else V="$VAR"; fi
  python scripts/exp21_drift.py --variant "$V" --draft "$OUT" --name "${T}@${NAME}_$LABEL" \
    --drafter-name ${NAME}_$LABEL --no-drift --n 10 --budgets 128 2>&1 | $F
done
