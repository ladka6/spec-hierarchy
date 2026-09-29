#!/usr/bin/env bash
#SBATCH --job-name=matrix
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --output=matrix-%j.out
# One drafter on every target (speed / acceptance only, official loops). Skips cells already done.
#   sbatch snellius/job_matrix.sbatch LABEL DRAFTER_PATH
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
LABEL=$1; DRAFTER=$2
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

for T in base math200 code200 chat1000; do
  if [ $T = base ]; then VAR=Qwen/Qwen3-8B; else VAR="$W/qwen3-8b-$T"; fi
  [ -f "$HSPEC_RESULTS/exp21_${T}@${LABEL}.json" ] && continue
  python scripts/exp21_drift.py --variant "$VAR" --draft "$DRAFTER" --name "${T}@${LABEL}" \
    --drafter-name $LABEL --no-drift --n 10 --budgets 128 2>&1 | $F
done
