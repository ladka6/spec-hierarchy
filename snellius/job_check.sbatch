#!/usr/bin/env bash
#SBATCH --job-name=check
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --output=check-%j.out
# Error bars for the drift numbers: one drafter on one target, 40 prompts per dataset, 2 prompt seeds.
#   sbatch snellius/job_check.sbatch DRAFTER_LABEL DRAFTER_PATH TARGET_LABEL
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${DRIFT_DIR:-/scratch-shared/$USER/hspec_drift}
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/check"
mkdir -p "$HSPEC_RESULTS"
DL=$1; DP=$2; T=$3
if [ $T = base ]; then V=Qwen/Qwen3-8B; else V="$W/qwen3-8b-$T"; fi
for s in 0 1; do
  python scripts/exp21_drift.py --variant "$V" --draft "$DP" --name "${T}@${DL}_s$s" --drafter-name $DL \
    --no-drift --n 40 --seed $s --budgets 128 2>&1 | $F
done
