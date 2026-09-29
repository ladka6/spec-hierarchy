#!/usr/bin/env bash
#SBATCH --job-name=drift
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=drift-%j.out
# Drafter under target change: LoRA-fine-tune Qwen3-8B on one domain (or use the base model),
# then measure the base drafter's acceptance / speed on it and how far the target moved.
#   sbatch snellius/job_drift.sbatch NAME DOMAIN STEPS      (STEPS=0: the base model itself)
# Needs `pip install peft` in the venv (login node, once).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
NAME=$1; DOMAIN=$2; STEPS=$3
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${DRIFT_DIR:-/scratch-shared/$USER/hspec_drift}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/drift"
F="grep --line-buffered -v -i warn"
mkdir -p "$W" "$HSPEC_RESULTS"
if [ "$STEPS" = "0" ]; then
  VAR=Qwen/Qwen3-8B
else
  VAR="$W/qwen3-8b-$NAME"
  [ -f "$VAR/config.json" ] || python scripts/lora_target.py --domain $DOMAIN --steps $STEPS --out "$VAR" 2>&1 | $F
fi
python scripts/exp21_drift.py --variant "$VAR" --name $NAME --n 10 2>&1 | $F
