#!/usr/bin/env bash
#SBATCH --job-name=ls-lora
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=ls-lora-%j.out
# LoRA post-training of the LayerSkip model (merged checkpoint), for exp22.
#   sbatch snellius/job_ls_lora.sbatch NAME DOMAIN STEPS
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
mkdir -p "$W"
[ -f "$W/ls-$NAME/config.json" ] || python scripts/lora_target.py --target facebook/layerskip-llama3-8B \
  --domain $DOMAIN --steps $STEPS --out "$W/ls-$NAME" 2>&1 | grep --line-buffered -v -i warn
