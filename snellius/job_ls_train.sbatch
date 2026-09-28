#!/usr/bin/env bash
#SBATCH --job-name=ls-train
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=ls-train-%j.out
# Adapt the Llama-3.1-8B-Instruct DFlash drafter to LayerSkip Llama3-8B (standard blocks, lag 0).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LS_DIR:-/scratch-shared/$USER/hspec_ls}
TGT=facebook/layerskip-llama3-8B
DRAFT0=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
python scripts/train_lag.py --target $TGT --draft $DRAFT0 --data "$W/data" --out "$W/drafter_ls" \
  --max-lag 0 --lr 5e-5 2>&1 | grep --line-buffered -v -i warn
