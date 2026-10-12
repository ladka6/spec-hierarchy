#!/usr/bin/env bash
#SBATCH --job-name=xwaste
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=01:30:00
#SBATCH --output=xwaste-%j.out
# exp45: share of MoE verification (Qwen3-30B-A3B) spent on DFlash tokens that get rejected; drafted vs true
# text expert scatter, new experts by position and by drafter confidence.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/horizon"
python scripts/exp45_expert_waste.py --data "${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data" ${XW_ARGS:-} \
  2>&1 | grep --line-buffered -v -i warn
