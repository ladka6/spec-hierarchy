#!/usr/bin/env bash
#SBATCH --job-name=blkroute
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=blockroute-%j.out
# exp47: quality cost of causal block-shared expert sets (S experts per block of n tokens) on small MoE models
# (OLMoE-1B-7B, Qwen1.5-MoE-A2.7B; one GPU), plus natural block expert unions and the verify-cost cap per S.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/horizon"
mkdir -p "$HSPEC_RESULTS"
python scripts/exp47_blockroute.py --data "${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data" 2>&1 \
  | grep --line-buffered -v -i warn
