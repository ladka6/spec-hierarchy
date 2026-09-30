#!/usr/bin/env bash
#SBATCH --job-name=ceiling-tp
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=04:00:00
#SBATCH --output=ceiling-tp-%j.out
# exp24 with the target tensor-parallel over 2 GPUs (drafter replicated): the 2-GPU DDTree baseline
# and the async ceiling on top of it. Compare ms_tok with the 1-GPU run (results/check/exp24_ceiling.json).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/check"
mkdir -p "$HSPEC_RESULTS"
torchrun --standalone --nproc_per_node 2 scripts/exp24_ceiling.py --tp --out exp24_ceiling_tp2 2>&1 \
  | grep --line-buffered -v -i warn
