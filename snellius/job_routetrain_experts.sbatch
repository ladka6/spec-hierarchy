#!/usr/bin/env bash
#SBATCH --job-name=rtrainx
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:30:00
#SBATCH --output=routetrain-experts-%j.out
# exp49: routers + one LoRA per expert (r=8, ~59M params) under causal block-shared routing, n=16.
# Usage: sbatch snellius/job_routetrain_experts.sbatch [S]   (default S=16; e.g. 24)
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
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_moe
S=${1:-16}
python scripts/exp49_routetrain.py --mode router+experts --S $S --n 16 --steps 1000 --data "$D" \
  --out "$W/routetrain_router_experts_S$S" \
  2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "eval\|trainable\|done\|rror"
