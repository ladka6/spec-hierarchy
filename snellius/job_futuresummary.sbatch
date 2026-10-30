#!/usr/bin/env bash
#SBATCH --job-name=futuresum
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=futuresum-%j.out
# exp75 at scale: can DFlash learn to use a NEW input (a summary of the block's true future) with a real training
# budget? The Colab runs (420 sequences, 800 steps) could not tell: even an oracle bag of the future tokens gave ~0.
# Train on the gen_traindata shards (thousands of sequences), evaluate on the same held-out sequences as on Colab.
#   sbatch snellius/job_futuresummary.sbatch                       # defaults below
#   CACHE=/path/seqs.pt DATA=/path/shards STEPS=6000 sbatch snellius/job_futuresummary.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/futuresummary"
mkdir -p "$HSPEC_RESULTS"
CACHE=${CACHE:-/scratch-shared/$USER/seqs.pt}
DATA=${DATA:-/scratch-shared/$USER/hspec_lagft/data}
STEPS=${STEPS:-6000}
F="grep --line-buffered -v -i warn"
# 1. oracle channel check: a bag of the true future tokens (must help if the drafter can learn the channel)
python scripts/exp75_futuresummary.py --cache "$CACHE" --data "$DATA" --steps "$STEPS" \
  --inject kv --chunks 4 --source tokens --ds 0 64 2>&1 | $F
# 2. the real question: a compact summary of the target's true future deep states
python scripts/exp75_futuresummary.py --cache "$CACHE" --data "$DATA" --steps "$STEPS" \
  --inject kv --chunks 4 --source deep --ds 16 64 2>&1 | $F
