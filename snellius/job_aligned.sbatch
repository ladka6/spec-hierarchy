#!/usr/bin/env bash
#SBATCH --job-name=aligned
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:30:00
#SBATCH --output=aligned-%j.out
# exp62: aligned looked-up features. When the output repeats earlier text, restart the drafter after the
# looked-up tokens with the target's features from the earlier occurrence (no copy pass). Against the copy refresh,
# prompt lookup, and an oracle with real features; recall and history at 8K / 16K.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/aligned"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
python scripts/exp62_aligned.py --ctx 8192 16384 --settings recall history --match ngram attn \
  --bases recent-1024-orig attn-1024-compact --n 6 --anchors 16 --out exp62_aligned 2>&1 | $F
