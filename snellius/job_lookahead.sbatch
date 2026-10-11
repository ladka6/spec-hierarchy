#!/usr/bin/env bash
#SBATCH --job-name=lookahead
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:45:00
#SBATCH --output=lookahead-%j.out
# exp39: does the frozen target compute useful lookahead at mask slots? Probe A (base) vs B (+ slot states)
# vs O (true-token states, upper bound) for future tokens x_(s+k); zero-shot slot logits too.
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
python scripts/exp39_lookahead.py --data "${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data" ${LA_ARGS:-} 2>&1 | grep --line-buffered -v -i warn
