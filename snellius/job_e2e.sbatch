#!/usr/bin/env bash
#SBATCH --job-name=e2e
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=e2e-%j.out
# exp28: feature-refreshed decoding end to end (greedy) vs plain DFlash; losslessness test first.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/e2e"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
L=/scratch-shared/$USER/hspec_lagft/drafter_lag16
python tests/test_refresh.py 2>&1 | grep -v -i warn | tail -3
python scripts/exp28_e2e.py --copy bnb4:Qwen/Qwen3-8B --ddtree 64 2>&1 | $F
