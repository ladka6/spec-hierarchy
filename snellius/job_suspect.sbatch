#!/usr/bin/env bash
#SBATCH --job-name=suspect
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=suspect-%j.out
# exp25: suspected-prefix pipelining offline (q = how often round t+1 can start early).
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
python scripts/exp25_suspect.py --lag-draft /scratch-shared/$USER/hspec_lagft/drafter_lag16 2>&1 \
  | grep --line-buffered -v -i warn
