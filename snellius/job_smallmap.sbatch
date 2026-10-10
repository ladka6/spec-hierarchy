#!/usr/bin/env bash
#SBATCH --job-name=smallmap
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=smallmap-%j.out
# exp32: small same-family model (Qwen3-0.6B / 1.7B) + closed-form linear map -> the 8B verifier's DFlash features on the frontier; drafter tau at lags 4/8/16.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/smallmap"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
for S in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B; do
  python scripts/exp32_smallmap.py --small $S --data "$D" 2>&1 | $F
done
