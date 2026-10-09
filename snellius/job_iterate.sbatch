#!/usr/bin/env bash
#SBATCH --job-name=iterate
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=iterate-%j.out
# exp29: k refresh iterations (drafter -> 4-bit copy -> drafter ...) before one target verify; chain vs tree verify; cost model.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/iterate"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
L=/scratch-shared/$USER/hspec_lagft/drafter_lag16
for T in 0.0 1.0; do
  python scripts/exp29_iterate.py --copy bnb4:Qwen/Qwen3-8B --ks 1 2 3 4 --temp $T --out exp29_iterate_t$T 2>&1 | $F
done
