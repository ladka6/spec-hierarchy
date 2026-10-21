#!/usr/bin/env bash
#SBATCH --job-name=benchrefresh
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --output=benchrefresh-%j.out
# Measured cost of the cheap refresh components (shallow target layers bf16 / 4-bit, ridge maps, drafter, restart
# call, 4-bit copy) against the target's verify at 16-64 tokens, CUDA graphs, batch 1. CTX="1024 4096" for several.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/benchrefresh"
mkdir -p "$HSPEC_RESULTS"
for C in ${CTX:-1024}; do
  python scripts/bench_refresh.py --ctx $C --out bench_refresh_ctx$C 2>&1 | grep --line-buffered -v -i warn
done
