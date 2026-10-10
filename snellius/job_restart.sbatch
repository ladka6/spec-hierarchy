#!/usr/bin/env bash
#SBATCH --job-name=restart
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=restart-%j.out
# Restart points from the drafter instead of the small corrector: 0.6B ridge features as context, restart at the first drafted token with drafter confidence < p (nothing inserted), or keep the whole chain and extend its end (T=0).
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
F="grep --line-buffered -v -i warn"
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
for R in dconf:0.5 dconf:0.7 dconf:0.9 append; do
  python scripts/exp29_iterate.py --small Qwen/Qwen3-0.6B --data "$D" --ks 1 2 3 4 5 6 --temp 0.0 --restart $R \
    --out exp34_q06_ridge_restart${R/:/}_t0.0 2>&1 | $F
done
