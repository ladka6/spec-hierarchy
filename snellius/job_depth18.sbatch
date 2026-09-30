#!/usr/bin/env bash
#SBATCH --job-name=depth18
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=16:00:00
#SBATCH --output=depth18-%j.out
# Depth-lag drafter for Qwen3-8B (recent positions keep only the features of layers < 18, i.e. what
# the first GPU of a 2-GPU pipeline has computed), then exp25 with suspected tokens getting those
# shallow features: does pipelined verification work when the next round sees the lower half?
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
W=/scratch-shared/$USER/hspec_lagft
F="grep --line-buffered -v -i warn"
[ -f "$W/drafter_depth18/config.json" ] || python scripts/train_lag.py --target Qwen/Qwen3-8B \
  --draft z-lab/Qwen3-8B-DFlash-b16 --data "$W/data" --out "$W/drafter_depth18" \
  --depth-exit 18 --max-lag 24 --p-zero 0.25 2>&1 | $F
python scripts/exp25_suspect.py --depth-exit 18 --subs orig=z-lab/Qwen3-8B-DFlash-b16:token \
  lag16="$W/drafter_lag16:token" orig_d=z-lab/Qwen3-8B-DFlash-b16:depth depth18="$W/drafter_depth18:depth" 2>&1 | $F
