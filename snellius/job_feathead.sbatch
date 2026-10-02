#!/usr/bin/env bash
#SBATCH --job-name=feathead
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --output=feathead-%j.out
# Feature prediction feasibility: train a feature head for the frozen DFlash drafter, then exp25
# with the chained block drafted from predicted context vectors (pred), against no features
# (token), real shallow features (depth18) and real features (fresh, the upper bound).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/feathead"
mkdir -p "$HSPEC_RESULTS"
W=/scratch-shared/$USER/hspec_lagft
D=z-lab/Qwen3-8B-DFlash-b16
F="grep --line-buffered -v -i warn"
[ -f "$W/feathead.pt" ] || python scripts/train_feathead.py --draft $D --data "$W/data" --out "$W/feathead.pt" \
  --max-train 2000 --epochs 1 2>&1 | $F
python scripts/exp25_suspect.py --ms 1 2 4 --subs orig=$D:token fresh=$D:fresh "pred=$D:pred@$W/feathead.pt" \
  depth18="$W/drafter_depth18:depth" 2>&1 | $F
