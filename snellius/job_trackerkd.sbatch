#!/usr/bin/env bash
#SBATCH --job-name=trackerkd
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:30:00
#SBATCH --output=trackerkd-%j.out
# Distil the 0.6B tracker on the 8B target (forward KL), then: exp32 single-block quality and the T=0 horizon curve with the distilled tracker.
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
K=/scratch-shared/$USER/tracker_kd_06b
python scripts/train_tracker_kd.py --data "$D" --out "$K" 2>&1 | $F
if [ -f "$K/config.json" ]; then
  python scripts/exp32_smallmap.py --small "$K" --data "$D" --out exp32_smallmap_kd06b 2>&1 | $F
  python scripts/exp29_iterate.py --small "$K" --data "$D" --ks 1 2 3 4 5 6 --temp 0.0 --out exp34_kd06b_ridge_t0.0 2>&1 | $F
fi
