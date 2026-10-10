#!/usr/bin/env bash
#SBATCH --job-name=shallow
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --output=shallow-%j.out
# Option 2: the target's first k layers + a trained adapter as the drafter's feature source.
# Four settings in parallel (exit layer k / adapter layers):
#   k10_a2  layers 1, 9 exact; 17, 25, 33 + final predicted by 2 layers init from target 10-11
#   k18_a2  layers 1, 9, 17 exact; 25, 33 + final predicted
#   k10_a4  as k10_a2 with 4 adapter layers
#   k10_a0  as k10_a2 with a per-position MLP instead of layers (no attention)
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_shallow
mkdir -p "$W"
ls "$D"/shard*.pt >/dev/null 2>&1 || { echo "no training data in $D"; exit 1; }
i=0
for C in "10 2" "18 2" "10 4" "10 0"; do
  set -- $C
  CUDA_VISIBLE_DEVICES=$i python scripts/train_shallow.py --exit $1 --adapter-layers $2 --data "$D" \
    --out "$W/k$1_a$2" ${SHALLOW_ARGS:-} > "$W/train_k$1_a$2.log" 2>&1 &
  i=$((i + 1))
done
wait
grep -h "step [0-9]*\]\|saved\|Error\|error" "$W"/train_*.log | grep --line-buffered -v -i warn
