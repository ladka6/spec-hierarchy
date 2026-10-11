#!/usr/bin/env bash
#SBATCH --job-name=pilot
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --output=pilot-%j.out
# Pilot at equal budget (same data / steps / blocks / validation), learning curves every 250 steps:
#   dflash_scratch   DFlash drafter architecture, re-initialised (fair small-budget baseline)
#   pdraft           block diffusion drafter on pretrained Qwen3-0.6B (28 layers) + target features
#   pdraft_nofeat    same without target features (ablation)
#   pdraft_L12       first 12 layers of Qwen3-0.6B (cheaper)
# Step 0 of every run also reports the official, fully trained DFlash drafter on the same anchors.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_pilot
mkdir -p "$W"
ls "$D"/shard*.pt >/dev/null 2>&1 || { echo "no training data in $D"; exit 1; }
i=0
for C in "dflash_scratch 0" "pdraft 0" "pdraft_nofeat 0" "pdraft 12"; do
  set -- $C
  N=$1; [ "$2" != 0 ] && N=${N}_L$2
  CUDA_VISIBLE_DEVICES=$i python scripts/train_pilot.py --arch $1 --n-layers $2 --data "$D" --out "$W/$N" \
    ${PILOT_ARGS:-} > "$W/train_$N.log" 2>&1 &
  i=$((i + 1))
done
wait
grep -h "step [0-9]*\] \|saved\|Error\|error" "$W"/train_*.log | grep --line-buffered -v -i warn
