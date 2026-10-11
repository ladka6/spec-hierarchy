#!/usr/bin/env bash
#SBATCH --job-name=pilot_kv
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=03:00:00
#SBATCH --output=pilot_kv-%j.out
# PDraft with the target features injected DFlash-style (gated extra K/V in every layer) instead of
# added to the input embeddings (where pdraft == pdraft_nofeat, i.e. the features went unused).
#   pdraft_kv        same budget as the pilot (2 epochs), directly comparable to the other arms
#   pdraft_kv_e4     4 epochs: does the curve keep rising with more data passes?
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
CUDA_VISIBLE_DEVICES=0 python scripts/train_pilot.py --arch pdraft_kv --data "$D" --out "$W/pdraft_kv" \
  > "$W/train_pdraft_kv.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_pilot.py --arch pdraft_kv --epochs 4 --data "$D" --out "$W/pdraft_kv_e4" \
  > "$W/train_pdraft_kv_e4.log" 2>&1 &
wait
grep -h "step [0-9]*\] \|gate\|saved\|Error\|error" "$W"/train_pdraft_kv*.log | grep -v -i warn | grep -v "/" 
grep -h "Error\|error" "$W"/train_pdraft_kv*.log | head
