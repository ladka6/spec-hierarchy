#!/usr/bin/env bash
#SBATCH --job-name=pilot_fo
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=pilot_featonly-%j.out
# PDraft features-only: blocks see only the anchor, their mask slots and the injected target features.
# Same data / steps / validation as the other pilot arms. Does the 0.6B use the 8B states when forced to?
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
python scripts/train_pilot.py --arch pdraft_featonly --data "$D" --out "$W/pdraft_featonly" \
  ${PILOT_ARGS:-} > "$W/train_pdraft_featonly.log" 2>&1
grep -h "step [0-9]*\] \|saved\|Error\|error" "$W/train_pdraft_featonly.log" | grep -v -i warn
