#!/usr/bin/env bash
#SBATCH --job-name=pilot_scr
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=pilot_scratch-%j.out
# Rerun of the dflash_scratch arm only, with the fixed re-initialisation (step 0 tau should be ~1).
# Same data / steps / blocks / validation as job_pilot.sbatch; writes to hspec_pilot/dflash_scratch_v2.
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
python scripts/train_pilot.py --arch dflash_scratch --data "$D" --out "$W/dflash_scratch_v2" \
  ${PILOT_ARGS:-} > "$W/train_dflash_scratch_v2.log" 2>&1
grep -h "step [0-9]*\] \|saved\|Error\|error" "$W/train_dflash_scratch_v2.log" | grep -v -i warn
