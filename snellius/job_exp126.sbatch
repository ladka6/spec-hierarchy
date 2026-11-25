#!/usr/bin/env bash
#SBATCH --job-name=exp126
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=exp126-%j.out
# exp126: what is in the trained code (no training): readouts, interventions, geometry on the exp125 checkpoints.
#   sbatch snellius/job_exp126.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
DATA=${DATA:-/scratch-shared/$USER/hspec_lagft/data}
C=${CKPT:-/scratch-shared/$USER/hspec_code}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp126_$SLURM_JOB_ID"
python scripts/exp126_code_content.py --data "$DATA" --ckpts ${ARMS:-$C/r0 $C/r16 $C/r64 $C/r256 $C/rreal} \
  ${EXTRA:-} 2>&1 | grep --line-buffered -v -i warn
cp -r "$HSPEC_RESULTS" $HOME/
