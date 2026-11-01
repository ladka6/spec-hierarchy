#!/usr/bin/env bash
#SBATCH --job-name=infill91
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=infill91-%j.out
# exp91: infill from a later anchor with the frozen DFlash: oracle gain G(m), wrong-anchor harm, confidence islands,
# retrieval anchors (vs plain prompt lookup).
#   sbatch --export=ALL snellius/job_exp91.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp91"
mkdir -p "$HSPEC_RESULTS"
python scripts/exp91_infill.py ${EXTRA:-} 2>&1 | grep --line-buffered -v -i warn
