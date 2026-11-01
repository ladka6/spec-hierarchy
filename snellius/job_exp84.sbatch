#!/usr/bin/env bash
#SBATCH --job-name=sub84
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=sub84-%j.out
# exp84: how much of L25 / L33 DFlash uses (PCA / Fisher / random subspaces of the real features; no training).
#   sbatch --export=ALL snellius/job_exp84.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp84"
mkdir -p "$HSPEC_RESULTS"
python scripts/exp84_subspace.py --cache "${CACHE:-$HOME/seqs.pt}" --stops ${STOPS:-600} ${EXTRA:-} 2>&1 \
  | grep --line-buffered -v -i warn
