#!/usr/bin/env bash
#SBATCH --job-name=est80
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:30:00
#SBATCH --output=est80-%j.out
# exp80: cache-reading target-state estimator, DFlash and target frozen; final stop eval (all rounds).
#   sbatch --export=ALL snellius/job_exp80.sbatch
#   eval only:  STEPS=0 EXTRA="--load $HOME/hspec_ckpt/est80.pt" sbatch --time=00:40:00 --export=ALL snellius/job_exp80.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp80"
mkdir -p "$HSPEC_RESULTS" "$HOME/hspec_ckpt"
CACHE=${CACHE:-$HOME/seqs.pt}
DATA=${DATA:-/scratch-shared/$USER/hspec_lagft/data}
STEPS=${STEPS:-8000}
echo "[job] STEPS=$STEPS EXTRA=${EXTRA:-}"
python scripts/exp80_cache_estimator.py --cache "$CACHE" --data "$DATA" --steps "$STEPS" \
  --out "$HOME/hspec_ckpt/est80${TAG:-}.pt" ${EXTRA:-} 2>&1 | grep --line-buffered -v -i warn
