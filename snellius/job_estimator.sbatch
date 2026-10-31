#!/usr/bin/env bash
#SBATCH --job-name=estimator
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:30:00
#SBATCH --output=estimator-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/estimator"
mkdir -p "$HSPEC_RESULTS" "$HOME/hspec_ckpt"
CACHE=${CACHE:-$HOME/seqs.pt}
DATA=${DATA:-/scratch-shared/$USER/hspec_lagft/data}
CTX=${CTX:-layer}
SHALLOW=${SHALLOW:-mean}
STEPS=${STEPS:-8000}
F="grep --line-buffered -v -i warn"
# exp79: learned target-state estimator (L25 / L33 of drafted tokens), DFlash frozen.
#   E0 (filler check, no training):  sbatch --time=00:40:00 --export=ALL,STEPS=0 snellius/job_estimator.sbatch
#   run 1 (fused context):           sbatch --export=ALL,CTX=fused snellius/job_estimator.sbatch
#   run 2 (layer-matched context):   sbatch --export=ALL,CTX=layer snellius/job_estimator.sbatch
#   v2 (robust loss, normalised context, last real state): 
#     CTX=layer TAG=_v2 EXTRA="--loss huber --zclip 10 --ctx-norm --prev --kl-weight 3" sbatch --export=ALL snellius/job_estimator.sbatch
echo "[job] CTX=$CTX SHALLOW=$SHALLOW STEPS=$STEPS EXTRA=${EXTRA:-}"
python scripts/exp79_estimator.py --cache "$CACHE" --data "$DATA" --ctx "$CTX" --shallow "$SHALLOW" \
  --steps "$STEPS" --out "$HOME/hspec_ckpt/est_${CTX}_${SHALLOW}${TAG:-}.pt" ${EXTRA:-} 2>&1 | $F
