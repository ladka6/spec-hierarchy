#!/usr/bin/env bash
#SBATCH --job-name=depth85
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=depth85-%j.out
# exp85: chained round with real shallow features for in-flight positions (depth-lag drafters).
#   DRAFT=$HOME/hspec_ckpt/drafter_depth2 EXTRA="--depth-exit 2" sbatch --export=ALL snellius/job_exp85.sbatch
#   sbatch --export=ALL snellius/job_exp85.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp85"
mkdir -p "$HSPEC_RESULTS"
python scripts/exp85_depth_eval.py --cache "${CACHE:-$HOME/seqs.pt}" --draft "${DRAFT:-z-lab/Qwen3-8B-DFlash-b16}" --stops ${STOPS:-600} ${EXTRA:-} 2>&1 \
  | grep --line-buffered -v -i warn
