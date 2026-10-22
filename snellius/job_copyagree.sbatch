#!/usr/bin/env bash
#SBATCH --job-name=copyagree
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:40:00
#SBATCH --output=copyagree-%j.out
# exp66: per-token greedy agreement q of cheap copies with the target, run lengths, conditional curve.
# Copies as arguments (any load_mid spec), e.g.: sbatch job_copyagree.sbatch bnb4:Qwen/Qwen3-8B rtn3:Qwen/Qwen3-8B
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/copyagree"
mkdir -p "$HSPEC_RESULTS"
W=/scratch-shared/$USER/hspec_copytrain
python scripts/exp66_copyagree.py --teacher $W/teacher.pt --copies ${@:-bnb4:Qwen/Qwen3-8B rtn3:Qwen/Qwen3-8B} 2>&1 \
  | grep --line-buffered -v -i warn
