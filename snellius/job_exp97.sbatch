#!/usr/bin/env bash
#SBATCH --job-name=decor97
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=decor97-%j.out
# exp97: are failures of alternative drafts (k-best paths, samples, other drafter checkpoints) correlated?
#   sbatch --export=ALL snellius/job_exp97.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp97"
mkdir -p "$HSPEC_RESULTS"
C=$HOME/hspec_ckpt
python scripts/exp97_decorrelation.py --models $C/drafter_infill $C/drafter_depth10 $C/drafter_lag16_fp32 ${EXTRA:-} 2>&1 \
  | grep --line-buffered -v -i warn
