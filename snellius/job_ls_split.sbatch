#!/usr/bin/env bash
#SBATCH --job-name=ls-split
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=ls-split-%j.out
# exp14: one-GPU split verification (early exit at layer 16/20 + upper pass) vs DFlash and
# DDTree on LayerSkip Llama3-8B, real eager wall clock and simulated clock.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LS_DIR:-/scratch-shared/$USER/hspec_ls}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/lssplit_$SLURM_JOB_ID"
python scripts/exp14_split.py --target facebook/layerskip-llama3-8B --n 10 \
  --base-draft "$W/drafter_ls" --split-draft "$W/drafter_ls_depth16" \
  --configs base:dflash base:ddtree-32 base:ddtree-64 \
            split-k16-t0.7-b32-m4 split-k16-t0.7-b32-m8 split-k16-t0.7-b64-m8 \
            split-k16-t0.5-b32-m4 split-k16-t0.9-b32-m4 split-k16-t0.7-b0-m4 \
            split-k20-t0.7-b32-m4 split-k16-t0.7-b32-m4@base \
  2>&1 | grep --line-buffered -v -i warn
