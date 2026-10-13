#!/usr/bin/env bash
#SBATCH --job-name=wide
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=wide-%j.out
# exp56: one wide refresh step (several corrections / doubt positions / stop restarts, one tree) instead of deep loops.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/wide"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
W=/scratch-shared/$USER/hspec_copytrain
mkdir -p $W
python scripts/exp56_wide.py --copy bnb4:Qwen/Qwen3-8B --temp 0.0 --out exp56_bnb4_t0 2>&1 | $F
python scripts/exp56_wide.py --copy bnb4:Qwen/Qwen3-8B --temp 1.0 --out exp56_bnb4_t1 2>&1 | $F
python scripts/exp56_wide.py --copy lora:$W/copy_rtn2 --temp 0.0 --out exp56_rtn2trained_t0 2>&1 | $F
