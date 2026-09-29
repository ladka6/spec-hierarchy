#!/usr/bin/env bash
#SBATCH --job-name=ls-official
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=ls-official-%j.out
# LayerSkip Llama3-8B with the adapted drafter:
#   1) full / split pass cost up to 512 tokens (when does a bigger tree start to cost more)
#   2) exp17: early-exit pruning of big DDTrees (offline acceptance)
#   3) exp16: the official DDTree / DFlash code, tree budgets 16..512, vs our DDTree
# Needs `pip install loguru` in the venv (login node, once).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LS_DIR:-/scratch-shared/$USER/hspec_ls}
TGT=facebook/layerskip-llama3-8B
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/lsofficial_$SLURM_JOB_ID"
F="grep --line-buffered -v -i warn"

python scripts/bench_split.py --target $TGT --exits 16 20 --qs 1 32 64 128 256 512 2>&1 | $F
python scripts/exp17_prune.py --target $TGT --draft "$W/drafter_ls" --n 10 2>&1 | $F
python scripts/exp16_official.py --target $TGT --draft "$W/drafter_ls" --n 10 --tag ls 2>&1 | $F
