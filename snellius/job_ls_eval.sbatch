#!/usr/bin/env bash
#SBATCH --job-name=ls-eval
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=ls-eval-%j.out
# LayerSkip Llama3-8B with the adapted drafter:
#   exp11  raw early exits (plain and confidence-gated) vs a bnb4 copy as the middle verifier
#   exp8   DFlash and DDTree baselines on the same model (adapted and original drafter)
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
DRAFT0=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/ls_$SLURM_JOB_ID"
python scripts/exp11_exit.py --target $TGT --draft "$W/drafter_ls" --mid bnb4:$TGT --n 10 \
  --layers 8 12 16 20 24 --gates 0.5 0.7 0.9 2>&1 | grep --line-buffered -v -i warn
python scripts/exp8_algos.py --target $TGT --draft "$W/drafter_ls" --n 20 --latencies-ms 0 --tag ls \
  --configs base:dflash base:ddtree-32 base:ddtree-64 2>&1 | grep --line-buffered -v -i warn
python scripts/exp8_algos.py --target $TGT --draft $DRAFT0 --n 20 --latencies-ms 0 --tag ls_orig --suffix @orig \
  --configs base:dflash 2>&1 | grep --line-buffered -v -i warn
python scripts/aggregate_exp8.py "$HSPEC_RESULTS" | grep -E "lat_ms|ALL"
