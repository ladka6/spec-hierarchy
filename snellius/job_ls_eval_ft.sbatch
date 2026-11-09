#!/usr/bin/env bash
#SBATCH --job-name=ls-evalft
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=ls-evalft-%j.out
# DFlash on LayerSkip Llama3-8B, batch 1, greedy: the original z-lab drafter vs a fine-tuned checkpoint.
# tau and tau_clean (cut at a new Question: turn or loop onset) on gsm8k / math500 / humaneval / mt-bench.
#   sbatch snellius/job_ls_eval_ft.sbatch /scratch-shared/$USER/hspec_ls/drafter_ls_ft/epoch1 ep1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=$1; NAME=${2:-ft}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/ls_eval_ft"
mkdir -p "$HSPEC_RESULTS"
TGT=facebook/layerskip-llama3-8B
F="grep --line-buffered -v -i warn"
[ -f "$HSPEC_RESULTS/exp8_orig.json" ] || python scripts/exp8_algos.py --target $TGT \
  --draft z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat --n 20 --latencies-ms 0 --tag orig --suffix @orig \
  --configs base:dflash 2>&1 | $F
python scripts/exp8_algos.py --target $TGT --draft "$CKPT" --n 20 --latencies-ms 0 --tag "$NAME" --suffix "@$NAME" \
  --configs base:dflash 2>&1 | $F
python scripts/aggregate_exp8.py "$HSPEC_RESULTS"
