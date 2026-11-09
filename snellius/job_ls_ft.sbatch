#!/usr/bin/env bash
#SBATCH --job-name=ls-ft
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=72:00:00
#SBATCH --output=ls-ft-%j.out
# Main baseline: the z-lab Llama-3.1 DFlash drafter fine-tuned on LayerSkip Llama3-8B greedy data, standard DFlash
# recipe (real target features, lag 0, block 10, all response positions as anchors, position-decayed CE).
# Data: job_gen_vllm.sbatch with STOP=$'\nQuestion:' into $W/data_ft, then scripts/clean_lsdata.py -> $W/data_ft_clean. Checkpoint per epoch; to continue a run:
#   RESUME=$W/drafter_ls_ft/epoch1 sbatch snellius/job_ls_ft.sbatch
# If it runs out of memory on the 40 GB A100: lower BLOCK_CHUNK, or sbatch -p gpu_h100.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${LS_DIR:-/scratch-shared/$USER/hspec_ls}
DATA=${LS_DATA:-$W/data_ft_clean}
OUT=${OUT:-$W/drafter_ls_ft}
DRAFT=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
RES=()
if [ -n "${RESUME:-}" ]; then DRAFT=$RESUME; RES=(--resume "$RESUME"); fi
python scripts/train_lag.py --target facebook/layerskip-llama3-8B --draft "$DRAFT" --data "$DATA" --out "$OUT" \
  --max-lag 0 --blocks 512 --block-chunk ${BLOCK_CHUNK:-128} --max-len 2048 \
  --batch ${BATCH:-8} --epochs ${EPOCHS:-3} --lr ${LR:-5e-5} --warmup 300 --gamma 0.82 --fp32 \
  --max-train ${MAX_TRAIN:-0} --val-seqs 200 --eval-every 2000 --save-epochs "${RES[@]}" \
  2>&1 | grep --line-buffered -v -i warn
