#!/usr/bin/env bash
# Submit through submit_r9.sh; that launcher sets the GPU/CPU allocation and log path.
# Runs both target policies sequentially in the same allocation, with separate JSONs.
set -euo pipefail
if (( $# != 3 )); then
  echo "Use: bash snellius/submit_r9.sh [2|3|both]" >&2
  exit 2
fi
REPO=$1
OUT=$2
GPUS=$3
case $GPUS in
  2) DRAFT_DEVICE=cuda:1 ;;
  3) DRAFT_DEVICE=cuda:2 ;;
  *) echo "Expected 2 or 3 GPUs" >&2; exit 2 ;;
esac
cd "$REPO"
RUN_DIR="$OUT/${GPUS}gpu_${SLURM_JOB_ID:?must run through sbatch}"
mkdir -p "$RUN_DIR"

# Match the existing Snellius setup.sh environment.
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

{
  echo "job=$SLURM_JOB_ID placement=${GPUS}gpu host=$(hostname)"
  echo "source_snapshot=$REPO"
  echo "commit=$(cat .hspec_commit)"
  module list 2>&1
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} | tee "$RUN_DIR/environment.log"

# Single process, three stage workers; SLURM exposes logical devices cuda:0,1[,2].
srun --ntasks=1 --gpus-per-task="$GPUS" python -c \
  'import sys, torch, transformers, dflash; n=int(sys.argv[1]); print("torch", torch.__version__, "transformers", transformers.__version__, "GPUs", torch.cuda.device_count()); assert torch.cuda.device_count() == n, "Unexpected visible GPU count"' "$GPUS"

COMMON=(--target-device cuda:0 --mid-device cuda:1 --draft-device "$DRAFT_DEVICE"
        --reserves 0 1 2 4 8 --datasets gsm8k math500 humaneval mt-bench
        --n "${HSPEC_N:-5}" --max-new "${HSPEC_MAX_NEW:-256}"
        --repeats "${HSPEC_REPEATS:-3}" --seed "${HSPEC_SEED:-0}"
        --fork-threshold "${HSPEC_FORK_THRESHOLD:-0.6}"
        --max-ahead "${HSPEC_MAX_AHEAD:-64}"
        --target-window "${HSPEC_TARGET_WINDOW:-32}"
        --target-verification "${HSPEC_TARGET_VERIFICATION:-block}" --trace)
for POLICY in fallback wait; do
  if [[ $POLICY == fallback ]]; then
    POLICY_ARG=--target-fallback
  else
    POLICY_ARG=--no-target-fallback
  fi
  echo "Starting ${GPUS}-GPU run, policy=$POLICY"
  srun --ntasks=1 --gpus-per-task="$GPUS" python -u scripts/exp9_reserve.py \
    "${COMMON[@]}" "$POLICY_ARG" --output "$RUN_DIR/$POLICY.json" \
    2>&1 | tee "$RUN_DIR/$POLICY.log"
done
echo "Finished: $RUN_DIR"
