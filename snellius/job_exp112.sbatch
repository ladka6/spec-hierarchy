#!/usr/bin/env bash
#SBATCH --job-name=exp112
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=exp112-%j.out
# exp112: draft-only branch of the target's first k layers (LoRA or a full copy) trained through the frozen
# drafter, against the reader-only level of exp108b. One configuration per job, set with environment variables:
#   METHODS  (default "ridge mlp read lora")   EXITS (default "4 8")   KVMAP own|half|spread (default own)
#   RANK (default 64)   CEW / MSEW loss weights (default 1.0 / 0.1)   STEPS (default 3000)   TAG (name suffix)
#   sbatch --export=ALL,METHODS="lora",RANK=256,TAG=r256 snellius/job_exp112.sbatch
# All standard configurations: bash snellius/submit_exp112.sh
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
TAG=${TAG:-base}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp112_${TAG}_$SLURM_JOB_ID"
QWEN_DATA=${QWEN_DATA:-/scratch-shared/$USER/hspec_lagft/data}
if [ -d "$QWEN_DATA" ]; then QSRC="--data $QWEN_DATA"; else QSRC="--cache $HOME/seqs.pt"; fi
python scripts/exp112_draft_branch.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 $QSRC \
  --methods ${METHODS:-ridge mlp read lora} --exits ${EXITS:-4 8} --kv-map ${KVMAP:-own} --rank ${RANK:-64} \
  --ce-w ${CEW:-1.0} --mse-w ${MSEW:-0.1} --steps ${STEPS:-3000} ${EXTRA:-} --save exp112_${TAG}_qwen3 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
