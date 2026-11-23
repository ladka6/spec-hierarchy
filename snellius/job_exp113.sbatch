#!/usr/bin/env bash
#SBATCH --job-name=exp113
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=exp113-%j.out
# exp113: shape of the shallow verify at a fixed budget (first / skip / sandwich / attn), scored by f = cut and fix
# at the drafter's first error. One configuration per job, set with environment variables:
#   SHAPES (default "first skip sandwich attn")   BUDGETS (default "4 8")   METHODS (default "fit train")
#   RANK LoRA rank on the kept layers (default 0)   STEPS (default 3000)   TAG (name suffix)   EXTRA (other flags)
#   sbatch --export=ALL,SHAPES="sandwich",TAG=sandwich snellius/job_exp113.sbatch
# All standard configurations: bash snellius/submit_exp113.sh
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
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp113_${TAG}_$SLURM_JOB_ID"
QWEN_DATA=${QWEN_DATA:-/scratch-shared/$USER/hspec_lagft/data}
if [ -d "$QWEN_DATA" ]; then QSRC="--data $QWEN_DATA"; else QSRC="--cache $HOME/seqs.pt"; fi
python scripts/exp113_shallow_verify.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 $QSRC \
  --shapes ${SHAPES:-first skip sandwich attn} --budgets ${BUDGETS:-4 8} --methods ${METHODS:-fit train} \
  --rank ${RANK:-0} --steps ${STEPS:-3000} ${EXTRA:-} --save exp113_${TAG}_qwen3 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
