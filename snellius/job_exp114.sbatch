#!/usr/bin/env bash
#SBATCH --job-name=exp114
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=exp114-%j.out
# exp114: small LoRA on the whole backbone, trained to decide early (exit loss at EXITS) while staying the same
# model (LAM * KL to the original output). Drift against the decision of the shallow verify at the drafter's
# first error. One configuration per job:
#   RANK (default 32; 0 = frozen backbone, readers only)   LAM (default 1.0)   EXITS (default "4 8")
#   STEPS (default 4000)   TAG (name suffix)   EXTRA (other flags)
#   sbatch --export=ALL,LAM=0.1,TAG=lam0.1 snellius/job_exp114.sbatch
# All standard configurations: bash snellius/submit_exp114.sh
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
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp114_${TAG}_$SLURM_JOB_ID"
QWEN_DATA=${QWEN_DATA:-/scratch-shared/$USER/hspec_lagft/data}
if [ -d "$QWEN_DATA" ]; then QSRC="--data $QWEN_DATA"; else QSRC="--cache $HOME/seqs.pt"; fi
python scripts/exp114_draftable_target.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 $QSRC \
  --rank ${RANK:-32} --lam ${LAM:-1.0} --exits ${EXITS:-4 8} --steps ${STEPS:-4000} ${EXTRA:-} \
  --out /scratch-shared/$USER/exp114/${TAG}.pt --save exp114_${TAG}_qwen3 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
