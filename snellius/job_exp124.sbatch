#!/usr/bin/env bash
#SBATCH --job-name=exp124
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=exp124-%j.out
# exp124: chain depth (no training): B = 1 .. 8 stages, oracle and confidence cuts, plain / soft / masked token at
# the cut, real and no features for the in-flight tokens; tokens per stage and speed with the measured timings.
#   sbatch snellius/job_exp124.sbatch
#   another drafter: sbatch --export=ALL,DRAFT=/path,TAG=lag snellius/job_exp124.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
QWEN_DATA=${QWEN_DATA:-/scratch-shared/$USER/hspec_lagft/data}
if [ -d "$QWEN_DATA" ]; then QSRC="--data $QWEN_DATA"; else QSRC="--cache $HOME/seqs.pt"; fi
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp124_$SLURM_JOB_ID"
python scripts/exp124_chain_depth.py --target Qwen/Qwen3-8B --draft "${DRAFT:-z-lab/Qwen3-8B-DFlash-b16}" $QSRC \
  --anchors ${ANCHORS:-1000} ${EXTRA:-} --save exp124_${TAG:-orig}_qwen3 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
