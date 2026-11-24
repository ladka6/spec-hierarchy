#!/usr/bin/env bash
#SBATCH --job-name=exp117
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=exp117-%j.out
# exp117: the token at the cut (no training). Does the redraft repeat the wrong token, what does excluding it give,
# how often is the target's token among the first 2 / 3 / 5 choices, split by how sure the target is.
#   sbatch snellius/job_exp117.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp117_$SLURM_JOB_ID"
QWEN_DATA=${QWEN_DATA:-/scratch-shared/$USER/hspec_lagft/data}
if [ -d "$QWEN_DATA" ]; then QSRC="--data $QWEN_DATA"; else QSRC="--cache $HOME/seqs.pt"; fi
python scripts/exp117_cut_token.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 $QSRC \
  --anchors ${ANCHORS:-2000} ${EXTRA:-} --save exp117_qwen3 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
