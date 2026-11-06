#!/usr/bin/env bash
#SBATCH --job-name=exp108b
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:30:00
#SBATCH --output=exp108b-%j.out
# exp108b: exp108 without the data limit (600k fit tokens, dropout, early stop) plus the drafter-through reader
# (MLP fine-tuned with the frozen drafter's block CE). Same reports as exp108, so rows compare directly.
#   1) Qwen3-8B, native drafter    2) LayerSkip Llama3-8B control (skip it with ONLY_QWEN=1)
#   sbatch snellius/job_exp108b.sbatch      or      sbatch --export=ALL,ONLY_QWEN=1 snellius/job_exp108b.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp108b_$SLURM_JOB_ID"
QWEN_DATA=${QWEN_DATA:-/scratch-shared/$USER/hspec_lagft/data}
if [ -d "$QWEN_DATA" ]; then QSRC="--data $QWEN_DATA"; else QSRC="--cache $HOME/seqs.pt"; fi
python scripts/exp108b_nonlinear_lift.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
  $QSRC --exits 4 8 --save exp108b_qwen3 2>&1 | $F
[ "${ONLY_QWEN:-0}" = 1 ] || python scripts/exp108b_nonlinear_lift.py --target facebook/layerskip-llama3-8B \
  --draft z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat --data $HOME/hspec_llama/data_layerskip-llama3-8B \
  --exits 4 8 --save exp108b_layerskip 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
