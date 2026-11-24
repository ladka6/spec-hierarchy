#!/usr/bin/env bash
#SBATCH --job-name=exp116
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=exp116-%j.out
# exp116: the drafter's 3/4-depth tap for in-flight tokens from a sub-stack of the target: which bridge over the
# skipped layers, and which slice of layers at the same budget. One list of configurations per job:
#   CONFIGS "shape:a:kind ..." (band / first / skip / plateau; none / lin / mlp / attn / ctx)
#   BUDGET (default 10 layers)   STEPS (default 2000)   TAG   EXTRA
# Standard runs: bash snellius/submit_exp116.sh
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
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp116_${TAG}_$SLURM_JOB_ID"
QWEN_DATA=${QWEN_DATA:-/scratch-shared/$USER/hspec_lagft/data}
if [ -d "$QWEN_DATA" ]; then QSRC="--data $QWEN_DATA"; else QSRC="--cache $HOME/seqs.pt"; fi
python scripts/exp116_band_bridge.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 $QSRC \
  --configs ${CONFIGS:-band:2:lin band:2:mlp band:2:ctx} --budget ${BUDGET:-10} --steps ${STEPS:-2000} \
  ${EXTRA:-} --save exp116_${TAG}_qwen3 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
