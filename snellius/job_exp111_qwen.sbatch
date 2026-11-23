#!/usr/bin/env bash
#SBATCH --job-name=exp111q
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=exp111q-%j.out
# exp111 on the plain Qwen3-8B pair (original z-lab drafter, taps L1 / L9 / L17 / L25 / L33): which taps the drafter
# depends on, overall and at the last 8 / 16 / 32 context positions. Same script as the LayerSkip-Llama run.
# The group rows are labelled "L1 only" / "L1+L8" by the script; for Qwen they are L1 and L1+L9.
#   sbatch snellius/job_exp111_qwen.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp111"
F="grep --line-buffered -v -i warn"
python scripts/exp111_tap_importance.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
  --data ${QWEN_DATA:-/scratch-shared/$USER/hspec_lagft/data} --ms 8 16 32 --save exp111_qwen3_8b 2>&1 | $F
