#!/usr/bin/env bash
#SBATCH --job-name=select93
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=select93-%j.out
# exp93: selection through DFlash's top-16 lattice: target width (DDTree-style), small LM, copies at 4/3/2 bits
# (ranking among 16 vs generating), oracle.
#   sbatch --export=ALL snellius/job_exp93.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp93"
mkdir -p "$HSPEC_RESULTS"
C=$HOME/hspec_ckpt/copyrole
python scripts/exp93_selection.py \
  --scorers rtn4:Qwen/Qwen3-8B lora:$C/copy_lm_4bit rtn3:Qwen/Qwen3-8B lora:$C/copy_lm lora:$C/copy_both hf:Qwen/Qwen3-0.6B \
  --costs 4.5 4.5 3.6 2.5 2.5 1.0 ${EXTRA:-} 2>&1 | grep --line-buffered -v -i warn
