#!/usr/bin/env bash
#SBATCH --job-name=forks99
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=forks99-%j.out
# exp99: is the cheap target copy right at DFlash's forks?
#   sbatch --export=ALL snellius/job_exp99.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp99"
mkdir -p "$HSPEC_RESULTS"
C=$HOME/hspec_ckpt/copyrole
python scripts/exp99_copyforks.py \
  --copies ${COPIES:-bnb4:Qwen/Qwen3-8B lora:$C/copy_lm_4bit rtn3:Qwen/Qwen3-8B lora:$C/copy_lm lora:$C/copy_both rtn2:Qwen/Qwen3-8B} \
  ${EXTRA:-} 2>&1 | grep --line-buffered -v -i warn
