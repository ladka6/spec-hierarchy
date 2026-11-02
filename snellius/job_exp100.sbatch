#!/usr/bin/env bash
#SBATCH --job-name=rerank100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=rerank100-%j.out
# exp100: can a cheap copy resolve DFlash's forks by re-ranking DFlash's own top-k?
#   sbatch --export=ALL snellius/job_exp100.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp100"
mkdir -p "$HSPEC_RESULTS"
C=$HOME/hspec_ckpt/copyrole
python scripts/exp100_rerank.py \
  --copies ${COPIES:-bnb4:Qwen/Qwen3-8B lora:$C/copy_lm_4bit rtn3:Qwen/Qwen3-8B lora:$C/copy_lm lora:$C/copy_both rtn2:Qwen/Qwen3-8B} \
  ${EXTRA:-} 2>&1 | grep --line-buffered -v -i warn
