#!/usr/bin/env bash
#SBATCH --job-name=middle4b
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=middle4b-%j.out
# Qwen3-4B (bf16 and AWQ) as the hedged middle model: DFlash speed with the 4B drafter, and
# disagreement / hedge statistics against the bf16 Qwen3-8B target (results/middle from job_middle).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
R="$SLURM_SUBMIT_DIR/results/middle"
source "$HOME/venvs/vllm/bin/activate"
NVCC=$(find "$VIRTUAL_ENV" -type f -name nvcc -path "*bin*" 2>/dev/null | head -1)
export CUDA_HOME=$(dirname "$(dirname "$NVCC")") PATH="$(dirname "$NVCC"):$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
Q='^\{|^ "|^\}|Error|error'
for M in Qwen/Qwen3-4B Qwen/Qwen3-4B-AWQ; do
  N=$(basename $M)
  echo "=== $M"
  python scripts/vllm_dflash.py --model $M --draft z-lab/Qwen3-4B-DFlash-b16 --spec --prompts "$R/prompts.json" \
    --out "$R/dflash_$N.json" 2>&1 | grep -E "$Q" | tail -4
  python scripts/vllm_middle.py hedge --model $M --traj "$R/traj.json" --out "$R/hedge_$N.json" 2>&1 \
    | grep -E 'Error|error' | tail -3
  python scripts/hedge_projection.py "$R" $N
done
