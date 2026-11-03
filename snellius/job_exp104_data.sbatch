#!/usr/bin/env bash
#SBATCH --job-name=exp104-data
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=exp104-data-%j.out
# exp104 training data: Qwen3-8B greedy responses + the target's continuation after its 2nd choice at forks (vLLM).
# Needs the prompts file from gen_traindata_vllm.py (login node, hspec venv), if it is not there yet:
#   python scripts/gen_traindata_vllm.py prompts --file /scratch-shared/$USER/prompts.pt
# Smoke test: LIMIT=200 OUT=/scratch-shared/$USER/forkdata_smoke sbatch snellius/job_exp104_data.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "$HOME/venvs/vllm/bin/activate"
NVCC=$(find "$VIRTUAL_ENV" -type f -name nvcc -path "*bin*" 2>/dev/null | head -1)
if [ -n "$NVCC" ]; then
  export CUDA_HOME=$(dirname "$(dirname "$NVCC")")
else
  module load CUDA 2>/dev/null || module load CUDA/12.6.0 2>/dev/null
  export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
fi
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-/scratch-shared/$USER/forkdata}
python scripts/gen_forkdata_vllm.py --file "${PROMPTS:-/scratch-shared/$USER/prompts.pt}" --out "$OUT" \
  --limit "${LIMIT:-0}" ${EXTRA:-} 2>&1 | grep --line-buffered -v -i warn
ls -la "$OUT"
