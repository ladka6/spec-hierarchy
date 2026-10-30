#!/usr/bin/env bash
#SBATCH --job-name=gen-vllm
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=gen-vllm-%j.out
# Drafter training data with vLLM on ONE GPU (budget-friendly replacement for job_lag_gen.sbatch).
# First, on the login node in the hspec venv:
#   python scripts/gen_traindata_vllm.py prompts --file /scratch-shared/$USER/prompts.pt
# then: sbatch snellius/job_gen_vllm.sbatch
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
python scripts/gen_traindata_vllm.py gen --file /scratch-shared/$USER/prompts.pt \
  --out /scratch-shared/$USER/hspec_lagft/data 2>&1 | grep --line-buffered -v -i warn
ls -la /scratch-shared/$USER/hspec_lagft/data
