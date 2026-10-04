#!/usr/bin/env bash
#SBATCH --job-name=vllm-cost
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=vllm-cost-%j.out
# Cost of the feature-refresh proxy in vLLM (CUDA graphs, Marlin int4): forward latency over q new
# tokens on a cached 1024-token prefix for bf16 Qwen3-8B vs 4-bit AWQ, batch 1.
# Needs Qwen/Qwen3-8B-AWQ downloaded into $HF_HOME on a login node first.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "$HOME/venvs/vllm/bin/activate"
# vLLM JIT-compiles kernels and needs nvcc: use the one pip installed into the venv, else a CUDA module
NVCC=$(find "$VIRTUAL_ENV" -type f -name nvcc -path "*bin*" 2>/dev/null | head -1)
if [ -n "$NVCC" ]; then
  export CUDA_HOME=$(dirname "$(dirname "$NVCC")")
else
  module load CUDA 2>/dev/null || module load CUDA/12.6.0 2>/dev/null
  export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
fi
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"; nvcc --version | tail -1
# FlashInfer JIT builds fail with the pip CUDA toolkit: use vLLM's precompiled kernels instead
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for M in Qwen/Qwen3-8B Qwen/Qwen3-8B-AWQ; do
  python scripts/bench_vllm.py --model $M --ctx 1024 --qs 1 16 32 64 128 --out results/bench_vllm_cost.json 2>&1 \
    | grep -v -i warn | tail -15
done
cat results/bench_vllm_cost.json
