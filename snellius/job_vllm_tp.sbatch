#!/usr/bin/env bash
#SBATCH --job-name=vllm-tp
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=02:00:00
#SBATCH --output=vllm-tp-%j.out
# Batch-1 decode latency of Qwen3-8B in vLLM with tensor parallel 1 vs 2 (CUDA graphs, custom
# all-reduce): the realistic speedup a second GPU gives, to rescale our verify time.
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
for TP in 1 2; do
  for OUT in 1 512; do    # output 1 = prefill only, so decode ms/token = (lat512 - lat1) / 511
    echo "=== TP=$TP output_len=$OUT"
    vllm bench latency --model Qwen/Qwen3-8B --tensor-parallel-size $TP --batch-size 1 \
      --input-len 256 --output-len $OUT --num-iters-warmup 3 --num-iters 10 --max-model-len 2048 \
      --gpu-memory-utilization 0.85 2>&1 | grep -E "Avg latency|percentile|Error|error" | head -5
  done
done
