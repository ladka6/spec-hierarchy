#!/usr/bin/env bash
#SBATCH --job-name=vllm-moe
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=01:00:00
#SBATCH --output=vllm-moe-%j.out
# exp42 check in vLLM: verify cost vs number of verified tokens on real text, MoE Qwen3-30B-A3B (TP=2) vs
# dense Qwen3-8B, batch 1 / 4 / 16 (fused MoE kernels: is cost driven by the experts loaded?).
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
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
for MT in "Qwen/Qwen3-30B-A3B 2" "Qwen/Qwen3-8B 1"; do
  set -- $MT
  python scripts/bench_vllm.py --model $1 --tp $2 --ctx 512 --n 64 --qs 1 2 4 8 16 --batches 1 4 16 --seqs 7 \
    --data "$D" --out results/bench_vllm_moe.json 2>&1 | grep -v -i warn | grep "verify cost\|decode step\|Error\|error" 
done
cat results/bench_vllm_moe.json
