#!/usr/bin/env bash
#SBATCH --job-name=benchbatch
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=benchbatch-%j.out
# exp59 cost curves: vLLM verify latency of the bf16 target and the 4-bit AWQ copy as a function of batch size x
# fresh tokens per request (real text, cached 512-token prefix). Gives the two knees (target compute-bound, copy
# compute-bound) that the exp59 replay needs. Same vLLM venv as job_vllm_cost.sbatch.
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
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
D=${BASEDATA:-/scratch-shared/$USER/hspec_lagft/data}
ls "$D"/shard*.pt >/dev/null 2>&1 || { echo "no real-text shards in $D"; exit 1; }
for M in Qwen/Qwen3-8B Qwen/Qwen3-8B-AWQ; do
  python scripts/bench_vllm.py --model $M --ctx 512 --qs 1 17 33 65 129 --batches 1 2 4 8 16 32 --seqs 4 \
    --data "$D" --out results/bench_vllm_batch.json 2>&1 | grep -v -i warn | tail -12
done
cat results/bench_vllm_batch.json
