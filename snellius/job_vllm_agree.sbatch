#!/usr/bin/env bash
#SBATCH --job-name=vllmagree
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:45:00
#SBATCH --output=vllmagree-%j.out
# exp66 in vLLM: greedy agreement of off-the-shelf quantized Qwen3-8B checkpoints with the bf16 target.
# Models as arguments; the target is always run first. Same vLLM venv as job_bench_batch.sbatch.
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
T=/scratch-shared/$USER/hspec_copytrain/teacher.pt
O=results/vllm_agree
for M in Qwen/Qwen3-8B "$@"; do
  python scripts/exp66_vllm_agree.py predict --model "$M" --teacher $T --out $O 2>&1 | grep -v -i warn | tail -4
done
python scripts/exp66_vllm_agree.py stats --out $O --target Qwen/Qwen3-8B
