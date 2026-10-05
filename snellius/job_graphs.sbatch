#!/usr/bin/env bash
#SBATCH --job-name=graphs
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=graphs-%j.out
# Stage latencies of feature-refreshed decoding under CUDA graphs: target verify, drafter (block 1, batched restarts), 4-bit copy.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/graphs"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
L=/scratch-shared/$USER/hspec_lagft/drafter_lag16
# gptqmodel JIT-compiles its Marlin kernels on first use: needs nvcc (take the one pip installed into the vLLM venv)
NVCC=$(find "$HOME/venvs/vllm" -type f -name nvcc -path "*bin*" 2>/dev/null | head -1)
if [ -n "$NVCC" ]; then export CUDA_HOME=$(dirname "$(dirname "$NVCC")"); export PATH="$CUDA_HOME/bin:$PATH"; fi
echo "CUDA_HOME=${CUDA_HOME:-unset}"
python scripts/bench_refresh.py --only-copies --copies hf:Qwen/Qwen3-8B-AWQ 2>&1 | $F
