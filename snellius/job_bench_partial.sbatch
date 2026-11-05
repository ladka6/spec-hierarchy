#!/usr/bin/env bash
#SBATCH --job-name=bpartial
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=bpartial-%j.out
# CUDA-graph cost of the partial pass (first k layers + exit head) vs a full forward, LayerSkip and Qwen3-8B.
#   sbatch snellius/job_bench_partial.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/bpartial_$SLURM_JOB_ID"
python scripts/bench_partial.py --target facebook/layerskip-llama3-8B --ks 2 4 8 16 --save bench_partial_layerskip 2>&1 | $F
python scripts/bench_partial.py --target Qwen/Qwen3-8B --ks 2 4 8 16 --save bench_partial_qwen3 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
