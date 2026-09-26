#!/usr/bin/env bash
#SBATCH --job-name=hspec-2gpu
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=06:00:00
#SBATCH --output=hspec-2gpu-%j.out
#
# Real two-GPU run (target in its own process on GPU 1) vs the simulated clock.
#   sbatch snellius/job_r8_real2gpu.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/r8_2gpu_$SLURM_JOB_ID"
mkdir -p "$HSPEC_RESULTS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python scripts/exp9_real2gpu.py --n 10 --latencies-ms 0 100 --configs sync-16+mt32 async-8+mt32 async-c32+mt32 2>&1 | grep -v -i warn | tee "$HSPEC_RESULTS/exp9.log"
