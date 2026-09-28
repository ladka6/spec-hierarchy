#!/usr/bin/env bash
#SBATCH --job-name=exit-scan
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=exit-scan-%j.out
# Early-exit agreement per layer: LayerSkip Llama3-8B vs plain Llama3-8B vs Qwen3-8B.
# Run snellius/setup_layerskip.sh on a login node first.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exitscan_$SLURM_JOB_ID"
MODELS=${MODELS:-facebook/layerskip-llama3-8B meta-llama/Meta-Llama-3-8B Qwen/Qwen3-8B}
python scripts/exp12_exit_scan.py --n 10 --models $MODELS 2>&1 | grep --line-buffered -v -i warn
