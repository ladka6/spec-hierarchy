#!/usr/bin/env bash
#SBATCH --job-name=mixedprec
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:15:00
#SBATCH --output=mixedprec-%j.out
# exp78: which layers must stay precise so that the drafter's deep taps (L25, L33) keep real-target quality?
# Simulated RTN quantization; no training. Keeps a CPU copy of the bf16 weights (~16 GB RAM).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/mixedprec"
mkdir -p "$HSPEC_RESULTS"
python scripts/exp78_mixedprec.py --cache "${CACHE:-/scratch-shared/$USER/seqs.pt}" 2>&1 | grep --line-buffered -v -i warn
