#!/usr/bin/env bash
#SBATCH --job-name=exp110
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=exp110-%j.out
# exp110: zero-shot alignment check of the z-lab Llama-3.1 DFlash drafter on LayerSkip Llama-3 8B.
# Drafter config (taps / block / mask id), embed+head drift, per-layer hidden-state shift (cos, norm ratio, CKA),
# shift after the drafter's fc, zero-shot tau on LayerSkip vs reference tau on Llama-3.1-Instruct, greedy match.
# Login node first: hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*" (rest via setup_layerskip.sh)
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/exp110
F="grep --line-buffered -v -i warn"
python scripts/exp110_ls_align_check.py --ref_tau --out "results/exp110/exp110_${SLURM_JOB_ID}.json" 2>&1 | $F
