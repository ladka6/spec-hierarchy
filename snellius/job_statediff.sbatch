#!/usr/bin/env bash
#SBATCH --job-name=statediff
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --output=statediff-%j.out
# Latent block diffusion over the verifier's hidden states (hspec/statediff.py), feasibility.
# A denoiser initialised from Qwen3-0.6B predicts the 8B's states for the next 16 positions; tokens via the
# target's head. Does iterating (K denoising steps) buy what extra depth would?
#   L28          all 28 layers of Qwen3-0.6B, diffusion
#   L12          first 12 layers, diffusion
#   L6           first 6 layers, diffusion
#   L28_regress  all 28 layers, single-pass predictor (trained at t = 0 only): control for iteration
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_statediff
mkdir -p "$W"
ls "$D"/shard*.pt >/dev/null 2>&1 || { echo "no training data in $D"; exit 1; }
i=0
for C in "28 x" "12 x" "6 x" "28 --regress"; do
  set -- $C
  X=""; N=L$1; [ "$2" != x ] && { X=$2; N=${N}_regress; }
  CUDA_VISIBLE_DEVICES=$i python scripts/train_statediff.py --n-layers $1 $X --data "$D" --out "$W/$N" \
    ${SD_ARGS:-} > "$W/train_$N.log" 2>&1 &
  i=$((i + 1))
done
wait
grep -h "step [0-9]*\] \|saved\|Error\|error" "$W"/train_*.log | grep --line-buffered -v -i warn
