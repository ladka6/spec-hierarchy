#!/usr/bin/env bash
#SBATCH --job-name=rtrain
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=routetrain-%j.out
# exp49: train OLMoE to work under causal block-shared routing (n=16, S=16), distilled from the original:
# routers only (locality adapter) vs routers + attention LoRA. One GPU, the two runs back to back.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/horizon"
mkdir -p "$HSPEC_RESULTS"
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_moe
for M in router router+attn; do
  python scripts/exp49_routetrain.py --mode $M --S 16 --n 16 --steps 1000 --data "$D" --out "$W/routetrain_${M/+/_}" \
    2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "eval\|trainable\|done\|rror"
done
