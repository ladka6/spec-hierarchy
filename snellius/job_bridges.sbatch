#!/usr/bin/env bash
#SBATCH --job-name=bridges
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=bridges-%j.out
# exp57: taps8 copy (target layers 0 1 9 17 25 33 34 35, frozen) + bridges for the removed spans.
# lin = closed-form affine bridges only (training-free), mlp / attn = affine init + trained nonlinear bridges.
# Then exp27 on all three. Compare: taps8 LoRA 9.20 / 10.26, 18L 10.45 / 11.55, 2-bit trained 11.19 / 12.30,
# 4-bit 11.90 / 13.05 (conf2 / all).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/bridges"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
W=/scratch-shared/$USER/hspec_copytrain
for K in ${KINDS:-lin mlp attn}; do
  python scripts/exp57_bridges.py --kind $K --teacher $W/teacher.pt --out $W/bridge_$K 2>&1 | $F
done
for K in ${KINDS:-lin mlp attn}; do
  python scripts/exp27_refresh.py --no-tokens --no-fresh --proxies bridge:$W/bridge_$K --out exp27_bridge_$K 2>&1 | $F
done
