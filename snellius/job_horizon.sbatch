#!/usr/bin/env bash
#SBATCH --job-name=horizon
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=horizon-%j.out
# How far can estimated target states carry the drafter without the target? exp29 with the small model + mapper as the corrector (open loop until one verify), T=0 and T=1, up to 6 refreshes (~60 tokens).
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
F="grep --line-buffered -v -i warn"
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
J=/scratch-shared/$USER/hspec_smalljoint
for T in 0.0 1.0; do
  # ridge map, original drafter (no training needed)
  python scripts/exp29_iterate.py --small Qwen/Qwen3-0.6B --data "$D" --ks 1 2 3 4 5 6 --temp $T \
    --out exp34_q06_ridge_t$T 2>&1 | $F
  # trained mapper + jointly trained drafter (from job_smalljoint), if present
  for S in 0.6B 1.7B; do
    R=$J/Qwen3-${S}_joint
    if [ -f "$R/mapper.pt" ] && [ -d "$R/drafter" ]; then
      python scripts/exp29_iterate.py --small Qwen/Qwen3-$S --mapper "$R/mapper.pt" --draft "$R/drafter" \
        --ks 1 2 3 4 5 6 --temp $T --out exp34_q${S/./}_joint_t$T 2>&1 | $F
    else
      echo "skip $S joint: $R/mapper.pt or $R/drafter missing"
    fi
  done
done
