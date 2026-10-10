#!/usr/bin/env bash
#SBATCH --job-name=corr
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=corr-%j.out
# Why do estimated states carry the drafter less far than the copy? Correction-token ablation with the 0.6B ridge corrector (T=0): oracle token (diagnostic), no insert, confidence-gated inserts.
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
for C in oracle none conf:0.5 conf:0.8; do
  python scripts/exp29_iterate.py --small Qwen/Qwen3-0.6B --data "$D" --ks 1 2 3 4 5 6 --temp 0.0 --corr $C \
    --out exp34_q06_ridge_corr${C/:/}_t0.0 2>&1 | $F
done
