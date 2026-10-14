#!/usr/bin/env bash
#SBATCH --job-name=shadow
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:45:00
#SBATCH --output=shadow-%j.out
# exp63: shadow refresh offline. Outcome prediction from the copy (hit@K), quality of next-round drafts prepared
# from copy features during verification (depth 0..3) vs drafted after verification, and the event model.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/shadow"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
for T in ${TEMPS:-0.0 1.0}; do
  python scripts/exp63_shadow.py --copy ${COPY:-bnb4:Qwen/Qwen3-8B} --temp $T --out exp63_shadow_T$T 2>&1 | $F
done
