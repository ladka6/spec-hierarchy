#!/usr/bin/env bash
#SBATCH --job-name=faker
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=faker-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/faker"
mkdir -p "$HSPEC_RESULTS"
# Train a faker (emulates target context vectors for drafted tokens), then measure its redraft fix rate.
W=/scratch-shared/$USER/hspec_lagft
F="grep --line-buffered -v -i warn"
FK=${FAKER_OUT:-$W/faker_d1024_l3.pt}
python scripts/train_faker.py --data "$W/data" --out "$FK" --max-train ${MAX_TRAIN:-4000} --epochs 1 \
  --dim ${FAKER_DIM:-1024} --layers ${FAKER_LAYERS:-3} 2>&1 | $F
python scripts/exp26_lossdiag.py --lag-draft $W/drafter_lag16 --faker "$FK" --out exp26_faker_t0 2>&1 | $F
