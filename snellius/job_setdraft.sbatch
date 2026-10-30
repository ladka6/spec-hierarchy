#!/usr/bin/env bash
#SBATCH --job-name=setdraft
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=setdraft-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/setdraft"
mkdir -p "$HSPEC_RESULTS"
CACHE=${CACHE:-/scratch-shared/$USER/seqs.pt}
DATA=${DATA:-/scratch-shared/$USER/hspec_lagft/data}
STEPS=${STEPS:-6000}
F="grep --line-buffered -v -i warn"
# exp77 at scale: set-valued block drafting (K modes as input vectors, winner-takes-all), trained on the large set,
# evaluated on the same held-out sequences as on Colab. Includes the K = 1 control fine-tune on the same data.
#   CACHE=... DATA=... STEPS=6000 K=3 sbatch snellius/job_setdraft.sbatch
python scripts/exp77_setdraft.py --cache "$CACHE" --data "$DATA" --steps "$STEPS" --K "${K:-3}" --mode-input 2>&1 | $F
