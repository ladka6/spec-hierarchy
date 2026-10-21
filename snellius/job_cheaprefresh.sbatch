#!/usr/bin/env bash
#SBATCH --job-name=cheaprefresh
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:50:00
#SBATCH --output=cheaprefresh-%j.out
# exp65: how cheap can a KV refresh be? Ridge maps from the drafter's own final states (+ exact layer-1 tap) to the
# target's 5 tap features, against the 4-bit copy and the real features; cosine per tap and chunk-and-restart tau.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/cheaprefresh"
mkdir -p "$HSPEC_RESULTS"
W=/scratch-shared/$USER/hspec_copytrain
python scripts/exp65_cheaprefresh.py --teacher $W/teacher.pt --copy ${COPY:-bnb4:Qwen/Qwen3-8B} 2>&1 \
  | grep --line-buffered -v -i warn
