#!/usr/bin/env bash
#SBATCH --job-name=rtrainall
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=routetrain-all-%j.out
# exp49 long run: routers + attention LoRA + one LoRA per expert (grouped kernels), block-shared routing, n=16.
# Saves at every eval; reports the expert union of aligned and unaligned 16-token windows.
# Usage: sbatch snellius/job_routetrain_all.sbatch [S] [steps] [mode]   (defaults 16 4000 router+attn+experts)
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
S=${1:-16}; STEPS=${2:-4000}; MODE=${3:-router+attn+experts}
python scripts/exp49_routetrain.py --mode $MODE --S $S --n 16 --steps $STEPS --eval-every 500 --data "$D" \
  --out "$W/routetrain_${MODE//+/_}_S${S}_${STEPS}" \
  2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "eval\|trainable\|done\|rror\|expert LoRA"
