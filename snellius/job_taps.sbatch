#!/usr/bin/env bash
#SBATCH --job-name=taps
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=taps-%j.out
# What an original DFlash drafter takes from its target: tap importance (exp111) on a z-lab target / drafter pair.
# Step 1 (skipped if the shard exists): the target's greedy responses to the same benchmark prompts (150 per set) for every pair.
# Step 2: exp111 with the depth curve (first j / last j taps real) and the recency curve (last 1 .. 32 positions).
#   TARGET, DRAFT (HF ids), TAG (directory / result name)
#   sbatch --export=ALL,TARGET=Qwen/Qwen3-8B,DRAFT=z-lab/Qwen3-8B-DFlash-b16,TAG=qwen3_8b snellius/job_taps.sbatch
# Standard pairs: bash snellius/submit_taps.sh
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/taps"
F="grep --line-buffered -v -i warn"
D=${TAPS_DIR:-/scratch-shared/$USER/hspec_taps}/$TAG
if [ ! -f "$D/shard0.pt" ]; then
  python scripts/gen_pairdata.py --target "$TARGET" --out "$D" --n ${NPROMPTS:-150} 2>&1 | $F
fi
[ -f "$D/shard0.pt" ] || { echo "[taps] data generation failed, stopping"; exit 1; }
python scripts/exp111_tap_importance.py --target "$TARGET" --draft "$DRAFT" --data "$D" \
  --ms 1 2 4 8 16 32 --anchors ${ANCHORS:-1000} --save taps_$TAG 2>&1 | $F
