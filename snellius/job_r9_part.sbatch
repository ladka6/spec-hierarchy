#!/usr/bin/env bash
#SBATCH --job-name=hspec-r9
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=hspec-r9-%j.out
#
# Round 9: two-level async (drafter + target, no middle model) with the lag-tolerant
# drafter. Same 80 prompts and clock as rounds 6/7, so results merge with them.
#   0  baselines: blind original, original + lagged tokens, lag16, lag16 synchronous
#   1  lag16: draft length 4 / 15, check every 4 tokens
#   2  lag16 with a cap on how far ahead of the target the drafter may run
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PART=$1
export HSPEC_RESULTS=$2
mkdir -p "$HSPEC_RESULTS"

module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}

COMMON="--n 20 --datasets gsm8k math500 humaneval mt-bench --latencies-ms 0 10 50 100 --drafters lag16=$W/drafter_lag16"
R="$HSPEC_RESULTS"
case $PART in
  0) python scripts/exp8_algos.py $COMMON --tag p0 \
       --configs pearl-8 pearl-8+lt pearl-8+lt@lag16 pearlsync-15+lt@lag16 > "$R/p0.log" 2>&1 ;;
  1) python scripts/exp8_algos.py $COMMON --tag p1 \
       --configs pearl-4+lt@lag16 pearl-15+lt@lag16 pearl-8+lt+w4@lag16 > "$R/p1.log" 2>&1 ;;
  2) python scripts/exp8_algos.py $COMMON --tag p2 \
       --configs pearl-8+lt+la16@lag16 pearl-4+lt+la8@lag16 pearl-15+lt+la16@lag16 > "$R/p2.log" 2>&1 ;;
esac
echo "part $PART exit $?"
