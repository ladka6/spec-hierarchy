#!/usr/bin/env bash
#SBATCH --job-name=hspec-r6p
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=hspec-r6p-%j.out
#
# One quarter of round 6 on a single GPU. Submit all four with snellius/submit_split.sh.
#   sbatch snellius/job_r6_part.sbatch <part 0-3> <results dir>
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
nvidia-smi --query-gpu=name,memory.total --format=csv

COMMON="--n 20 --datasets gsm8k math500 humaneval mt-bench --latencies-ms 0 10 50 100"
case $PART in
  0) python scripts/exp7_calib.py --n 20 > "$HSPEC_RESULTS/calib.log" 2>&1
     python scripts/exp8_algos.py $COMMON --tag gpu0 \
       --configs base:dflash base:ddtree-64 base:ddtree-32 pearl-8 pearlsync-8 pearl-4 > "$HSPEC_RESULTS/gpu0.log" 2>&1 ;;
  1) python scripts/exp8_algos.py $COMMON --tag gpu1 \
       --configs sync-16+mt32 async-8+mt32 async-16+mt32 > "$HSPEC_RESULTS/gpu1.log" 2>&1 ;;
  2) python scripts/exp8_algos.py $COMMON --tag gpu2 --tau 0.5 --min-window 4 \
       --configs async-c32+mt32 sync-c32+mt32 async-c32+mt32+h2 > "$HSPEC_RESULTS/gpu2.log" 2>&1 ;;
  3) python scripts/exp8_algos.py $COMMON --tag gpu3 --fork-thr 0.6 \
       --configs async-8+mt32+h2 async-8+mt32+h4 sync-16+mt32+h2 > "$HSPEC_RESULTS/gpu3.log" 2>&1 ;;
esac
echo "part $PART exit $?"
