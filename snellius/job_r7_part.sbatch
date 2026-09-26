#!/usr/bin/env bash
#SBATCH --job-name=hspec-r7
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=hspec-r7-%j.out
#
# Round 7, one quarter per job. Submit all four with snellius/submit_r7.sh.
#   0  direct CUDA-graph costs (bf16 target, torchao int4, bnb4; drafter vs batch)
#      + calibration and one config with the torchao int4 middle model
#   1  confidence-timed checks, async: tau 0.3 / 0.7, min window 2
#   2  confidence-timed checks, sync (single-GPU algorithm) + sync hedging
#   3  hedging with cheap side branches (chains), fork threshold 0.7, 2 / 4 branches
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
R="$HSPEC_RESULTS"
case $PART in
  0) python scripts/bench_graph.py --mids ao4:Qwen/Qwen3-8B bnb4:Qwen/Qwen3-8B > "$R/bench_graph.log" 2>&1
     HSPEC_RESULTS="$R/calib_ao4" python scripts/exp7_calib.py --n 20 --mid ao4:Qwen/Qwen3-8B > "$R/calib_ao4.log" 2>&1
     python scripts/exp8_algos.py $COMMON --tag p0 --mid ao4:Qwen/Qwen3-8B --suffix @ao4 \
       --configs sync-c32+mt32 async-c32+mt32 > "$R/p0.log" 2>&1 ;;
  1) python scripts/exp8_algos.py $COMMON --tag p1 \
       --configs async-c32t0.3+mt32 async-c32t0.7+mt32 async-c32t0.5m2+mt32 > "$R/p1.log" 2>&1 ;;
  2) python scripts/exp8_algos.py $COMMON --tag p2 \
       --configs sync-c32t0.3+mt32 sync-c32t0.7+mt32 sync-c32t0.5m2+mt32 sync-c16+mt32 \
                 sync-c32+mt32+h4+f0.7+at0 > "$R/p2.log" 2>&1 ;;
  3) python scripts/exp8_algos.py $COMMON --tag p3 \
       --configs async-c32+mt32+h2+f0.7+at0 async-c32+mt32+h4+f0.7+at0 async-8+mt32+h4+f0.7+at0 > "$R/p3.log" 2>&1 ;;
esac
echo "part $PART exit $?"
