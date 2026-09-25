#!/usr/bin/env bash
#SBATCH --job-name=hspec-r6
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --output=hspec-r6-%j.out
#
# Round 6: algorithm comparison on a simulated two-device clock, 4 GPUs in parallel.
#   cd spec-hierarchy && sbatch snellius/job_r6.sbatch
# Run snellius/setup.sh on a login node first.
#
#   GPU0  exp7 calibration, then baselines: dflash, ddtree-64, PEARL-style two-stage async
#   GPU1  existing three-stage: sync-16, async-8, async-16 (all with a 32-node mid tree)
#   GPU2  confidence-timed checks (C) and C + hedging
#   GPU3  hedging (A): async with 2 / 4 branches, sync with 2 branches
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/r6_$SLURM_JOB_ID"
mkdir -p "$HSPEC_RESULTS"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"

COMMON="--n 20 --datasets gsm8k math500 humaneval mt-bench --latencies-ms 0 10 50 100"
run() {  # gpu tag args...
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python "$@" > "$HSPEC_RESULTS/$tag.log" 2>&1
  echo "[$(date +%H:%M)] $tag exit $?"
}

( run 0 calib scripts/exp7_calib.py --n 20
  run 0 gpu0 scripts/exp8_algos.py $COMMON --tag gpu0 \
      --configs base:dflash base:ddtree-64 base:ddtree-32 pearl-8 pearlsync-8 pearl-4 ) &
run 1 gpu1 scripts/exp8_algos.py $COMMON --tag gpu1 \
    --configs sync-16+mt32 async-8+mt32 async-16+mt32 &
run 2 gpu2 scripts/exp8_algos.py $COMMON --tag gpu2 --tau 0.5 --min-window 4 \
    --configs async-c32+mt32 sync-c32+mt32 async-c32+mt32+h2 &
run 3 gpu3 scripts/exp8_algos.py $COMMON --tag gpu3 --fork-thr 0.6 \
    --configs async-8+mt32+h2 async-8+mt32+h4 sync-16+mt32+h2 &
wait

echo "== calibration =="
grep -v -i warn "$HSPEC_RESULTS/calib.log" | sed -n '/==/,$p'
echo "== algorithms =="
python scripts/aggregate_exp8.py "$HSPEC_RESULTS" | tee "$HSPEC_RESULTS/summary.txt"
