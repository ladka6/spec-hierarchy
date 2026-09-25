#!/usr/bin/env bash
# Round 2: measure forward costs (r = c_T / c), then the pipeline under simulated target latency.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
pip install -q torchao

echo "== forward latency: target vs middle models =="
python scripts/bench_forward.py --mids bnb4:Qwen/Qwen3-8B ao4:Qwen/Qwen3-8B ao8:Qwen/Qwen3-8B \
  2>&1 | tee results/bench_forward.log

echo "== exp3 with simulated remote target =="
python scripts/exp3_pipeline.py --mids ao4:Qwen/Qwen3-8B --n 6 --no-ar \
  --windows 1 8 16 32 64 --latencies-ms 0 20 50 100 --tag latency \
  2>&1 | tee results/exp3_latency.log
