#!/usr/bin/env bash
# Round 2:
#  1. per-call latency of target / middle models / drafter, eager and compiled (CUDA graphs)
#  2. exp3 at 0 ms with the fixed drafter masking, to get correct call counts per config
#  3. cost model: counts x latencies (+ simulated network delay per target call)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

echo "== CPU correctness test =="
python tests/test_cpu.py 2>&1 | tail -3

echo "== forward latency =="
python scripts/bench_forward.py --mids bnb4:Qwen/Qwen3-8B ao4:Qwen/Qwen3-8B ao8:Qwen/Qwen3-8B \
  2>&1 | tee results/bench_forward.log

echo "== exp3 call counts (fixed) =="
python scripts/exp3_pipeline.py --mids bnb4:Qwen/Qwen3-8B --n 10 --no-ar \
  --windows 1 4 8 16 32 64 128 2>&1 | tee results/exp3_fixed.log

echo "== cost model =="
python scripts/cost_model.py --exp3 results/exp3_pipeline.json --bench results/bench_forward.json \
  2>&1 | tee results/cost_model.log
