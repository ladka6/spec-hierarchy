#!/usr/bin/env bash
# Round 2:
#  1. eager per-call latency of target / middle models / drafter
#  2. exp3 at 0 ms with the fixed drafter masking -> correct call counts per config
#  3. vLLM decode latency (CUDA graphs) for bf16 and AWQ int4, in a separate environment
#  4. cost model: counts x latencies (+ simulated network delay per target call)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

echo "== CPU correctness test =="
python tests/test_cpu.py 2>&1 | tail -3

echo "== eager forward latency =="
python scripts/bench_forward.py --mode eager --mids bnb4:Qwen/Qwen3-8B \
  2>&1 | tee results/bench_forward.log

echo "== exp3 call counts (fixed) =="
python scripts/exp3_pipeline.py --mids bnb4:Qwen/Qwen3-8B --n 10 --no-ar \
  --windows 1 4 8 16 32 64 128 2>&1 | tee results/exp3_fixed.log

echo "== vLLM decode latency (separate env) =="
if [[ ! -x /content/vllm_env/bin/python ]]; then
  pip install -q uv
  uv venv -q -p 3.12 /content/vllm_env
  uv pip install -q -p /content/vllm_env/bin/python vllm ninja
fi
uv pip install -q -p /content/vllm_env/bin/python ninja
for m in Qwen/Qwen3-8B Qwen/Qwen3-8B-AWQ; do
  PATH=/content/vllm_env/bin:$PATH /content/vllm_env/bin/python scripts/bench_vllm.py --model "$m" 2>&1 | grep -E "ms per decode|Error|error" | tee -a results/bench_vllm.log || true
done

echo "== cost model =="
python scripts/cost_model.py --exp3 results/exp3_pipeline.json --bench results/bench_forward.json \
  2>&1 | tee results/cost_model.log
