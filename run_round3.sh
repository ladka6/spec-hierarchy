#!/usr/bin/env bash
# Round 3, from a fresh Colab runtime (A100):
#   git clone https://github.com/ladka6/spec-hierarchy && bash spec-hierarchy/run_round3.sh
#  1. CPU tests (DDTree, branching, tree attention)
#  2. drafter cost under CUDA graphs (bs 16 / 32)
#  3. exp3 call counts: dflash b16/b32, DDTree 32/64/128, three-stage with and without branching
#  4. vLLM: decode cost and verification cost vs q, bf16 and AWQ
#  5. cost model
# Everything is logged to results/; results.tgz at the end is what to send back.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
DFLASH_COMMIT=07ebd93db9f472af339b644bb70221ad8428328a

pip install -q -r requirements.txt
pip install -q --no-deps "dflash @ git+https://github.com/z-lab/dflash@${DFLASH_COMMIT}"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| gpu', torch.cuda.get_device_name(0))"

echo "== CPU tests =="
python tests/test_cpu.py > results/test_cpu.log 2>&1 && tail -1 results/test_cpu.log || { tail -30 results/test_cpu.log; exit 1; }

echo "== drafter cost (CUDA graphs) =="
python scripts/bench_draft.py 2>&1 | grep -v Warning | tee results/bench_draft.log

echo "== exp3 call counts =="
python scripts/exp3_pipeline.py --mids bnb4:Qwen/Qwen3-8B --n 10 --no-ar \
  --windows 8 16 32 64 --dflash-blocks 32 --ddtree-budgets 32 64 128 \
  --branch-k 0 4 --branch-len 8 --branch-margin 0.5 --tag r3 \
  2>&1 | grep -v -i warn | tee results/exp3_r3.log

echo "== vLLM: decode + verify cost vs q =="
if [[ ! -x /content/vllm_env/bin/python ]]; then
  pip install -q uv
  uv venv -q -p 3.12 /content/vllm_env
  uv pip install -q -p /content/vllm_env/bin/python vllm ninja
fi
for m in Qwen/Qwen3-8B Qwen/Qwen3-8B-AWQ; do
  PATH=/content/vllm_env/bin:$PATH /content/vllm_env/bin/python scripts/bench_vllm.py --model "$m" \
    > "results/bench_vllm_${m//\//_}.log" 2>&1 || echo "vLLM failed for $m (see log)"
  grep -E "ms per decode|verify cost" "results/bench_vllm_${m//\//_}.log" || true
done

echo "== cost model =="
# bench_forward.json from round 2 is not in a fresh runtime; eager scenarios are skipped then
[[ -f results/bench_forward.json ]] || echo '{"rows": [], "draft_ms": 8.4}' > results/bench_forward.json
python scripts/cost_model.py --exp3 results/exp3_pipeline_r3.json --bench results/bench_forward.json \
  --latencies-ms 0 5 10 20 50 100 2>&1 | tee results/cost_model_r3.log

tar czf results.tgz results
echo "done: send results.tgz (or results/exp3_r3.log + results/cost_model_r3.log)"
