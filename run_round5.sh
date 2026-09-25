#!/usr/bin/env bash
# Round 5, from a fresh Colab runtime (A100):
#   git clone https://github.com/ladka6/spec-hierarchy && bash spec-hierarchy/run_round5.sh
#  1. CPU tests (now including the asynchronous decoder)
#  2. exp6: drafter feature-lag tolerance + candidate next-block hit rate (~15 min)
#  3. exp5: async vs sync three-stage on a simulated clock (~70 min)
# Costs are the round-4 vLLM / CUDA-graph measurements (built into hspec/async3.py).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
DFLASH_COMMIT=07ebd93db9f472af339b644bb70221ad8428328a

pip install -q -r requirements.txt
pip install -q --no-deps "dflash @ git+https://github.com/z-lab/dflash@${DFLASH_COMMIT}"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| gpu', torch.cuda.get_device_name(0))"

echo "== CPU tests =="
python tests/test_cpu.py > results/test_cpu.log 2>&1 && tail -1 results/test_cpu.log || { tail -30 results/test_cpu.log; exit 1; }

echo "== exp6: drafting ahead of the middle model =="
python scripts/exp6_lag.py --n 10 2>&1 | grep -v -i warn | tee results/exp6_lag.log

echo "== exp5: asynchronous target =="
python scripts/exp5_async.py --n 10 --latencies-ms 0 10 50 100 2>&1 | grep -v -i warn | tee results/exp5_async.log

tar czf results.tgz results
echo "done: send results/exp6_lag.log and results/exp5_async.log (or results.tgz)"
