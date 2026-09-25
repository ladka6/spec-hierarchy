#!/usr/bin/env bash
# One-shot setup + pilot run on a Colab A100. Logs and JSON results land in results/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

DFLASH_COMMIT=07ebd93db9f472af339b644bb70221ad8428328a

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  pip install -q -r requirements.txt
  # no-deps: keep Colab's torch instead of dflash's pinned one
  pip install -q --no-deps "dflash @ git+https://github.com/z-lab/dflash@${DFLASH_COMMIT}"
fi

python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| gpu', torch.cuda.get_device_name(0))"

echo "== CPU correctness test =="
python tests/test_cpu.py 2>&1 | tee results/test_cpu.log

echo "== exp1: middle-model agreement =="
python scripts/exp1_agreement.py 2>&1 | tee results/exp1.log

echo "== exp2: drafter on middle-model features =="
python scripts/exp2_features.py 2>&1 | tee results/exp2.log

echo "== exp3: three-stage pipeline =="
python scripts/exp3_pipeline.py 2>&1 | tee results/exp3.log
