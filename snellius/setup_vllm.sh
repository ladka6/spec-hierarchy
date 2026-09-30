#!/usr/bin/env bash
# Run once on a LOGIN node: a separate venv for vLLM (it pins its own torch/transformers,
# so it must not share the hspec venv).
#   bash snellius/setup_vllm.sh
set -euo pipefail
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
python -m venv "$HOME/venvs/vllm"
source "$HOME/venvs/vllm/bin/activate"
pip install --upgrade pip
pip install vllm
python -c "import vllm; print('vllm', vllm.__version__)"
