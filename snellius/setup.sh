#!/usr/bin/env bash
# One-time setup on a Snellius LOGIN node (compute nodes may not have internet):
#   cd spec-hierarchy && bash snellius/setup.sh
# Creates a venv, installs dependencies, downloads models and datasets into $HF_HOME.
set -euo pipefail
cd "$(dirname "$0")/.."

module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
python3 --version

VENV=${HSPEC_VENV:-$HOME/venvs/hspec}
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
mkdir -p "$HF_HOME"
[[ -d $VENV ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch
pip install -q -r requirements.txt
pip install -q --no-deps "dflash @ git+https://github.com/z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a"

python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16"]:
    print("downloading", m, snapshot_download(m))
import sys; sys.path.insert(0, ".")
from hspec.data import load_prompts
for d in ["gsm8k", "math500", "humaneval", "mt-bench"]:
    print(d, len(load_prompts(d, 20)), "prompts cached")
PY
python -c "import torch, transformers, bitsandbytes; print('torch', torch.__version__, 'transformers', transformers.__version__)"
echo "setup done. HF_HOME=$HF_HOME VENV=$VENV"
