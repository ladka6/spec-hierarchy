#!/usr/bin/env bash
# Run once on a LOGIN node before snellius/submit_lagft.sh: downloads the training prompt
# datasets (gsm8k train, MBPP train, Alpaca) into $HF_HOME.
set -euo pipefail
cd "$(dirname "$0")/.."
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
python - <<'PY'
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from gen_traindata import training_prompts
p = training_prompts()
print(len(p), "training prompts cached; example:", p[0][:120].replace("\n", " "))
PY
