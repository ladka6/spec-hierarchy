#!/usr/bin/env bash
# Login node, once: build the ~53k fine-tune prompts and submit the vLLM generation (stop at "\nQuestion:").
#   bash snellius/prep_ls_ft.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token")}
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
P=/scratch-shared/$USER/prompts_ls_ft.pt
[ -f "$P" ] || python scripts/gen_traindata_vllm.py prompts --target facebook/layerskip-llama3-8B \
  --n-gsm8k 7400 --n-alpaca 20000 --n-ultrachat 25000 --file "$P"
FILE=$P OUT=/scratch-shared/$USER/hspec_ls/data_ft MAXNEW=1024 STOP='\nQuestion:' \
  sbatch --export=ALL snellius/job_gen_vllm.sbatch
