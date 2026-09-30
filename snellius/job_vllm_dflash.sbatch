#!/usr/bin/env bash
#SBATCH --job-name=vllm-dflash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=03:00:00
#SBATCH --output=vllm-dflash-%j.out
# Batch-1 plain vs DFlash decode in vLLM on 1 and 2 GPUs (real prompts, greedy): the engine-level
# per-round split (drafter share) and the async ceiling in a real engine.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
R="$SLURM_SUBMIT_DIR/results/vllm"
mkdir -p "$R"
# 1) prompts, formatted with the chat template (hspec env)
source "$HOME/venvs/hspec/bin/activate"
python scripts/dump_prompts.py --out "$R/prompts.json" --n 10 2>&1 | grep -v -i warn
deactivate
# 2) vLLM runs
source "$HOME/venvs/vllm/bin/activate"
NVCC=$(find "$VIRTUAL_ENV" -type f -name nvcc -path "*bin*" 2>/dev/null | head -1)
export CUDA_HOME=$(dirname "$(dirname "$NVCC")") PATH="$(dirname "$NVCC"):$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
for TP in 1 2; do
  python scripts/vllm_dflash.py --prompts "$R/prompts.json" --tp $TP --out "$R/ar_tp$TP.json" 2>&1 \
    | grep -E '^\{|^ "|^\}|Error|error' | tail -15
  python scripts/vllm_dflash.py --prompts "$R/prompts.json" --tp $TP --spec --out "$R/dflash_tp$TP.json" 2>&1 \
    | grep -E '^\{|^ "|^\}|Error|error' | tail -25
done
python - "$R" <<'PY'
import json, sys, os
R = sys.argv[1]
for tp in (1, 2):
    try:
        ar = json.load(open(f"{R}/ar_tp{tp}.json")); df = json.load(open(f"{R}/dflash_tp{tp}.json"))
    except FileNotFoundError as e:
        print("missing", e); continue
    line = f"TP={tp}: plain {ar['decode_ms_per_token']:.2f} ms/tok, DFlash {df['decode_ms_per_token']:.2f} ms/tok, " \
           f"speedup {ar['decode_ms_per_token'] / df['decode_ms_per_token']:.2f}x"
    if "round_ms" in df:
        share = max(0.0, 1 - ar["decode_ms_per_token"] / df["round_ms"])
        line += f", tau {df['tau']:.2f}, round {df['round_ms']:.2f} ms, draft+overhead share ~{share:.2f}, " \
                f"async ceiling ~{1 / (1 - share):.2f}x"
    else:
        line += f" (no spec counters: {df.get('metrics_error')})"
    print(line)
PY
