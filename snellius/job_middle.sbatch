#!/usr/bin/env bash
#SBATCH --job-name=middle
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=middle-%j.out
# Middle-model candidates for the three-model PEARL pipeline (vLLM, batch 1): speed (plain and
# with DFlash), agreement e with the bf16 target, and the projected pipeline throughput.
# Download the candidates once on a login node first (see the message / submit instructions).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
R="$SLURM_SUBMIT_DIR/results/middle"
mkdir -p "$R"
source "$HOME/venvs/hspec/bin/activate"
[ -f "$R/prompts.json" ] || python scripts/dump_prompts.py --out "$R/prompts.json" --n 10 2>&1 | grep -v -i warn
deactivate
source "$HOME/venvs/vllm/bin/activate"
NVCC=$(find "$VIRTUAL_ENV" -type f -name nvcc -path "*bin*" 2>/dev/null | head -1)
export CUDA_HOME=$(dirname "$(dirname "$NVCC")") PATH="$(dirname "$NVCC"):$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
Q='^\{|^ "|^\}|Error|error'
T=Qwen/Qwen3-8B
[ -f "$R/traj.json" ] || python scripts/vllm_middle.py traj --model $T --prompts "$R/prompts.json" --out "$R/traj.json" 2>&1 | grep -E "$Q|wrote" | tail -3
[ -f "$R/speed_target.json" ] || python scripts/vllm_dflash.py --model $T --prompts "$R/prompts.json" --out "$R/speed_target.json" 2>&1 | grep -E "$Q" | tail -3
[ -f "$R/dflash_target.json" ] || python scripts/vllm_dflash.py --model $T --spec --prompts "$R/prompts.json" --out "$R/dflash_target.json" 2>&1 | grep -E "$Q" | tail -3
for M in Qwen/Qwen3-8B-AWQ Qwen/Qwen3-8B-FP8 Qwen/Qwen3-4B Qwen/Qwen3-1.7B; do
  N=$(basename $M)
  echo "=== $M"
  python scripts/vllm_middle.py agree --model $M --traj "$R/traj.json" --out "$R/agree_$N.json" 2>&1 | grep -E "$Q" | tail -3
  python scripts/vllm_dflash.py --model $M --prompts "$R/prompts.json" --out "$R/speed_$N.json" 2>&1 | grep -E "$Q" | tail -3
  case $M in *8B*)   # the 8B DFlash drafter fits 8B variants only
    python scripts/vllm_dflash.py --model $M --spec --prompts "$R/prompts.json" --out "$R/dflash_$N.json" 2>&1 | grep -E "$Q" | tail -3 ;;
  esac
done
python scripts/middle_projection.py "$R"
