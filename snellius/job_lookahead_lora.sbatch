#!/usr/bin/env bash
#SBATCH --job-name=la_lora
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=lookahead_lora-%j.out
# exp40: lookahead adapters. Slot-gated LoRA trained so mask-slot states match the oracle (true-token) states;
# eval: cosine to oracle, direct top-1 per offset, exp39 probe gap closed (held-out sequences).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/horizon"
mkdir -p "$HSPEC_RESULTS"
python scripts/exp40_lookahead_lora.py --data "${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data" ${LORA_ARGS:-} 2>&1 | grep --line-buffered -v -i warn
