#!/usr/bin/env bash
#SBATCH --job-name=la_save
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:30:00
#SBATCH --output=lora_save-%j.out
# exp40 again, larger (4000 sequences, 1 epoch), saving the slot-gated LoRA for train_slotdflash.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/horizon"
mkdir -p "$HSPEC_RESULTS" /scratch-shared/$USER/hspec_lookahead
python scripts/exp40_lookahead_lora.py --data "${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data" \
  --train-seqs 4000 --epochs 1 --no-probe --out exp40_lookahead_lora_4k \
  --save /scratch-shared/$USER/hspec_lookahead/lora.pt 2>&1 | grep --line-buffered -v -i warn
