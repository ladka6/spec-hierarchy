#!/usr/bin/env bash
#SBATCH --job-name=selfspec
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=selfspec-%j.out
# exp22: self-speculative acceptance + GSM8K accuracy, base LayerSkip model vs its LoRA fine-tunes.
#   sbatch snellius/job_selfspec.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${DRIFT_DIR:-/scratch-shared/$USER/hspec_drift}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/selfspec"
mkdir -p "$HSPEC_RESULTS"
M="base=facebook/layerskip-llama3-8B"
for n in math200 math1000 code200 chat1000; do
  [ -f "$W/ls-$n/config.json" ] && M="$M $n=$W/ls-$n"
done
python scripts/exp22_selfspec.py --models $M 2>&1 | grep --line-buffered -v -i warn
