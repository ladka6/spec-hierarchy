#!/usr/bin/env bash
#SBATCH --job-name=domdata
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=domdata-%j.out
# Per-domain Qwen3-8B greedy data for the drafter-forgetting study (math, code, chat, think).
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
W=/scratch-shared/$USER/hspec_domains
for DM in math code chat think; do
  N=1100; [ "$DM" = think ] && N=600
  python scripts/gen_domaindata.py --domain $DM --n $N --out "$W" 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "saved\|rror\|00/"
done
