#!/usr/bin/env bash
#SBATCH --job-name=lossproxy
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=lossproxy-%j.out
# exp26: where block-drafter acceptance is lost (coverage vs path vs cap; tokens vs fresh redraft fixes).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/lossdiag"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
L=/scratch-shared/$USER/hspec_lagft/drafter_lag16
P="bnb4:Qwen/Qwen3-8B bnb8:Qwen/Qwen3-8B"
python scripts/exp26_lossdiag.py --lag-draft $L --proxies $P --out exp26_proxy_t0 2>&1 | $F
python scripts/exp26_lossdiag.py --lag-draft $L --proxies $P --temp 1.0 --out exp26_proxy_t1 2>&1 | $F
