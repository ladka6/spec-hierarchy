#!/usr/bin/env bash
#SBATCH --job-name=pcabl
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --output=pcabl-%j.out
# exp27 ablation: copy correction with copy-feature continuation vs tokens-only continuation vs correction alone.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/pcabl"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
L=/scratch-shared/$USER/hspec_lagft/drafter_lag16
for T in 0.0 1.0; do
  python scripts/exp27_refresh.py --lag-draft $L --no-tokens --no-fresh --proxies bnb4:Qwen/Qwen3-8B --temp $T \
    --ks 2 --gates 0.3 --out exp27_pcabl_t$T 2>&1 | $F
done
python scripts/costmodel_refresh.py "$HSPEC_RESULTS"/exp27_pcabl_t*.json --top 12
