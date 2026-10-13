#!/usr/bin/env bash
#SBATCH --job-name=copyeval
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:30:00
#SBATCH --output=copyeval-%j.out
# exp27 (offline refresh, T=0) with the trained copies and the untrained 18-layer copy (untrained rtn2/rtn4: results/lowbit).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/copytrain"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
W=/scratch-shared/$USER/hspec_copytrain
mkdir -p $W
for P in lora:$W/copy_rtn2 lora:$W/copy_keep18 keep18:Qwen/Qwen3-8B; do
  N=$(basename ${P#*:}); [ "$P" = keep18:Qwen/Qwen3-8B ] && N=keep18_untrained
  python scripts/exp27_refresh.py --no-tokens --no-fresh --proxies $P --out exp27_$N 2>&1 | $F
done
