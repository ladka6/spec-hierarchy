#!/usr/bin/env bash
#SBATCH --job-name=hedge89
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=hedge89-%j.out
# exp89: hedged multi-round spines (block drafter + copy), copy-margin vs drafter-confidence hedging, cost model.
#   sbatch --export=ALL snellius/job_exp89.sbatch
#   COPY=lora:$HOME/hspec_ckpt/copyrole/copy_both_4bit sbatch --export=ALL snellius/job_exp89.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp89"
mkdir -p "$HSPEC_RESULTS"
C=${COPY:-rtn4:Qwen/Qwen3-8B}
python scripts/exp89_hedge.py --copy $C --out exp89_$(basename ${C#*:}) ${EXTRA:-} 2>&1 | grep --line-buffered -v -i warn
