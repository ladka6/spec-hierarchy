#!/usr/bin/env bash
#SBATCH --job-name=ls-gen1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=ls-gen-%j.out
# One of the 4 data shards of job_ls_gen.sbatch on a single GPU (same output, starts sooner).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
K=$1
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LS_DIR:-/scratch-shared/$USER/hspec_ls}
mkdir -p "$W/data"
python scripts/gen_traindata.py --target facebook/layerskip-llama3-8B --shard $K --nshards 4 --out "$W/data" \
  2>&1 | grep --line-buffered -v -i warn
