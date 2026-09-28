#!/usr/bin/env bash
#SBATCH --job-name=ls-gen
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --output=ls-gen-%j.out
# LayerSkip Llama3-8B greedy responses to the ~5k training prompts, 4 shards in parallel.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
W=${LS_DIR:-/scratch-shared/$USER/hspec_ls}
TGT=facebook/layerskip-llama3-8B
DRAFT0=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
mkdir -p "$W/data"
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k python scripts/gen_traindata.py --target $TGT --shard $k --nshards 4 --out "$W/data" \
    > "$W/gen_$k.log" 2>&1 &
done
wait
tail -n 2 "$W"/gen_*.log
ls -la "$W/data"
