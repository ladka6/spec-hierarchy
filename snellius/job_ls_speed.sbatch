#!/usr/bin/env bash
#SBATCH --job-name=lsspeed
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=lsspeed-%j.out
# Split-verification cost on LayerSkip Llama3-8B at exits 4 / 8 / 12 / 16 (eager: use as ratios) and the cost of one
# Llama DFlash drafter call (CUDA graphs, block 10). No training.
#   sbatch snellius/job_ls_speed.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/lsspeed_$SLURM_JOB_ID"
M=facebook/layerskip-llama3-8B; DR=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
python scripts/bench_split.py --target $M --exits 4 8 12 16 --qs 1 10 20 64 2>&1 | $F
python scripts/bench_draft.py --target $M --draft $DR --block-sizes 10 --ctxs 8 64 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
