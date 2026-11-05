#!/usr/bin/env bash
#SBATCH --job-name=lsforks
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=lsforks-%j.out
# exp99 on LayerSkip Llama3-8B: top-1/2/3 of the exit head after 4 / 8 / 16 layers (keepA-B-..: other layers identity,
# final norm + LM head) at DFlash's forks, vs a 4-bit copy of the same model. Drafter: the original Llama-3.1 DFlash
# (no LayerSkip adaptation, so forks come from a mismatched drafter; rough). No training.
#   sbatch snellius/job_ls_forks.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/lsforks_$SLURM_JOB_ID"
M=facebook/layerskip-llama3-8B; DR=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
python scripts/exp99_copyforks.py --target $M --draft $DR --copies \
  keep0-1-2-3:$M keep0-1-2-3-4-5-6-7:$M keep0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15:$M bnb4:$M 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
