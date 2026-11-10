#!/usr/bin/env bash
#SBATCH --job-name=exp111
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=exp111-%j.out
# exp111: which taps the drafter depends on (mean ablation per tap, overall and at the last m positions),
# for the adapted LayerSkip drafter (epoch 3) and the original z-lab drafter, on the held-out validation split.
#   sbatch snellius/job_exp111.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp111"
W=${LS_DIR:-/scratch-shared/$USER/hspec_ls}
F="grep --line-buffered -v -i warn"
python scripts/exp111_tap_importance.py --draft "${CKPT:-$W/drafter_ls_ft/epoch3}" --data $W/data_ft_clean \
  --save exp111_ep3 2>&1 | $F
python scripts/exp111_tap_importance.py --draft z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat --data $W/data_ft_clean \
  --save exp111_orig 2>&1 | $F
