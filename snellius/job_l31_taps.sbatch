#!/usr/bin/env bash
#SBATCH --job-name=l31taps
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=l31taps-%j.out
# exp71 on the Llama DFlash drafter's native target (Llama-3.1-8B-Instruct, gated: download after access is granted):
# which of taps 1 / 8 / 15 / 22 / 29 does the drafter rely on, and how exact must they be? No training.
# exp69 only builds the sequence cache here; an error after the cache is saved (empty --copy) is expected.
#   sbatch snellius/job_l31_taps.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/l31taps_$SLURM_JOB_ID"
LI=meta-llama/Llama-3.1-8B-Instruct; DR=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
C=$HOME/hspec_ckpt; mkdir -p $C
python scripts/exp69_copymargin.py --target $LI --copy "" --n 60 --cache $C/seqs_l31.pt 2>&1 | tail -3
python scripts/exp71_fidelity.py --target $LI --draft $DR --copy "" --cache $C/seqs_l31.pt \
  --m 8 --m-all 4 9 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
