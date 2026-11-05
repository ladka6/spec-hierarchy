#!/usr/bin/env bash
#SBATCH --job-name=exp106
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=exp106-%j.out
# exp106: how close is the exit state h_k to the deep taps DFlash reads, and what does the frozen drafter do if the
# missing taps of the last 8 positions are filled from h_k (mean / identity / scaled / closed-form ridge)? No training.
#   1) LayerSkip Llama3-8B (exit-trained) with the original Llama-3.1 DFlash (mismatched drafter: compare rows, not
#      absolute tau)
#   2) Llama-3.1-8B-Instruct with its native drafter (control: no exit training)
#   sbatch snellius/job_exp106.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp106_$SLURM_JOB_ID"
DR=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
python scripts/exp106_shallow_fill.py --target facebook/layerskip-llama3-8B --draft $DR \
  --data $HOME/hspec_llama/data_layerskip-llama3-8B --exits 4 8 16 --save exp106_layerskip 2>&1 | $F
python scripts/exp106_shallow_fill.py --target meta-llama/Llama-3.1-8B-Instruct --draft $DR \
  --cache $HOME/hspec_ckpt/seqs_l31.pt --exits 4 8 16 --fit-tokens 35000 --val-tokens 5000 \
  --save exp106_llama31 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
