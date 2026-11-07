#!/usr/bin/env bash
#SBATCH --job-name=exp109
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:30:00
#SBATCH --output=exp109-%j.out
# exp109: amortized lookahead, ORACLE ceiling. Does the Qwen3-8B DFlash drafter gain from right-hand anchors (true
# future tokens), and how fast does the gain fall with wrong anchors?
#   1) layout beyond: 2 anchors after the block (s+B+16, s+B+32), with the no-sketch fine-tune control
#   2) layout grid: a sparse grid every 16 tokens (in-block grid tokens revealed, tau_net reported), no control rerun
#   sbatch snellius/job_exp109.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp109_$SLURM_JOB_ID"
QWEN_DATA=${QWEN_DATA:-/scratch-shared/$USER/hspec_lagft/data}
python scripts/exp109_sketch_oracle.py --data $QWEN_DATA --layout beyond --period 16 --n-right 2 2>&1 | $F
python scripts/exp109_sketch_oracle.py --data $QWEN_DATA --layout grid --period 16 --skip-control 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
