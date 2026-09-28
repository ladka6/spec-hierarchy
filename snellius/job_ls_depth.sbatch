#!/usr/bin/env bash
#SBATCH --job-name=ls-depth
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=ls-depth-%j.out
# LayerSkip Llama3-8B, one-GPU three-level prerequisites:
#   1) cost of split verification (lower 16/20 layers + exit head, upper layers, full pass)
#   2) depth-lag drafter: fine-tune drafter_ls with shallow-only features (exit after 16
#      layers) for the last 0..24 positions
#   3) exp13: acceptance with shallow-only recent features, adapted vs depth-trained drafter
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
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/lsdepth_$SLURM_JOB_ID"
F="grep --line-buffered -v -i warn"

python scripts/bench_split.py --target $TGT --exits 16 20 2>&1 | $F
python scripts/train_lag.py --target $TGT --draft "$W/drafter_ls" --data "$W/data" --out "$W/drafter_ls_depth16" \
  --depth-exit 16 --max-lag 24 --p-zero 0.25 2>&1 | $F
python scripts/exp13_depth_eval.py --target $TGT --exit-layer 16 \
  --drafters base="$W/drafter_ls" depth16="$W/drafter_ls_depth16" 2>&1 | $F
