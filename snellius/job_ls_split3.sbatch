#!/usr/bin/env bash
#SBATCH --job-name=ls-split3
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=ls-split3-%j.out
# Split verification, round 3 on LayerSkip Llama3-8B:
#   1) split cost ratios for exits 16 / 20 / 24 (lower + exit head, lower alone, upper)
#   2) trained linear exit heads at 16 / 20 / 24 (KL to the final layer)
#   3) exp14: merged upper pass (-x), joint gate (-j), trained heads (-h), exit 24
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
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/lssplit3_$SLURM_JOB_ID"
F="grep --line-buffered -v -i warn"

python scripts/bench_split.py --target $TGT --exits 16 20 24 2>&1 | $F
if [ ! -f "$W/exit_heads_ls.pt" ]; then
  python scripts/train_exit.py --target $TGT --data "$W/data" --out "$W/exit_heads_ls.pt" \
    --layers 16 20 24 --kinds lin 2>&1 | $F
fi
python scripts/exp14_split.py --target $TGT --n 10 --tag r3 \
  --base-draft "$W/drafter_ls" --split-draft "$W/drafter_ls_depth16" \
  --bench "$HSPEC_RESULTS/bench_split.json" --exit-heads "$W/exit_heads_ls.pt" \
  --configs base:dflash base:ddtree-64 \
            split-k20-t0.3-b32-m4 \
            split-k20-t0.3-b32-m4-x \
            split-k20-t0.3-b32-m4-j0.1-x \
            split-k20-t0.3-b32-m4-j0.1-x-h \
            split-k20-t0.5-b32-m6-j0.1-x-h \
            split-k20-t0.3-b64-m4-j0.1-x-h \
            split-k16-t0.3-b32-m4-j0.1-x-h \
            split-k24-t0.3-b32-m4-j0.1-x \
            split-k24-t0.3-b32-m4-j0.1-x-h \
            split-k24-t0.5-b32-m6-j0.1-x-h \
  2>&1 | $F
