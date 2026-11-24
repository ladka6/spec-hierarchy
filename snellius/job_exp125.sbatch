#!/usr/bin/env bash
#SBATCH --job-name=exp125
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=exp125-%j.out
# exp125: learned r-dim code of the real features at the in-flight positions, drafter + code trained on the block
# loss only (one arm per job). CODE = 0 (lag baseline) | r | real.
#   bash snellius/submit_exp125.sh            # arms 0 16 64 256 real
#   sbatch --export=ALL,CODE=64 snellius/job_exp125.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
DATA=${DATA:-/scratch-shared/$USER/hspec_lagft/data}
DRAFT=${DRAFT:-$HOME/hspec_ckpt/drafter_lag16_fp32}
CODE=${CODE:-64}
OUT=${OUT:-/scratch-shared/$USER/hspec_code/r$CODE}
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp125_$SLURM_JOB_ID"
echo "[job] CODE=$CODE DRAFT=$DRAFT OUT=$OUT EXTRA=${EXTRA:-}"
python scripts/exp125_code_bottleneck.py --data "$DATA" --draft "$DRAFT" --code "$CODE" --out "$OUT" \
  --epochs ${EPOCHS:-1} ${EXTRA:-} --save exp125_code_r$CODE 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
