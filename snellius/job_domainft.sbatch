#!/usr/bin/env bash
#SBATCH --job-name=domft
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=domft-%j.out
# Drafter forgetting study: baseline, per-domain fine-tunes, mixed fine-tune, sequential math -> code.
# tau on every domain's held-out set at each eval. Usage: sbatch snellius/job_domainft.sbatch [steps]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/horizon"
mkdir -p "$HSPEC_RESULTS"
W=/scratch-shared/$USER/hspec_domains
S=${1:-1000}
run() { python scripts/train_domaindflash.py --root "$W" --steps $S --eval-every 500 "$@" \
          2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "tau\|train \|saved\|rror\|s/seq"; }
run --out "$W/runs/base"
run --train math  --save "$W/drafters/math.pt"  --out "$W/runs/math"
run --train code  --save "$W/drafters/code.pt"  --out "$W/runs/code"
run --train think --save "$W/drafters/think.pt" --out "$W/runs/think"
run --train math code chat think --save "$W/drafters/mix.pt" --out "$W/runs/mix"
run --train code --init "$W/drafters/math.pt" --out "$W/runs/math_then_code"
