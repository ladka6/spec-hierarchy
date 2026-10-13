#!/usr/bin/env bash
#SBATCH --job-name=hints
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=hints-%j.out
# exp58: the copy's features as gated hints at the restart MASK positions (DFlow-style relay from a cheap copy).
# HintMLP only, drafter frozen. Copies: 4-bit (strong) and taps8 + mlp bridges (weak, shares the target's weights).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/hints"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
W=/scratch-shared/$USER/hspec_copytrain
for C in ${COPIES:-bridge:$W/bridge_mlp bnb4:Qwen/Qwen3-8B}; do
  N=$(basename ${C#*:}); [ "$N" = Qwen3-8B ] && N=bnb4
  python scripts/exp58_hints.py --copy $C --teacher $W/teacher.pt --out $W/hints_$N --save exp58_$N 2>&1 | $F
done
