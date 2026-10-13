#!/usr/bin/env bash
#SBATCH --job-name=kvshare
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:30:00
#SBATCH --output=kvshare-%j.out
# exp56 --kv: the copy attends to the target's KV cache for the committed prefix (vs its own), T=0.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/kvshare"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
W=/scratch-shared/$USER/hspec_copytrain
mkdir -p $W
for C in bnb4:Qwen/Qwen3-8B lora:$W/copy_rtn2 lora:$W/copy_keep18 rtn2:Qwen/Qwen3-8B; do
  N=$(basename ${C#*:}); [ "$C" = rtn2:Qwen/Qwen3-8B ] && N=rtn2_untrained
  for K in own target; do
    [ "$N" = rtn2_untrained ] && [ $K = own ] && continue
    python scripts/exp56_wide.py --copy $C --kv $K --temp 0.0 --out exp56_${N}_kv$K 2>&1 | $F | grep --line-buffered -v "anchors$"
  done
done
