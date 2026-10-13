#!/usr/bin/env bash
#SBATCH --job-name=copytaps
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:30:00
#SBATCH --output=copytaps-%j.out
# exp55 tap-layer copies: only the layers DFlash reads (8 layers) and taps + bridge layers (12), trained with the
# speculation objective (teacher from job_copytrain), then exp27 on both and on the untrained 8-layer copy.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/copytrain"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
W=/scratch-shared/$USER/hspec_copytrain
mkdir -p $W
T8=keep0-1-9-17-25-33-34-35
T12=keep0-1-8-9-16-17-24-25-32-33-34-35
for B in $T8 $T12; do
  python scripts/exp55_copytrain.py --phase train --base $B:Qwen/Qwen3-8B --teacher $W/teacher.pt --out $W/copy_$B 2>&1 | $F
done
for P in lora:$W/copy_$T8 lora:$W/copy_$T12 $T8:Qwen/Qwen3-8B; do
  N=$(basename ${P#*:}); [ "$P" = $T8:Qwen/Qwen3-8B ] && N=taps8_untrained
  python scripts/exp27_refresh.py --no-tokens --no-fresh --proxies $P --out exp27_$N 2>&1 | $F
done
