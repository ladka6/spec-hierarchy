#!/usr/bin/env bash
#SBATCH --job-name=copytrain
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:30:00
#SBATCH --output=copytrain-%j.out
# exp55: teacher signals from the target, then LoRA-train two cheap copies (2-bit RTN, 18 of 36 layers) for speculation.
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
python scripts/exp55_copytrain.py --phase teacher --root /scratch-shared/$USER/hspec_domains --teacher $W/teacher.pt 2>&1 | $F
for B in rtn2 keep18; do
  python scripts/exp55_copytrain.py --phase train --base $B:Qwen/Qwen3-8B --teacher $W/teacher.pt --out $W/copy_$B 2>&1 | $F
done
