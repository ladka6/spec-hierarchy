#!/usr/bin/env bash
#SBATCH --job-name=slotdfl
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=54
#SBATCH --time=01:15:00
#SBATCH --output=slotdflash-%j.out
# Fine-tune the official DFlash drafter with (a) no extra context, (b) the lookahead-slot features from the
# saved LoRA, (c) the true features (upper bound) at positions s..s+7. Same data / steps / lr; tau curves.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_lookahead
L=$W/lora.pt
[ -f "$L" ] || { echo "no LoRA at $L (run job_lora_save first)"; exit 1; }
i=0
for A in control slots oracle; do
  CUDA_VISIBLE_DEVICES=$i python scripts/train_slotdflash.py --arm $A --lora "$L" --data "$D" --out "$W/dflash_$A" \
    ${SD_ARGS:-} > "$W/train_dflash_$A.log" 2>&1 &
  i=$((i + 1))
done
wait
grep -h "step [0-9]*\] \|done\|Error" "$W"/train_dflash_*.log | grep -v -i warn
