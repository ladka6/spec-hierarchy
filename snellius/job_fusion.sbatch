#!/usr/bin/env bash
#SBATCH --job-name=fusion
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=10:00:00
#SBATCH --output=fusion-%j.out
# Fusion vs K/V injection: four fine-tunes of the DFlash drafter in parallel, same data / steps / seed.
#   kv (control) | fuse_in | fuse_all | kv_fuse     (see hspec/fusion.py)
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_fusion
mkdir -p "$W"
ls "$D"/shard*.pt >/dev/null 2>&1 || { echo "no training data in $D (run the lag data generation first)"; exit 1; }
i=0
for M in kv fuse_in fuse_all kv_fuse; do
  CUDA_VISIBLE_DEVICES=$i python scripts/train_fusion.py --mode $M --data "$D" --out "$W/$M" ${FUSION_ARGS:-} \
    > "$W/train_$M.log" 2>&1 &
  i=$((i + 1))
done
wait
grep -h "val\|saved\|Error\|error" "$W"/train_*.log | grep --line-buffered -v -i warn
