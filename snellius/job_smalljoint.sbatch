#!/usr/bin/env bash
#SBATCH --job-name=smalljoint
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --output=smalljoint-%j.out
# Proposal 1, trained: a small same-family model (Qwen3-0.6B / 1.7B) + mapper (ridge init + residual MLP) predicts
# the 8B verifier's DFlash features for the last 1-16 drafted positions. Same data / steps / validation as the
# shallow, joint and low-bit runs. Columns: "shallow" = ridge init (exp32), "adapter" = trained mapper.
#   Qwen3-0.6B_joint / Qwen3-1.7B_joint   mapper + drafter trained
#   Qwen3-0.6B / Qwen3-1.7B               mapper trained, original drafter
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
W=/scratch-shared/$USER/hspec_smalljoint
mkdir -p "$W"
ls "$D"/shard*.pt >/dev/null 2>&1 || { echo "no training data in $D"; exit 1; }
i=0
for C in "0.6B --train-drafter" "1.7B --train-drafter" "0.6B" "1.7B"; do
  set -- $C
  N=Qwen3-$1${2:+_joint}
  CUDA_VISIBLE_DEVICES=$i python scripts/train_shallow.py --small Qwen/Qwen3-$1 ${2:-} --data "$D" --out "$W/$N" \
    ${SMALL_ARGS:-} > "$W/train_$N.log" 2>&1 &
  i=$((i + 1))
done
wait
grep -h "step [0-9]*\]\|saved\|Error\|error" "$W"/train_*.log | grep --line-buffered -v -i warn
