#!/usr/bin/env bash
#SBATCH --job-name=joint
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --output=joint-%j.out
# Joint training: drafter fine-tuned (fp32 master weights) together with the shallow adapter. Same data / steps as
# job_shallow (frozen drafter) for a direct comparison; 25% of blocks keep real features only.
#   k10_a2_joint    10 target layers + 2 adapter layers, drafter trained
#   k18_a2_joint    18 target layers + 2 adapter layers, drafter trained
#   k10_a0_joint    10 target layers + MLP adapter, drafter trained
#   k10_a0_joint_noadapt  10 target layers, no adapter (missing slices = layer-10 state), drafter trained:
#                   does the adapter matter once the drafter adapts?
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_joint
mkdir -p "$W"
ls "$D"/shard*.pt >/dev/null 2>&1 || { echo "no training data in $D"; exit 1; }
i=0
for C in "10 2 x" "18 2 x" "10 0 x" "10 0 --freeze-adapter"; do
  set -- $C
  X=""; N=k$1_a$2; [ "$3" != x ] && { X=$3; N=${N}_noadapt; }
  CUDA_VISIBLE_DEVICES=$i python scripts/train_shallow.py --exit $1 --adapter-layers $2 --train-drafter $X --data "$D" \
    --out "$W/$N" ${JOINT_ARGS:-} > "$W/train_$N.log" 2>&1 &
  i=$((i + 1))
done
wait
grep -h "step [0-9]*\]\|saved\|Error\|error" "$W"/train_*.log | grep --line-buffered -v -i warn
