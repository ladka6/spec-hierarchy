#!/usr/bin/env bash
#SBATCH --job-name=lowbit-joint
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --output=lowbitjoint-%j.out
# Missing depth vs noisy depth: the drafter's context for the last 1-16 positions comes from a FULL-depth
# low-bit copy of the target (simulated round-to-nearest, group 64), drafter fine-tuned (fp32 master weights).
# Same data / steps / validation as job_shallow and job_joint. Step 0 = the original (frozen) drafter.
#   rtn2_joint   2-bit copy (failed with the original drafter: offline 8.44 vs 7.98 plain)
#   rtn3_joint   3-bit copy (12.09 offline with the original drafter)
# Each run uses two GPUs: drafter + target on the first, the copy on the second.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_lowbit_joint
mkdir -p "$W"
ls "$D"/shard*.pt >/dev/null 2>&1 || { echo "no training data in $D"; exit 1; }
for B in 2 3; do
  G=$(( (B - 2) * 2 ))
  CUDA_VISIBLE_DEVICES=$G,$((G + 1)) python scripts/train_shallow.py --copy rtn$B:Qwen/Qwen3-8B --train-drafter \
    --device cuda:0 --copy-device cuda:1 --data "$D" --out "$W/rtn$B" ${JOINT_ARGS:-} > "$W/train_rtn$B.log" 2>&1 &
done
wait
grep -h "step [0-9]*\]\|saved\|Error\|error" "$W"/train_*.log | grep --line-buffered -v -i warn
