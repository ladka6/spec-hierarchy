#!/usr/bin/env bash
#SBATCH --job-name=rollout
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=rollout-%j.out
# exp51: leak-free evaluation of the 'cur' checkpoint (sets from a greedy draft of the original model), plus
# the causal block-routed checkpoint under the same sets. One GPU, evaluation only.
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
D=${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data
W=/scratch-shared/$USER/hspec_moe
for C in routetrain_router_experts_S16_1000_routecur routetrain_router_experts_S16_1000; do
  echo "== $C"
  python scripts/exp51_rollout_eval.py --ckpt "$W/$C" --S 16 --n 16 --data "$D" \
    2>&1 | grep --line-buffered -v -i warn | grep --line-buffered "^\[\|loaded\|rollout\]\|rror"
done
