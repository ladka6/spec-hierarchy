#!/usr/bin/env bash
#SBATCH --job-name=moecost
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=01:00:00
#SBATCH --output=moecost-%j.out
# exp42: verify-pass cost vs number of verified tokens for an MoE target (Qwen3-30B-A3B, over 2 GPUs) and the
# dense Qwen3-8B; then throughput of fixed / confidence / oracle draft lengths with DFlash acceptance data.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/horizon"
B=/scratch-shared/$USER/hspec_moe/blocks.pt
[ -f "$B" ] && BL="--blocks $B" || BL=""
python scripts/exp42_moe_cost.py --models ${MOE_MODELS:-Qwen/Qwen3-30B-A3B Qwen/Qwen3-8B} \
  --data "${LAGFT_DIR:-/scratch-shared/$USER/hspec_lagft}/data" $BL 2>&1 | grep --line-buffered -v -i warn
