#!/usr/bin/env bash
#SBATCH --job-name=lowbit
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=lowbit-%j.out
# exp27 with lower-bit target copies (simulated 4/3/2-bit round-to-nearest) + forward latency bench (bf16 target, 4-bit, drafter).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/lowbit"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
L=/scratch-shared/$USER/hspec_lagft/drafter_lag16
for B in 4 3 2; do
  python scripts/exp27_refresh.py --no-tokens --no-fresh --proxies rtn$B:Qwen/Qwen3-8B --out exp27_rtn${B}_t0 2>&1 | $F
done
python scripts/bench_forward.py --mode eager compiled --mids bnb4:Qwen/Qwen3-8B --ctx 1024 --qs 1 16 32 64 128 2>&1 | $F
