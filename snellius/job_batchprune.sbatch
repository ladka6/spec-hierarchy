#!/usr/bin/env bash
#SBATCH --job-name=batchprune
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=batchprune-%j.out
# exp59: the copy over a DDTree-64 (one tree forward on the target's KV) + restarts, per-anchor dump (GPU), then the
# replay under the batch cost curves of job_bench_batch.sbatch if they exist (CPU). The replay can be rerun alone:
#   python scripts/exp59_batchprune.py replay --dump results/batchprune/exp59_dump.json \
#       --target-costs results/bench_vllm_batch.json --copy-costs results/bench_vllm_batch.json
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/batchprune"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
python scripts/exp59_batchprune.py collect --copy bnb4:Qwen/Qwen3-8B --big 64 --out exp59_dump 2>&1 | $F
C=results/bench_vllm_batch.json
if [ -f "$C" ]; then
  python scripts/exp59_batchprune.py replay --dump results/batchprune/exp59_dump.json \
    --target-costs "$C" --copy-costs "$C" --out exp59_replay 2>&1 | $F
else
  echo "no $C yet: run job_bench_batch.sbatch, then the replay command in this file's header"
fi
