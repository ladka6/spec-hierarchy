#!/usr/bin/env bash
#SBATCH --job-name=sparsedraft
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=sparsedraft-%j.out
# exp61: DFlash drafter over a selected feature context (sinks + recent / + top attention, true or renumbered
# positions) at 8K-32K, on unrelated history and on a recall task that needs distant text; with and without the
# KV-sparse 4-bit copy refresh.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/sparsedraft"
mkdir -p "$HSPEC_RESULTS"
F="grep --line-buffered -v -i warn"
python scripts/exp61_sparsedraft.py --ctx 8192 16384 32768 --settings history recall --budgets 1024 3072 \
  --n 6 --anchors 16 --out exp61_sparsedraft 2>&1 | $F
