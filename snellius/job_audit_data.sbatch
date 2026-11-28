#!/usr/bin/env bash
#SBATCH --job-name=auditdata
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=auditdata-%j.out
# Share of stored response tokens that are the loaded target's argmax (see scripts/audit_data_greedy.py).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python scripts/audit_data_greedy.py --data /scratch-shared/$USER/hspec_lagft/data 2>&1 | grep -v -i warn
for pair in "Qwen/Qwen3-8B qwen3_8b" "meta-llama/Llama-3.1-8B-Instruct llama31_8b"; do
  set -- $pair
  D=/scratch-shared/$USER/hspec_taps/$2
  [ -d "$D" ] && python scripts/audit_data_greedy.py --target "$1" --data "$D" 2>&1 | grep -v -i warn
done
