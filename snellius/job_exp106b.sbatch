#!/usr/bin/env bash
#SBATCH --job-name=exp106b
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=exp106b-%j.out
# exp106 follow-up (no training):
#   1) LayerSkip Llama3-8B with the same 35k-token ridge fit as the Llama-3.1 control (equal-data comparison)
#   2) Qwen3-8B with its native DFlash drafter (taps hs 2/10/18/26/34), exits 4 / 8 / 16, sequence cache $HOME/seqs.pt
#   sbatch snellius/job_exp106b.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp106b_$SLURM_JOB_ID"
python scripts/exp106_shallow_fill.py --target facebook/layerskip-llama3-8B \
  --draft z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat --data $HOME/hspec_llama/data_layerskip-llama3-8B \
  --exits 4 8 16 --fit-tokens 35000 --val-tokens 5000 --save exp106_layerskip_fit35k 2>&1 | $F
QCACHE=${QCACHE:-$HOME/seqs.pt}
if [ -f "$QCACHE" ]; then
  python scripts/exp106_shallow_fill.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
    --cache "$QCACHE" --exits 4 8 16 --fit-tokens 35000 --val-tokens 5000 --save exp106_qwen3 2>&1 | $F
else
  echo "[skip] Qwen run: $QCACHE not found"
fi
cp -r "$HSPEC_RESULTS" $HOME/
