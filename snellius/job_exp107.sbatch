#!/usr/bin/env bash
#SBATCH --job-name=exp107
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:30:00
#SBATCH --output=exp107-%j.out
# exp107: where does "partial pass + closed-form map" break? A1 chain length (m = 4..32), A2 map inputs (h_k / stacked
# shallow layers / verified deep context / both), A3 per-tap ablation + confidence splits, A4 fork resolution by the
# lifted final state. No gradient training.
#   1) Qwen3-8B, native drafter (clean drafter, target without exit training)
#   2) LayerSkip Llama3-8B, original Llama-3.1 drafter (exit-trained target, mismatched drafter: compare rows)
#   sbatch snellius/job_exp107.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp107_$SLURM_JOB_ID"
python scripts/exp107_lift_diag.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
  --cache $HOME/seqs.pt --exits 4 8 --save exp107_qwen3 2>&1 | $F
python scripts/exp107_lift_diag.py --target facebook/layerskip-llama3-8B \
  --draft z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat --data $HOME/hspec_llama/data_layerskip-llama3-8B \
  --exits 4 8 --save exp107_layerskip 2>&1 | $F
cp -r "$HSPEC_RESULTS" $HOME/
