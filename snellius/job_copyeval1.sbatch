#!/usr/bin/env bash
#SBATCH --job-name=copyeval1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=copyeval1-%j.out
# exp27 offline refresh (T=0) for ONE copy, so several copies run in parallel on the same prompts (same seed).
#   P=lora:$HOME/hspec_ckpt/copyrole/copy_role N=role sbatch --export=ALL snellius/job_copyeval1.sbatch
#   P=rtn2:Qwen/Qwen3-8B N=rtn2_untrained sbatch --export=ALL snellius/job_copyeval1.sbatch
#   P=rtn4:Qwen/Qwen3-8B N=rtn4_untrained sbatch --export=ALL snellius/job_copyeval1.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/copyrole"
mkdir -p "$HSPEC_RESULTS"
echo "[job] P=$P N=$N"
python scripts/exp27_refresh.py --no-tokens --no-fresh --proxies $P --out exp27_$N ${EXTRA:-} 2>&1 \
  | grep --line-buffered -v -i warn
