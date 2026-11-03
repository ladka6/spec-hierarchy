#!/usr/bin/env bash
#SBATCH --job-name=exp104-eval
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=exp104-eval-%j.out
# exp104 evaluation on exp77's 500 held-out anchors: released DFlash (reference) + the trained checkpoints.
#   CKPTS="K1 K4" sbatch snellius/job_exp104_eval.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/exp104"
mkdir -p "$HSPEC_RESULTS"
P=""
for c in ${CKPTS:-K1 K4}; do P="$P /scratch-shared/$USER/mix/$c.pt"; done
python scripts/exp104_mixdraft.py eval --cache "${CACHE:-/scratch-shared/$USER/seqs.pt}" --ckpts $P --ref \
  --out "exp104_${OUTTAG:-eval}" ${EXTRA:-} 2>&1 | grep --line-buffered -v -i warn
