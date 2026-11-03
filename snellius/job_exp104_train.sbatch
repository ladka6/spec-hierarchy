#!/usr/bin/env bash
#SBATCH --job-name=exp104-train
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=exp104-train-%j.out
# exp104: train one mixture block drafter (K plans; K=1 is the control) on the fork-branched data.
#   K=1 ALTW=0 NAME=K1 sbatch snellius/job_exp104_train.sbatch     control: one plan, standard objective
#   K=4 NAME=K4 sbatch snellius/job_exp104_train.sbatch            4 plans, soft mixture likelihood
# Variables: K, LAYERS (5; 8 needs more than 40 GB: add -p gpu_h100), INIT (scratch | target | dflash), ALTW (1.0),
# EPOCHS (2), NAME, EXTRA (more flags, e.g. "--assign hard" or "--steps 30 --limit-seqs 200 --eval-every 0").
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
NAME=${NAME:-K${K:-4}}
python scripts/exp104_mixdraft.py train --data ${DATA:-/scratch-shared/$USER/forkdata} \
  --cache "${CACHE:-/scratch-shared/$USER/seqs.pt}" --K "${K:-4}" --layers "${LAYERS:-5}" --init "${INIT:-scratch}" \
  --alt-w "${ALTW:-1.0}" --epochs "${EPOCHS:-2}" --ckpt "/scratch-shared/$USER/mix/${NAME}.pt" ${EXTRA:-} \
  2>&1 | grep --line-buffered -v -i warn
