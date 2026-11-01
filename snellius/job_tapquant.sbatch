#!/usr/bin/env bash
#SBATCH --job-name=tapquant
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --output=tapquant-%j.out
# exp88 tap-aware quantization: learn only the 2-bit quantizer (step size, offset per group of 64) with
#   feat   loss on the drafter's fused input (what DFlash sees)          [tap-aware]
#   lm     next-token loss on the target's greedy text                  [standard QAT objective]
#   both   feat + 0.1 x next-token
# then the exp55 agreement metrics on the same validation split (comparable with the LoRA copies).
# Each copy is saved as a bf16 HF checkpoint (16 GB) in $HOME/hspec_ckpt/tapquant_<variant>.
#   sbatch --export=ALL snellius/job_tapquant.sbatch
#   VARIANTS="feat" EXTRA="--hi-layers 24 25 32 33 --hi-bits 3" TAG=mixed sbatch --export=ALL snellius/job_tapquant.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/tapquant"
mkdir -p "$HSPEC_RESULTS" /scratch-shared/$USER/tapquant
F="grep --line-buffered -v -i warn"
SHARDS=${SHARDS:-/scratch-shared/$USER/hspec_lagft/data}
for V in ${VARIANTS:-feat lm both}; do
  case $V in
    feat) A="--w-feat 1 --w-agree 0" ;;
    lm)   A="--w-feat 0 --w-agree 1" ;;
    both) A="--w-feat 1 --w-agree 0.1" ;;
  esac
  OUT=$HOME/hspec_ckpt/tapquant_${V}${TAG:+_$TAG}
  echo "[job] variant $V: $A ${EXTRA:-} -> $OUT"
  python scripts/exp88_tapquant.py --shards $SHARDS --teacher /scratch-shared/$USER/tapquant/teacher.pt $A \
    ${EXTRA:-} --out $OUT 2>&1 | $F
  python scripts/exp55_copytrain.py --phase train --base hf:$OUT --teacher $HOME/hspec_ckpt/copyrole/teacher.pt \
    --epochs 0 2>&1 | grep "eval step"
done
