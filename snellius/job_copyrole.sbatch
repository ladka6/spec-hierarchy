#!/usr/bin/env bash
#SBATCH --job-name=copyrole
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
#SBATCH --output=copyrole-%j.out
# Role-trained low-bit copy, main comparison: the same 2-bit RTN copy + LoRA trained with
#   both  drafter loss + 0.5 x next-token loss   (the exp55 recipe)
#   role  drafter loss only                      (what the drafter needs)
#   lm    next-token loss only                   (the copy trained as a language model, standard QAT-style objective)
# on the same data (lag training shards) and budget. Checkpoints in $HOME (scratch purges).
#   sbatch --export=ALL snellius/job_copyrole.sbatch
#   then: for V in both role lm; do P=lora:$HOME/hspec_ckpt/copyrole/copy_$V N=$V sbatch --export=ALL snellius/job_copyeval1.sbatch; done
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
F="grep --line-buffered -v -i warn"
W=$HOME/hspec_ckpt/copyrole
mkdir -p $W
SHARDS=${SHARDS:-/scratch-shared/$USER/hspec_lagft/data}
NTRAIN=${NTRAIN:-2400}
[ -f $W/teacher.pt ] || python scripts/exp55_copytrain.py --phase teacher --shards $SHARDS --n-train $NTRAIN \
  --teacher $W/teacher.pt 2>&1 | $F
for V in ${VARIANTS:-both role lm}; do
  case $V in
    both) A="--w-draft 1 --w-agree 0.5" ;;
    role) A="--w-draft 1 --w-agree 0" ;;
    lm)   A="--w-draft 0 --w-agree 1" ;;
  esac
  echo "[job] variant $V: $A"
  python scripts/exp55_copytrain.py --phase train --base ${BASE:-rtn2}:Qwen/Qwen3-8B --teacher $W/teacher.pt $A \
    --out $W/copy_$V 2>&1 | $F
done
