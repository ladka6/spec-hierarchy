#!/usr/bin/env bash
#SBATCH --job-name=llprobe
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=llprobe-%j.out
# exp87 future probes on LayerSkip Llama3-8B and plain Llama3-8B (control): how much does each layer's state know
# about tokens t+1 .. t+16? Compare with the Qwen3-8B run (hs[3] at t+1: 0.31). No training.
#   sbatch snellius/job_ll_probe.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
export HSPEC_RESULTS="$SLURM_SUBMIT_DIR/results/llprobe_$SLURM_JOB_ID"
W=$HOME/hspec_llama; mkdir -p $W
LAYERS="0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"
for M in facebook/layerskip-llama3-8B meta-llama/Meta-Llama-3-8B; do
  T=$(basename $M)
  python scripts/gen_traindata.py --target $M --max-prompts 1000 --out $W/data_$T 2>&1 | tail -2
  python scripts/exp87_future_probe.py --target $M --data $W/data_$T --layers $LAYERS \
    --save exp87_$T --plot $HSPEC_RESULTS/exp87_$T.png 2>&1 | $F
done
cp -r "$HSPEC_RESULTS" $HOME/
