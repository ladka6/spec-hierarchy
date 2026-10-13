#!/usr/bin/env bash
#SBATCH --job-name=bridgefit
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:45:00
#SBATCH --output=bridgefit-%j.out
# exp57 diagnostics, evaluation only (no training): taps8 with plain skips (no bridges), closed-form affine bridges
# fitted independently (as job_bridges) and sequentially (each span on the copy's own inputs).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
F="grep --line-buffered -v -i warn"
W=/scratch-shared/$USER/hspec_copytrain
echo "== plain skip (no bridges)"
python scripts/exp57_bridges.py --kind lin --fit-seqs 0 --teacher $W/teacher.pt 2>&1 | $F
for R in 1e-3 1e-1; do
  echo "== sequential fit, ridge $R"
  python scripts/exp57_bridges.py --kind lin --fit-mode seq --ridge $R --teacher $W/teacher.pt --out $W/bridge_linseq_$R 2>&1 | $F
done
echo "== independent fit, ridge 1e-1"
python scripts/exp57_bridges.py --kind lin --ridge 1e-1 --teacher $W/teacher.pt 2>&1 | $F
