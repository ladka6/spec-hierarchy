#!/usr/bin/env bash
#SBATCH --job-name=exp110c
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=exp110c-%j.out
# exp110 calibration for the full fine-tune budget: time per optimizer step and peak memory of train_lag.py
# at the planned settings (block 10 from the drafter, 512 anchors/seq, lag 0, DFlash-like gamma), on the
# existing LayerSkip data. 400 seqs / batch 4 = 100 steps. Read "s/step" and the peak MiB line at the end.
#   sbatch snellius/job_exp110_calib.sbatch [BLOCKS]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load 2024 2>/dev/null && module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { module load 2023 && module load Python/3.11.3-GCCcore-12.3.0; }
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${LS_DIR:-/scratch-shared/$USER/hspec_ls}
BLOCKS=${1:-512}
[ -d "$W/data" ] || { echo "no data in $W/data (run job_ls_gen first)"; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "exp110c-$SLURM_JOB_ID.mem" &
SMI=$!
python scripts/train_lag.py --target facebook/layerskip-llama3-8B --draft z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat \
  --data "$W/data" --out "$W/calib_tmp" --max-lag 0 --blocks $BLOCKS --batch 4 --epochs 1 --max-train 400 \
  --val-seqs 8 --eval-every 100000 --gamma 0.82 --lr 5e-5 2>&1 | grep --line-buffered -v -i warn
kill $SMI
echo "peak MiB: $(sort -n exp110c-$SLURM_JOB_ID.mem | tail -1)"
python - <<PY
import torch, glob
d=[]; [d.extend(torch.load(f)) for f in sorted(glob.glob("$W/data/shard*.pt"))]
r=[len(x["ids"])-x["n_prompt"] for x in d]; t=[len(x["ids"]) for x in d]
r.sort(); print(f"data: {len(d)} seqs, response mean {sum(r)/len(r):.0f} median {r[len(r)//2]} tokens, total len mean {sum(t)/len(t):.0f}")
PY
