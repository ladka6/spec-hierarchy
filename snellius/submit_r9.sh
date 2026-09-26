#!/usr/bin/env bash
# Submit the measured feature-ready reserve experiment from a Snellius login node.
#   bash snellius/submit_r9.sh           # three GPUs, one per stage
#   bash snellius/submit_r9.sh 2         # middle and draft share a GPU
#   bash snellius/submit_r9.sh both      # submit both placements as separate jobs
#   DRY_RUN=1 bash snellius/submit_r9.sh both
# Optional environment: HSPEC_N, HSPEC_MAX_NEW, HSPEC_REPEATS, HSPEC_TIME,
# HSPEC_ACCOUNT, HSPEC_OUT, HSPEC_VENV, HF_HOME (see job_r9.sbatch).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
case ${1:-3} in
  2) PLACEMENTS=(2) ;;
  3) PLACEMENTS=(3) ;;
  both) PLACEMENTS=(2 3) ;;
  *) echo "Usage: bash snellius/submit_r9.sh [2|3|both]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: bash snellius/submit_r9.sh [2|3|both]" >&2
  exit 2
fi
OUT=${HSPEC_OUT:-$REPO/results/r9_$(date +%Y%m%d_%H%M%S)_$$}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
COMMIT=$(git rev-parse HEAD)
if ! git diff --quiet HEAD --; then
  echo "Commit tracked changes before submitting: jobs run a snapshot of HEAD." >&2
  exit 2
fi
if ! git cat-file -e "$COMMIT:scripts/exp9_reserve.py" 2>/dev/null; then
  echo "This commit has no reserve experiment. Switch to codex/feature-ready-reserve first." >&2
  exit 2
fi
SOURCE=$(mktemp -d "$OUT/source_${COMMIT:0:12}_XXXXXX")
git archive "$COMMIT" | tar -x -C "$SOURCE"
printf '%s\n' "$COMMIT" > "$SOURCE/.hspec_commit"
echo "Source snapshot: $SOURCE ($COMMIT)"
for GPUS in "${PLACEMENTS[@]}"; do
  # gpu_a100 allocates 18 CPU cores per GPU; all stages stay on the same node.
  CMD=(sbatch --parsable --export=ALL --partition=gpu_a100 --nodes=1 --ntasks=1
       --gpus-per-node="$GPUS" --cpus-per-task="$((18 * GPUS))"
       --time="${HSPEC_TIME:-08:00:00}" --job-name="hspec-r9-${GPUS}g"
       --output="$OUT/slurm_${GPUS}gpu_%j.out")
  if [[ -n ${HSPEC_ACCOUNT:-} ]]; then
    CMD+=(--account="$HSPEC_ACCOUNT")
  fi
  CMD+=("$SOURCE/snellius/job_r9.sbatch" "$SOURCE" "$OUT" "$GPUS")
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
  else
    JOB=$("${CMD[@]}")
    echo "Submitted ${GPUS}-GPU experiment: $JOB"
  fi
done
echo "Results and logs: $OUT"
