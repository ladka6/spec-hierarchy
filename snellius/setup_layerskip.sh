#!/usr/bin/env bash
# Run once on a LOGIN node. The LayerSkip checkpoints are gated (FAIR noncommercial research
# license): first accept the license on each model page on huggingface.co with your HF account,
# then `hf auth login` (or export HF_TOKEN) here.
#   https://huggingface.co/facebook/layerskip-llama3-8B
#   https://huggingface.co/meta-llama/Meta-Llama-3-8B   (control: same model without LayerSkip)
set -euo pipefail
source "${HSPEC_VENV:-$HOME/venvs/hspec}/bin/activate"
# the login token lives in the default HF home; pick it up before moving HF_HOME to scratch
export HF_TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
[ -n "$HF_TOKEN" ] || echo "no HF token found: run 'hf auth login' first"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf}
for m in facebook/layerskip-llama3-8B meta-llama/Meta-Llama-3-8B; do
  hf download "$m" --exclude "original/*" --exclude "*.pth" || echo "FAILED: $m (license accepted? logged in?)"
done
