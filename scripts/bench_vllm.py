"""Per-forward decode latency in vLLM (CUDA graphs, fast int4/fp8 kernels) at batch size 1.

Run with the Python of a separate vLLM environment (vLLM pins its own torch):

  python -m venv /content/vllm_env
  /content/vllm_env/bin/pip install -q vllm
  /content/vllm_env/bin/python scripts/bench_vllm.py --model Qwen/Qwen3-8B
  /content/vllm_env/bin/python scripts/bench_vllm.py --model Qwen/Qwen3-8B-AWQ

Measures ms per decode step as (T(1 + n) - T(1)) / n with a fixed prompt, greedy, ignoring
EOS. Each model runs in its own process (vLLM keeps the GPU memory until exit). Results are
appended to results/bench_vllm.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/bench_vllm.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=args.model, quantization=args.quantization, max_model_len=args.ctx + args.n + 64,
              gpu_memory_utilization=0.85, enable_prefix_caching=False)
    prompt = TokensPrompt(prompt_token_ids=list(range(1000, 1000 + args.ctx)))

    def run(n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate([prompt], sp, use_tqdm=False)
        return time.perf_counter() - t0

    for _ in range(2):
        run(8)
    short = sorted(run(1) for _ in range(args.reps))[args.reps // 2]
    long = sorted(run(1 + args.n) for _ in range(args.reps))[args.reps // 2]
    ms = (long - short) / args.n * 1000
    print(f"{args.model}: {ms:.2f} ms per decode step (prefill+1 token {short * 1000:.1f} ms)")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {"rows": []}
    data["rows"] = [r for r in data["rows"] if r["model"] != args.model]
    data["rows"].append({"model": args.model, "decode_ms": ms, "prefill_ms": short * 1000,
                         "ctx": args.ctx})
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
