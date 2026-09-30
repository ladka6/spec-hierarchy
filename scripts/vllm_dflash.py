"""Batch-1 decode speed in vLLM, plain vs DFlash, and the per-round split it implies.

Runs in the vLLM environment. Greedy, one request at a time, prompts from dump_prompts.py.
Decode ms/token = (generation time - time to first token) / (tokens - 1). With DFlash, vLLM's
spec-decode counters give rounds and tau; verify of a block is ~ one plain decode step (memory-
bound), so draft_share ~ 1 - ar_step / round_ms and the async ceiling ~ 1 / (1 - draft_share).

  python scripts/vllm_dflash.py --prompts p.json --tp 1 [--spec] --out r.json
"""

from __future__ import annotations

import argparse
import json
import time

from vllm import LLM, SamplingParams


def spec_counters(llm):
    out = {}
    try:
        for m in llm.get_metrics():
            if "spec_decode" in m.name and hasattr(m, "value"):
                out[m.name] = out.get(m.name, 0) + m.value
    except Exception as e:                      # metrics API differs between versions
        out["error"] = repr(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kw = dict(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096, gpu_memory_utilization=0.85,
              dtype="bfloat16", disable_log_stats=False)
    if args.spec:
        kw["speculative_config"] = {"method": "dflash", "model": args.draft, "num_speculative_tokens": args.k}
    llm = LLM(**kw)
    prompts = json.load(open(args.prompts))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    for p in prompts[:2]:                       # warmup
        llm.generate([p["text"]], sp, use_tqdm=False)
    c0 = spec_counters(llm)
    t_dec, n_dec = 0.0, 0
    for p in prompts:
        t = time.perf_counter()
        llm.generate([p["text"]], sp1, use_tqdm=False)
        ttft = time.perf_counter() - t
        t = time.perf_counter()
        o = llm.generate([p["text"]], sp, use_tqdm=False)
        dt = time.perf_counter() - t
        n = len(o[0].outputs[0].token_ids)
        t_dec += dt - ttft
        n_dec += n - 1
    c1 = spec_counters(llm)
    res = {"tp": args.tp, "spec": args.spec, "prompts": len(prompts), "decode_ms_per_token": 1e3 * t_dec / n_dec,
           "tokens": n_dec, "spec_counters_delta": {k: c1.get(k, 0) - c0.get(k, 0) for k in c1 if k != "error"},
           "metrics_error": c1.get("error")}
    d = res["spec_counters_delta"]
    drafts = next((v for k, v in d.items() if k.endswith("num_drafts")), 0)
    acc = next((v for k, v in d.items() if k.endswith("num_accepted_tokens")), 0)
    if drafts:
        # the ttft call per prompt also runs no draft rounds of note; rounds ~ drafts
        res["tau"] = acc / drafts + 1
        res["round_ms"] = 1e3 * t_dec / drafts
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
