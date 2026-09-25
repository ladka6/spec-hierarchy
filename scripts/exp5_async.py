"""Experiment 5: asynchronous target checks (simulated clock).

For each network delay L, compares on a virtual clock built from measured per-call costs:
  dflash, ddtree-B        baselines (time from their exact call sequence)
  sync-P+mtB              three-stage, lower stages wait for every target check
  async-P+mtB             three-stage, lower stages keep drafting while checks are in flight

Costs default to the round-4 measurements (vLLM bf16 target, vLLM AWQ middle model with
small-q values clamped to the decode cost, drafter under CUDA graphs); override with
--vllm / --draft-ms. L = 0 for async means the target runs on a second GPU.

  python scripts/exp5_async.py --n 10 --latencies-ms 0 10 50 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.async3 import Costs, async_three_stage_generate  # noqa: E402
from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import free, gpu_mem_gb, load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import ddtree_generate, two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def clamp(tab: dict[int, float], floor: float) -> dict[int, float]:
    run, out = floor, {}
    for q in sorted(tab):
        run = max(run, tab[q])
        out[q] = run
    return out


def baseline_ms(r, costs: Costs) -> float:
    """Virtual decode time of a two-stage run (one target call per round, then draft)."""
    return sum(costs.t(q) + costs.latency_ms for q in r.tq) + r.draft_calls * costs.draft_ms


def parse_cfg(name: str):
    """sync-16+mt32 / async-8 -> (blocking, window, mid_tree)."""
    mode, rest = name.split("-", 1)
    parts = rest.split("+")
    tree = int(parts[1][2:]) if len(parts) > 1 else 0
    return mode == "sync", int(parts[0]), tree


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mid", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--latencies-ms", nargs="+", type=float, default=[0, 10, 50, 100])
    ap.add_argument("--configs", nargs="+",
                    default=["sync-16+mt32", "sync-32+mt32", "async-8+mt32", "async-16+mt32",
                             "async-32+mt32", "async-16"])
    ap.add_argument("--ddtree-budget", type=int, default=64)
    ap.add_argument("--vllm", default=None, help="bench_vllm.json to take costs from")
    ap.add_argument("--vllm-target", default="Qwen/Qwen3-8B")
    ap.add_argument("--vllm-mid", default="Qwen/Qwen3-8B-AWQ")
    ap.add_argument("--draft-ms", type=float, default=3.55)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    base = Costs(draft_ms=args.draft_ms)
    if args.vllm and Path(args.vllm).exists():
        rows = {r["model"]: r for r in json.loads(Path(args.vllm).read_text())["rows"]}
        for attr, key in (("target", args.vllm_target), ("mid", args.vllm_mid)):
            r = rows.get(key)
            if r and r.get("q_ms"):
                setattr(base, attr, clamp({int(k): v for k, v in r["q_ms"].items()}, r["decode_ms"]))
    print(f"costs: target {base.target}\n       mid {base.mid}\n       draft {base.draft_ms} ms")

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    mid = load_mid(args.mid)
    stops = stop_ids(target, tok)
    prompts = {d: load_prompts(d, args.n) for d in args.datasets}
    w = encode(tok, prompts[args.datasets[0]][0])
    two_stage_generate(draft, target, target, target, w, 32, stops)

    records = []
    for d in args.datasets:
        for i, p in enumerate(prompts[d]):
            ids = encode(tok, p)
            runs = {"dflash": two_stage_generate(draft, target, target, target, ids, args.max_new, stops),
                    f"ddtree-{args.ddtree_budget}": ddtree_generate(draft, target, ids, args.max_new, stops,
                                                                    budget=args.ddtree_budget)}
            for name, r in runs.items():
                for lat in args.latencies_ms:
                    c = Costs(base.target, base.mid, base.draft_ms, lat)
                    records.append({"dataset": d, "i": i, "lat": lat, "config": name,
                                    "tokens": r.num_output_tokens, "ms": baseline_ms(r, c),
                                    "tgt_calls_per_tok": r.target_calls / max(r.num_output_tokens, 1)})
        print(f"[baselines | {d}] done", flush=True)

    for lat in args.latencies_ms:
        costs = Costs(base.target, base.mid, base.draft_ms, lat)
        for cfg in args.configs:
            blocking, window, tree = parse_cfg(cfg)
            for d in args.datasets:
                for i, p in enumerate(prompts[d]):
                    r = async_three_stage_generate(draft, target, mid, encode(tok, p), args.max_new, stops,
                                                   costs, window=window, blocking=blocking, mid_tree=tree)
                    s = r.summary()
                    records.append({"dataset": d, "i": i, "lat": lat, "config": cfg,
                                    "tokens": r.num_output_tokens, "ms": r.decode_time * 1000,
                                    "tgt_calls_per_tok": s["target_calls_per_token"],
                                    "mid_calls_per_tok": s["mid_calls_per_token"],
                                    "tau": s["mean_round_len"], "mean_q": s["mean_target_q"],
                                    **r.async_stats})
            print(f"[{cfg} | {lat} ms] done", flush=True)

    rows = []
    for key in sorted({(r["lat"], r["dataset"], r["config"]) for r in records}):
        rs = [r for r in records if (r["lat"], r["dataset"], r["config"]) == key]
        tps = sum(r["tokens"] for r in rs) / (sum(r["ms"] for r in rs) / 1000)
        rows.append({"lat_ms": key[0], "dataset": key[1], "config": key[2], "tok/s": tps,
                     "tgt_calls/tok": mean(r["tgt_calls_per_tok"] for r in rs),
                     "mid_calls/tok": mean(r.get("mid_calls_per_tok", float("nan")) for r in rs),
                     "tau": mean(r.get("tau", float("nan")) for r in rs),
                     "rollbacks": mean(r.get("rollbacks", float("nan")) for r in rs),
                     "waste_tok": mean(r.get("waste_tokens", float("nan")) for r in rs),
                     "wait_ms": mean(r.get("lower_wait_ms", float("nan")) for r in rs)})
    for row in rows:
        ref = [r for r in rows if r["lat_ms"] == row["lat_ms"] and r["dataset"] == row["dataset"]]
        by = {r["config"]: r["tok/s"] for r in ref}
        row["x_dflash"] = row["tok/s"] / by["dflash"]
        row["x_ddtree"] = row["tok/s"] / by[f"ddtree-{args.ddtree_budget}"]
    print_table(rows, ["lat_ms", "dataset", "config", "tok/s", "x_dflash", "x_ddtree", "tgt_calls/tok",
                       "mid_calls/tok", "tau", "rollbacks", "waste_tok", "wait_ms"],
                "async vs sync three-stage (virtual clock)")
    print(f"peak GPU memory: {gpu_mem_gb():.1f} GB")
    save_json("exp5_async" + (f"_{args.tag}" if args.tag else ""),
              {"args": vars(args), "costs": {"target": base.target, "mid": base.mid, "draft": base.draft_ms},
               "rows": rows, "records": records})
    del mid, target, draft
    free()


if __name__ == "__main__":
    main()
