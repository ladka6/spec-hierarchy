"""Experiment 9: real two-GPU run vs the simulated clock.

GPU 0: drafter + middle model (+ a local target copy for the drafter's embeddings / LM
head and for the baselines). GPU 1: the target, in its own process (hspec.real2gpu).
Network delay is emulated with real sleeps.

1. Real runs: each config at each delay, wall-clock decode time.
2. Baselines on GPU 0 (DFlash, DDTree): wall time + L per target call.
3. Cost tables fitted from the real runs themselves (median target compute time and
   median lower-stage step time, by query length).
4. The same prompts and configs replayed on the simulated clock with those costs.
   If simulated and real times agree, the simulator's clock logic (overlap, waits,
   rollbacks, in-flight checks) is right, and results computed with vLLM costs carry over.

All runs use HF eager kernels, so absolute speeds are not production numbers.

  python scripts/exp9_real2gpu.py --n 10
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.async3 import Costs  # noqa: E402
from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.hier import hier_generate  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import ddtree_generate, two_stage_generate  # noqa: E402
from hspec.real2gpu import ProcTarget  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp8_algos import parse  # noqa: E402

EDGES = [1, 9, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 10**6]


def fit(pairs):
    """[(q, ms)] -> {q_median: ms_median} per bin."""
    bins = defaultdict(list)
    for q, ms in pairs:
        for lo, hi in zip(EDGES, EDGES[1:]):
            if lo <= q < hi:
                bins[lo].append((q, ms))
                break
    tab = {}
    for vals in bins.values():
        qs = sorted(v[0] for v in vals)
        ms = sorted(v[1] for v in vals)
        tab[qs[len(qs) // 2]] = ms[len(ms) // 2]
    return tab


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mid", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--latencies-ms", nargs="+", type=float, default=[0, 20, 100])
    ap.add_argument("--configs", nargs="+",
                    default=["sync-16+mt32", "async-8+mt32", "async-c32+mt32", "async-c32+mt32+h2+f0.7+at0"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--min-window", type=int, default=4)
    ap.add_argument("--fork-thr", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--server", default=None, help="model spec for the target server (default: --target)")
    ap.add_argument("--server-device", type=int, default=1, help="-1 = CPU (tests)")
    ap.add_argument("--local-device", default="cuda:0")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    mid = load_mid(args.mid)
    stops = stop_ids(target, tok)
    prompts = [(d, i, encode(tok, p)) for d in args.datasets for i, p in enumerate(load_prompts(d, args.n))]
    two_stage_generate(draft, target, target, target, prompts[0][2], 32, stops)
    be = ProcTarget(args.server or args.target, device=None if args.server_device < 0 else args.server_device,
                    latency_ms=0.0, local_device=args.local_device)

    records, steps = [], []
    # 1. real runs
    for spec in args.configs:
        cfg = parse(spec, args)
        for lat in args.latencies_ms:
            be.set_latency(lat)
            for d, i, ids in prompts:
                r = hier_generate(draft, target, mid, ids, args.max_new, stops, Costs(latency_ms=lat), cfg,
                                  backend=be, use_target_feats=False)
                s = r.async_stats
                steps += list(zip(s["step_q"], s["step_ms"]))
                records.append({"kind": "real", "config": spec, "lat": lat, "dataset": d, "i": i,
                                "tokens": r.num_output_tokens, "ms": r.decode_time * 1000,
                                "rollbacks": s["rollbacks"], "caught": s["caught"],
                                "out": r.generated.tolist()})
            print(f"[real {spec} | {lat} ms] done", flush=True)
            save_json("exp9_real2gpu", {"args": vars(args), "records": records})
    be.close()

    # 2. baselines on GPU 0 (synchronous: wall time + L per decode-time target call)
    for name in ("dflash", "ddtree-64"):
        for d, i, ids in prompts:
            if name == "dflash":
                r = two_stage_generate(draft, target, target, target, ids, args.max_new, stops)
            else:
                r = ddtree_generate(draft, target, ids, args.max_new, stops, budget=64)
            for lat in args.latencies_ms:
                records.append({"kind": "real", "config": name, "lat": lat, "dataset": d, "i": i,
                                "tokens": r.num_output_tokens, "ms": r.decode_time * 1000 + lat * len(r.tq)})
        print(f"[baseline {name}] done", flush=True)

    # 3. costs fitted from the real runs
    t_tab = fit(be.target_ms)
    s_tab = fit(steps)
    print(f"fitted target ms by q: {t_tab}\nfitted lower step ms by q: {s_tab}", flush=True)

    # 4. same prompts / configs on the simulated clock
    for spec in args.configs:
        cfg = parse(spec, args)
        cfg.draft_batch_beta = 0.0
        for lat in args.latencies_ms:
            costs = Costs(target=t_tab, mid=s_tab, draft_ms=0.0, latency_ms=lat)
            for d, i, ids in prompts:
                r = hier_generate(draft, target, mid, ids, args.max_new, stops, costs, cfg, use_target_feats=False)
                records.append({"kind": "sim", "config": spec, "lat": lat, "dataset": d, "i": i,
                                "tokens": r.num_output_tokens, "ms": r.decode_time * 1000,
                                "rollbacks": r.async_stats["rollbacks"], "out": r.generated.tolist()})
            print(f"[sim {spec} | {lat} ms] done", flush=True)
    save_json("exp9_real2gpu", {"args": vars(args), "target_tab": t_tab, "step_tab": s_tab, "records": records})

    rows = []
    real = {(r["config"], r["lat"], r["dataset"], r["i"]): r for r in records if r["kind"] == "real"}
    for spec in args.configs + ["dflash", "ddtree-64"]:
        for lat in args.latencies_ms:
            rr = [r for r in records if r["kind"] == "real" and r["config"] == spec and r["lat"] == lat]
            ss = [r for r in records if r["kind"] == "sim" and r["config"] == spec and r["lat"] == lat]
            row = {"config": spec, "lat_ms": lat,
                   "real_tok/s": sum(r["tokens"] for r in rr) / (sum(r["ms"] for r in rr) / 1000)}
            if ss:
                row["sim_tok/s"] = sum(r["tokens"] for r in ss) / (sum(r["ms"] for r in ss) / 1000)
                row["sim_err_%"] = 100 * (row["sim_tok/s"] / row["real_tok/s"] - 1)
                row["same_output"] = mean(float(real[(r["config"], r["lat"], r["dataset"], r["i"])]["out"] == r["out"])
                                          for r in ss)
                row["rollbacks_real"] = mean(r["rollbacks"] for r in rr)
                row["rollbacks_sim"] = mean(r["rollbacks"] for r in ss)
            rows.append(row)
    for row in rows:
        base = [r["real_tok/s"] for r in rows if r["lat_ms"] == row["lat_ms"] and r["config"] == "ddtree-64"]
        row["x_ddtree_real"] = row["real_tok/s"] / base[0]
    print_table(rows, ["config", "lat_ms", "real_tok/s", "sim_tok/s", "sim_err_%", "x_ddtree_real",
                       "rollbacks_real", "rollbacks_sim", "same_output"],
                "real two-GPU run vs simulated clock (HF eager kernels)")
    save_json("exp9_real2gpu", {"args": vars(args), "target_tab": t_tab, "step_tab": s_tab,
                                "rows": rows, "records": records})


if __name__ == "__main__":
    main()
