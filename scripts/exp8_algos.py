"""Experiment 8: algorithm comparison on a simulated two-device clock.

Config specs (combine with "+"):
  base:dflash, base:ddtree-B   baselines, virtual time from their exact call sequence
  pearl-K / pearlsync-K        two-stage DFlash, drafter tokens (K per round) sent unverified,
                               drafter only sees target features of confirmed positions
  sync-P / async-P             three-stage, check every P unsubmitted tokens
  sync-cP / async-cP           confidence-timed checks (tau, min window, max window P)
  +mtB                         mid verifies a DDTree of B drafter nodes
  +hK                          hedging with up to K live alternative branches
  +fX                          fork threshold (mid top-1 probability below X)
  +atB                         mid tree budget on alternative branches (0 = chain)
  cPtXmM                       conf check with tau X and min window M (e.g. async-c32t0.3m2)

Blocking configs behave identically at every latency, so they run once and their time is
shifted by L per target check.

  python scripts/exp8_algos.py --configs sync-16+mt32 async-8+mt32+h2 --tag gpu1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.async3 import Costs  # noqa: E402
from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.hier import HierConfig, hier_generate  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import ddtree_generate, two_stage_generate  # noqa: E402
from hspec.utils import save_json  # noqa: E402


def parse(spec: str, args) -> HierConfig:
    """e.g. async-c32t0.3m2+mt32+h4+f0.7+at0 (see module docstring)."""
    parts = spec.split("+")
    mode, arg = parts[0].split("-")
    cfg = HierConfig(tau=args.tau, min_window=args.min_window, fork_thr=args.fork_thr,
                     draft_batch_beta=args.beta)
    if mode.startswith("pearl"):
        cfg.pearl, cfg.pearl_len, cfg.window = True, int(arg), int(arg)
        cfg.blocking = mode == "pearlsync"
    else:
        cfg.blocking = mode == "sync"
        m = re.fullmatch(r"(c?)(\d+)(?:t([\d.]+))?(?:m(\d+))?", arg)
        if not m:
            raise ValueError(f"bad window spec {arg} in {spec}")
        cfg.window = int(m.group(2))
        if m.group(1):
            cfg.check_rule = "conf"
        if m.group(3):
            cfg.tau = float(m.group(3))
        if m.group(4):
            cfg.min_window = int(m.group(4))
    for p in parts[1:]:
        if p.startswith("mt"):
            cfg.mid_tree = int(p[2:])
        elif p.startswith("at"):
            cfg.alt_tree = int(p[2:])
        elif p.startswith("h"):
            cfg.max_branches = int(p[1:])
        elif p.startswith("f"):
            cfg.fork_thr = float(p[1:])
        else:
            raise ValueError(f"unknown option {p} in {spec}")
    return cfg


def baseline_ms(r, costs: Costs) -> float:
    return sum(costs.t(q) + costs.latency_ms for q in r.tq) + r.draft_calls * costs.draft_ms


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mid", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--latencies-ms", nargs="+", type=float, default=[0, 10, 50, 100])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--min-window", type=int, default=4)
    ap.add_argument("--fork-thr", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--draft-ms", type=float, default=3.55)
    ap.add_argument("--suffix", default="", help="appended to config names in the records (e.g. @ao4)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    base = Costs(draft_ms=args.draft_ms)
    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    need_mid = any(not (c.startswith("base:") or c.startswith("pearl")) for c in args.configs)
    mid = load_mid(args.mid) if need_mid else None
    stops = stop_ids(target, tok)
    prompts = {d: load_prompts(d, args.n) for d in args.datasets}
    two_stage_generate(draft, target, target, target, encode(tok, prompts[args.datasets[0]][0]), 32, stops)

    records = []
    out_name = "exp8_" + (args.tag or "run")
    for spec in args.configs:
        if spec.startswith("base:"):
            name = spec[5:]
            for d in args.datasets:
                for i, p in enumerate(prompts[d]):
                    ids = encode(tok, p)
                    if name == "dflash":
                        r = two_stage_generate(draft, target, target, target, ids, args.max_new, stops)
                    else:
                        r = ddtree_generate(draft, target, ids, args.max_new, stops, budget=int(name.split("-")[1]))
                    for lat in args.latencies_ms:
                        c = Costs(base.target, base.mid, base.draft_ms, lat)
                        records.append({"dataset": d, "i": i, "lat": lat, "config": name,
                                        "tokens": r.num_output_tokens, "ms": baseline_ms(r, c),
                                        "tgt_calls_per_tok": r.target_calls / max(r.num_output_tokens, 1)})
            print(f"[{name}] done", flush=True)
            save_json(out_name, {"args": vars(args), "records": records})
            continue
        cfg = parse(spec, args)
        lats = [0.0] if cfg.blocking else args.latencies_ms
        for lat in lats:
            costs = Costs(base.target, base.mid, base.draft_ms, lat)
            for d in args.datasets:
                for i, p in enumerate(prompts[d]):
                    r = hier_generate(draft, target, mid, encode(tok, p), args.max_new, stops, costs, cfg)
                    s = r.summary()
                    rec = {"dataset": d, "i": i, "config": spec + args.suffix, "tokens": r.num_output_tokens,
                           "tgt_calls_per_tok": s["target_calls_per_token"],
                           "mid_calls_per_tok": s["mid_calls_per_token"], "tau": s["mean_round_len"],
                           "mean_q": s["mean_target_q"], **r.async_stats}
                    if cfg.blocking:   # identical behaviour at any latency: shift by L per check
                        for L in args.latencies_ms:
                            records.append({**rec, "lat": L, "ms": r.decode_time * 1000 + L * len(r.tq)})
                    else:
                        records.append({**rec, "lat": lat, "ms": r.decode_time * 1000})
            print(f"[{spec} | {lat} ms] done", flush=True)
            save_json(out_name, {"args": vars(args), "records": records})


if __name__ == "__main__":
    main()
