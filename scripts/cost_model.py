"""Cost model: predicted decoding speed from call counts x per-call latency.

exp3 records how many target, middle and drafter calls each configuration needs per
generated token. Those counts do not depend on how fast the kernels are, so they can be
combined with per-call latencies measured elsewhere (bench_forward.py, eager or compiled)
and with a simulated network delay L on every target call:

    time/token = n_T (c_T(q) + L) + n_M c_M + n_D c_D

q is the mean number of query tokens per decode-time target forward (recorded by exp3 as
mean_target_q: block size for DFlash, tree size + 1 for DDTree, pending (+ branches) + 1 for
three-stage), interpolated from the per-q latency tables. The drafter cost comes from
bench_draft.json (CUDA graphs) when present.
At L=0 the table also shows the measured exp3 speed, to check the model against reality.

  python scripts/cost_model.py --exp3 results/exp3_pipeline.json --bench results/bench_forward.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.utils import print_table, save_json  # noqa: E402


def interp(table: dict[int, float], q: float) -> float:
    ks = sorted(table)
    if q <= ks[0]:
        return table[ks[0]]
    if q >= ks[-1]:
        return table[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= q <= b:
            w = (q - a) / (b - a)
            return table[a] * (1 - w) + table[b] * w
    return table[ks[-1]]


def row_table(row) -> dict[int, float]:
    return {int(k[2:]): v for k, v in row.items() if k.startswith("q=")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp3", default="results/exp3_pipeline.json")
    ap.add_argument("--bench", default="results/bench_forward.json")
    ap.add_argument("--mid", default=None, help="bench row to use as the middle model (default: first non-bf16)")
    ap.add_argument("--latencies-ms", nargs="+", type=float, default=[0, 10, 20, 50, 100])
    ap.add_argument("--mid-ratios", nargs="*", type=float, default=[0.25, 0.5],
                    help="also evaluate hypothetical middle models costing ratio x target")
    ap.add_argument("--draft-ms", type=float, default=None, help="override drafter cost")
    ap.add_argument("--draft-bench", default="results/bench_draft.json", help="bench_draft.py output, if present")
    ap.add_argument("--draft-ctx", type=int, default=8, help="bench_draft ctx row to use")
    ap.add_argument("--vllm", default="results/bench_vllm.json", help="bench_vllm.py output, if present")
    ap.add_argument("--vllm-target", default="Qwen/Qwen3-8B")
    ap.add_argument("--manual", nargs="*", default=[],
                    help="extra scenarios label:target_ms:mid_ms (flat in q)")
    args = ap.parse_args()

    exp3 = json.loads(Path(args.exp3).read_text())
    bench = json.loads(Path(args.bench).read_text())
    c_d = args.draft_ms if args.draft_ms is not None else bench.get("draft_ms", 5.0)
    c_d_bs = {}   # drafter cost by block size (CUDA graphs)
    dpath = Path(args.draft_bench)
    if args.draft_ms is None and dpath.exists():
        for r in json.loads(dpath.read_text())["rows"]:
            if r["ctx"] == args.draft_ctx and r["graph_ms"] == r["graph_ms"]:
                c_d_bs[r["bs"]] = r["graph_ms"]
        c_d = c_d_bs.get(16, c_d)

    def draft_cost(cfg):
        bs = int(cfg.split("-b")[1]) if cfg.startswith("dflash-b") else 16
        return c_d_bs.get(bs, c_d)

    # per-config mean call counts (token-weighted), from lat=0 records
    agg = defaultdict(lambda: defaultdict(float))
    for r in exp3["records"]:
        if r.get("lat", 0) != 0:
            continue
        k = (r["dataset"], r["config"])
        n = r["num_output_tokens"]
        a = agg[k]
        a["tok"] += n
        a["time"] += r["decode_time"]
        a["T"] += r["target_calls_per_token"] * n
        a["M"] += r["mid_calls_per_token"] * n
        a["D"] += r["draft_calls_per_token"] * n
        a["pend"] += r.get("mean_pending_per_check", 16) * n
        if "mean_target_q" in r:
            w = r["target_calls_per_token"] * n
            a["qw"] += r["mean_target_q"] * w
            a["qn"] += w
    counts = {}
    for (d, cfg), v in agg.items():
        c = {k: v[k] / v["tok"] for k in ("T", "M", "D", "pend")} | {"meas": v["tok"] / v["time"]}
        if v["qn"] > 0:
            c["q"] = v["qw"] / v["qn"]
        else:   # older exp3 files without mean_target_q
            c["q"] = 1 if cfg == "ar" else (17 if cfg == "dflash" else c["pend"] + 1)
        counts[(d, cfg)] = c

    # cost scenarios: (label, target table, mid cost at q=17)
    rows_b = bench["rows"]
    scen = []
    for mode in ("eager", "compiled"):
        tgt = next((r for r in rows_b if r["model"].startswith("bf16") and r["mode"] == mode), None)
        if tgt is None:
            continue
        t_tab = row_table(tgt)
        mids = [r for r in rows_b if not r["model"].startswith("bf16") and r["mode"] == mode]
        if args.mid:
            mids = [r for r in mids if r["model"] == args.mid]
        for m in mids:
            scen.append((f"{mode}:{m['model']}", t_tab, interp(row_table(m), 17)))
        for ratio in args.mid_ratios:
            scen.append((f"{mode}:mid={ratio}xT", t_tab, ratio * interp(t_tab, 17)))

    # vLLM (CUDA graphs): per-q verification cost if measured, else flat in q
    vpath = Path(args.vllm)
    if vpath.exists():
        vtab = {}
        for r in json.loads(vpath.read_text())["rows"]:
            vtab[r["model"]] = ({int(k): v for k, v in r["q_ms"].items()} if r.get("q_ms")
                                else {1: r["decode_ms"], 129: r["decode_ms"]})
        if args.vllm_target in vtab:
            t_tab = vtab[args.vllm_target]
            for name, m_tab in vtab.items():
                if name != args.vllm_target:
                    scen.append((f"vllm:{name}", t_tab, interp(m_tab, 17)))
            for ratio in args.mid_ratios:
                scen.append((f"vllm:mid={ratio}xT", t_tab, ratio * interp(t_tab, 17)))
            print("vLLM target cost by q (ms): " + ", ".join(f"{q}:{v:.2f}" for q, v in sorted(t_tab.items())))
    for spec in args.manual:
        label, t_ms, m_ms = spec.split(":")
        scen.append((f"manual:{label}", {1: float(t_ms), 129: float(t_ms)}, float(m_ms)))

    out = []
    for label, t_tab, c_m in scen:
        for lat in args.latencies_ms:
            per = {}
            for (d, cfg), c in counts.items():
                t = c["T"] * (interp(t_tab, c["q"]) + lat) + c["M"] * c_m + c["D"] * draft_cost(cfg)
                per[(d, cfg)] = 1000.0 / t
            for (d, cfg), tps in per.items():
                out.append({"costs": label, "lat_ms": lat, "dataset": d, "config": cfg, "q": counts[(d, cfg)]["q"],
                            "pred_tok/s": tps, "x_dflash": tps / per[(d, "dflash")],
                            "meas_tok/s": counts[(d, cfg)]["meas"] if lat == 0 and label.startswith("eager") else float("nan")})

    for label, *_ in scen:
        rows = sorted((r for r in out if r["costs"] == label),
                      key=lambda r: (r["dataset"], r["lat_ms"], r["config"]))
        print_table(rows, ["dataset", "lat_ms", "config", "q", "pred_tok/s", "x_dflash", "meas_tok/s"],
                    f"predicted speed, costs = {label}, drafter {c_d:.2f} ms (bs16)")
    save_json("cost_model", {"args": vars(args), "counts": {f"{k[0]}|{k[1]}": v for k, v in counts.items()},
                             "rows": out})


if __name__ == "__main__":
    main()
