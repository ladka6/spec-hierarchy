"""Experiment 14: one-GPU three-level decoding with split verification (hspec.split) against
DFlash and DDTree on the same early-exit target, measured two ways:

  wall   real eager-HF wall clock on one GPU (all methods run the same kernels)
  sim    simulated clock: target pass t(q) from hspec.async3.Costs, drafter draft_ms,
         exit check r_exit * t(q), upper pass r_up * t(q) (ratios from bench_split.py)

Config specs:
  base:dflash, base:ddtree-B                   baselines (baseline drafter)
  split-kK-tT-bB-mM[-pP][@name]                exit after K layers, gate T, drafter tree of B
                                               nodes (0 = chain), M exit rounds per upper pass,
                                               max pending P; @name picks a drafter
                                               (default: the split drafter)

  python scripts/exp14_split.py --target facebook/layerskip-llama3-8B \
      --base-draft .../drafter_ls --split-draft .../drafter_ls_depth16 \
      --configs base:dflash base:ddtree-64 split-k16-t0.7-b32-m4
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.async3 import Costs  # noqa: E402
from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import ddtree_generate, two_stage_generate  # noqa: E402
from hspec.split import SplitConfig, split_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

# measured on LayerSkip Llama3-8B (bench_split.py): lower k layers + exit head, upper layers
SPLIT_RATIOS = {16: (0.53, 0.53), 20: (0.647, 0.409)}


def parse_split(spec):
    m = re.fullmatch(r"split-k(\d+)-t([\d.]+)-b(\d+)-m(\d+)(?:-p(\d+))?", spec.split("@")[0])
    if not m:
        raise ValueError(f"bad split spec {spec}")
    k = int(m.group(1))
    if k not in SPLIT_RATIOS:
        raise ValueError(f"no measured cost ratios for exit layer {k}: {sorted(SPLIT_RATIOS)}")
    r_exit, r_up = SPLIT_RATIOS[k]
    cfg = SplitConfig(exit_layer=k, thr=float(m.group(2)), budget=int(m.group(3)), m=int(m.group(4)),
                      r_exit=r_exit, r_up=r_up)
    if m.group(5):
        cfg.max_pending = int(m.group(5))
    return cfg


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="facebook/layerskip-llama3-8B")
    ap.add_argument("--base-draft", required=True)
    ap.add_argument("--split-draft", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--draft-ms", type=float, default=3.55)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    costs = Costs(draft_ms=args.draft_ms)
    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    stops = stop_ids(target, tok)
    drafts = {"base": load_draft(args.base_draft), "split": load_draft(args.split_draft)}
    prompts = {d: load_prompts(d, args.n) for d in args.datasets}
    warm = encode(tok, prompts[args.datasets[0]][0])
    two_stage_generate(drafts["base"], target, target, target, warm, 32, stops)
    split_generate(drafts["split"], target, warm, 32, stops, SplitConfig())

    records = []
    first = {}                        # outputs of the first config: every method must match them
    for spec in args.configs:
        mism = 0
        for d in args.datasets:
            for i, p in enumerate(prompts[d]):
                ids = encode(tok, p)
                rec = {"config": spec, "dataset": d, "i": i}
                if spec.startswith("base:"):
                    name = spec[5:]
                    if name == "dflash":
                        r = two_stage_generate(drafts["base"], target, target, target, ids, args.max_new, stops)
                    else:
                        r = ddtree_generate(drafts["base"], target, ids, args.max_new, stops,
                                            budget=int(name.split("-")[1]))
                    rec["sim_ms"] = sum(costs.t(q) for q in r.tq) + r.draft_calls * costs.draft_ms
                else:
                    cfg = parse_split(spec)
                    dname = spec.split("@")[1] if "@" in spec else "split"
                    r = split_generate(drafts[dname], target, ids, args.max_new, stops, cfg, costs)
                    s = r.split_stats
                    rec.update({"sim_ms": r.sim_ms, "rounds": s.rounds, "upper": s.upper_passes,
                                "rollbacks": s.rollbacks, "unsure": s.unsure_flushes,
                                "exit_acc": mean(s.exit_accepted) if s.exit_accepted else 0.0,
                                "upper_q": mean(s.upper_q) if s.upper_q else 0.0})
                rec["tokens"] = r.num_output_tokens
                rec["wall_ms"] = r.decode_time * 1000
                rec["tgt_passes"] = r.target_calls
                records.append(rec)
                out = r.generated.tolist()
                if (d, i) in first:
                    mism += out != first[(d, i)]
                else:
                    first[(d, i)] = out
        print(f"[{spec}] done, outputs differing from the first config: {mism}", flush=True)
        save_json(f"exp14_{args.tag}", {"args": vars(args), "records": records})

    by = defaultdict(list)
    for r in records:
        by[r["config"]].append(r)
    ref_cfg = next((c for c in by if c.startswith("base:ddtree-64")), next(iter(by)))

    def tps(rs, key):
        return sum(r["tokens"] for r in rs) / (sum(r[key] for r in rs) / 1000)

    base_sim, base_wall = tps(by[ref_cfg], "sim_ms"), tps(by[ref_cfg], "wall_ms")
    rows = []
    for c, rs in by.items():
        toks = sum(r["tokens"] for r in rs)
        row = {"config": c, "sim_tok/s": tps(rs, "sim_ms"), "x_ref_sim": tps(rs, "sim_ms") / base_sim,
               "wall_tok/s": tps(rs, "wall_ms"), "x_ref_wall": tps(rs, "wall_ms") / base_wall,
               "tok/pass": toks / sum(r["tgt_passes"] for r in rs)}
        if "rounds" in rs[0]:
            row.update({"exit_acc": mean(r["exit_acc"] for r in rs), "upper_q": mean(r["upper_q"] for r in rs),
                        "rb/100tok": 100 * sum(r["rollbacks"] for r in rs) / toks,
                        "unsure/100tok": 100 * sum(r["unsure"] for r in rs) / toks})
        rows.append(row)
    print_table(rows, ["config", "sim_tok/s", "x_ref_sim", "wall_tok/s", "x_ref_wall", "tok/pass", "exit_acc",
                       "upper_q", "rb/100tok", "unsure/100tok"],
                f"one-GPU split verification vs baselines (reference: {ref_cfg})")
    save_json(f"exp14_{args.tag}", {"args": vars(args), "records": records, "rows": rows})


if __name__ == "__main__":
    main()
