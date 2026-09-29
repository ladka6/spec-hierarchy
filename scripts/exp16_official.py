"""Experiment 16: the official DDTree / DFlash code (third_party/ddtree_official, unchanged
decoding loops) on any target + drafter, next to our DDTree reimplementation.

For base models without a chat template (LayerSkip Llama3-8B) the prompt is the plain
completion format of hspec.data.format_prompt; the official benchmark only differs there.
Speed as in the official benchmark: time per output token of each method vs the plain
autoregressive loop (their dflash_generate with block size 1), same prompts, same GPU.

  python scripts/exp16_official.py --target facebook/layerskip-llama3-8B --draft .../drafter_ls
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

from ddtree_official.ddtree import ddtree_generate as off_ddtree, maybe_enable_cpp_compact  # noqa: E402
from ddtree_official.dflash import dflash_generate as off_dflash  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import ddtree_generate as our_ddtree  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="facebook/layerskip-llama3-8B")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--ours", nargs="*", type=int, default=[64], help="budgets for our DDTree implementation")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    maybe_enable_cpp_compact(True)
    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, mask = draft.block_size, draft.mask_token_id

    def run(method, ids):
        """Returns (output ids list, time per output token s, tokens per round)."""
        if method == "baseline":
            r = off_dflash(draft, target, ids, mask, args.max_new, 1, stops)
        elif method == "dflash":
            r = off_dflash(draft, target, ids, mask, args.max_new, bs, stops)
        elif method.startswith("ddtree_tb"):
            r = off_ddtree(draft, target, ids, mask, args.max_new, bs, stops, tree_budget=int(method[9:]))
        else:                                               # our DDTree, "ours_tbB"
            g = our_ddtree(draft, target, ids, args.max_new, stops, budget=int(method[7:]))
            out = g.output_ids[0].tolist()
            return out, g.decode_time / max(g.num_output_tokens, 1), mean(g.rounds)
        acc = getattr(r, "acceptance_lengths", None) or [1]
        return r.output_ids[0].tolist(), r.time_per_output_token, mean(acc)

    methods = ["baseline", "dflash"] + [f"ddtree_tb{b}" for b in args.budgets] + [f"ours_tb{b}" for b in args.ours]
    warm = encode(tok, load_prompts(args.datasets[0], 1)[0])
    for m in methods:
        run(m, warm)

    rec = defaultdict(list)
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            outs = {}
            for m in methods:
                out, tpot, tau = run(m, ids)
                n_out = len(out) - ids.shape[1]
                outs[m] = out
                rec[m].append({"dataset": d, "tpot": tpot, "tau": tau, "n": n_out,
                               "same": out[: len(outs["baseline"])] == outs["baseline"][: len(out)]})
        print(f"[{d}] done", flush=True)
        save_json(f"exp16_{args.tag}", {"args": vars(args), "records": rec})

    base = rec["baseline"]
    rows = []
    for m in methods:
        rs = rec[m]
        tot_t = sum(r["tpot"] * r["n"] for r in rs)
        base_t = sum(b["tpot"] * b["n"] for b in base)
        rows.append({"method": m, "tau": mean(r["tau"] for r in rs),
                     "speedup_total": (base_t / sum(b["n"] for b in base)) / (tot_t / sum(r["n"] for r in rs)),
                     "speedup_mean": mean(b["tpot"] / r["tpot"] for b, r in zip(base, rs) if r["tpot"] > 0),
                     "tok/s": sum(r["n"] for r in rs) / tot_t if tot_t else float("nan"),
                     "same_as_ar": sum(r["same"] for r in rs), "prompts": len(rs)})
    print_table(rows, ["method", "tau", "speedup_total", "speedup_mean", "tok/s", "same_as_ar", "prompts"],
                "official DFlash / DDTree vs our DDTree (speedup over plain decoding, same loop)")
    save_json(f"exp16_{args.tag}", {"args": vars(args), "records": rec, "rows": rows})


if __name__ == "__main__":
    main()
