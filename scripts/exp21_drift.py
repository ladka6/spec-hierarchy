"""Experiment 21: how much does a DFlash drafter lose when its target is fine-tuned?

For one target variant (e.g. a LoRA fine-tune of Qwen3-8B, scripts/lora_target.py) and the
drafter trained for the BASE target:

  1. speed and acceptance of the official DFlash / DDTree loops (third_party/ddtree_official)
     vs plain decoding, on the usual evaluation prompts
  2. how far the variant moved from the base target, on the variant's own greedy outputs:
     KL(variant || base) per token, top-1 agreement, and the cosine similarity of the hidden
     states the drafter reads (its 5 target layers)

Run once with --variant = the base model for the reference numbers.

  python scripts/exp21_drift.py --variant /scratch-shared/$USER/hspec_drift/qwen3-8b-math1000 --name math1000
"""

from __future__ import annotations

import argparse
import gc
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
from hspec.utils import mean, print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--budgets", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--no-drift", action="store_true", help="speed / acceptance only")
    ap.add_argument("--drafter-name", default="", help="label of the drafter (for the matrix)")
    args = ap.parse_args()

    maybe_enable_cpp_compact(True)
    tok = load_tokenizer(args.variant)
    target = load_target(args.variant)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, mask = draft.block_size, draft.mask_token_id
    methods = ["baseline", "dflash"] + [f"ddtree_tb{b}" for b in args.budgets]

    def run(m, ids):
        if m == "baseline":
            return off_dflash(draft, target, ids, mask, args.max_new, 1, stops)
        if m == "dflash":
            return off_dflash(draft, target, ids, mask, args.max_new, bs, stops)
        return off_ddtree(draft, target, ids, mask, args.max_new, bs, stops, tree_budget=int(m[9:]))

    warm = encode(tok, load_prompts(args.datasets[0], 1)[0])
    for m in methods:
        run(m, warm)

    rec = defaultdict(list)
    trajs = []
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            for m in methods:
                r = run(m, ids)
                rec[m].append({"dataset": d, "tpot": r.time_per_output_token, "n": r.num_output_tokens,
                               "tau": mean(r.acceptance_lengths) if getattr(r, "acceptance_lengths", None) else 1.0})
                if m == "baseline":
                    trajs.append((d, ids.shape[1], r.output_ids.to(ids.device)))
        print(f"[{args.name} | {d}] speed done", flush=True)

    # drift of the variant from the base, on the variant's own greedy outputs
    layer_ids = list(draft.target_layer_ids)
    var_out = []
    for d, n0, x in ([] if args.no_drift else trajs):
        o = target(x, output_hidden_states=True)
        var_out.append((torch.log_softmax(o.logits[0, n0 - 1 : -1].float(), -1).half().cpu(),
                        [o.hidden_states[i + 1][0, n0 - 1 : -1].float().cpu() for i in layer_ids]))
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()
    drift = defaultdict(list)
    base = None if args.no_drift else load_target(args.base)
    for (d, n0, x), (vlp, vh) in zip(trajs, var_out):
        o = base(x, output_hidden_states=True)
        blp = torch.log_softmax(o.logits[0, n0 - 1 : -1].float(), -1).cpu()
        vlp = vlp.float()
        drift["kl"].append(float((vlp.exp() * (vlp - blp)).sum(-1).mean()))
        drift["top1"].append(float((vlp.argmax(-1) == blp.argmax(-1)).float().mean()))
        for j, i in enumerate(layer_ids):
            bh = o.hidden_states[i + 1][0, n0 - 1 : -1].float().cpu()
            drift[f"cos_L{i}"].append(float(torch.nn.functional.cosine_similarity(vh[j], bh, dim=-1).mean()))

    base_t = sum(b["tpot"] * b["n"] for b in rec["baseline"]) / sum(b["n"] for b in rec["baseline"])
    rows = []
    for m in methods:
        rs = rec[m]
        t = sum(r["tpot"] * r["n"] for r in rs) / sum(r["n"] for r in rs)
        row = {"variant": args.name, "drafter": args.drafter_name or args.draft, "method": m,
               "tau": mean(r["tau"] for r in rs), "speedup": base_t / t}
        for dd in args.datasets:
            row[f"tau_{dd}"] = mean(r["tau"] for r in rs if r["dataset"] == dd)
        rows.append(row)
    print_table(rows, ["variant", "method", "tau", "speedup"] + [f"tau_{dd}" for dd in args.datasets],
                f"base-target drafter on target variant '{args.name}' (official loops)")
    drow = {"variant": args.name, **{k: mean(v) for k, v in drift.items()}}
    if not args.no_drift:
        print_table([drow], list(drow), "variant vs base target, on the variant's greedy outputs")
    save_json(f"exp21_{args.name}", {"args": vars(args), "rows": rows, "drift": drow, "records": rec})


if __name__ == "__main__":
    main()
