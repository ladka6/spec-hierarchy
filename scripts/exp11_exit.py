"""Experiment 11: can the target's own first k layers be the middle verifier?

On target-greedy trajectories of the evaluation sets, DFlash blocks are drafted every
`stride` positions (fresh target features, i.e. the drafter's best case) and checked by each
candidate middle verifier, teacher-forced on the trajectory (see hspec.exit.block_outcome):

  delivered   correct tokens the verifier hands to the target per block (draft + correction)
  eff         delivered / what a verifier identical to the target would deliver
  leak        share of blocks where it hands the target a wrong token (causes a rollback)
  eps         per-token disagreement with the target (teacher forced)
  conf90      share of positions with top-1 probability >= 0.9, and the disagreement rate there
  cost        forward cost relative to the full target (layers run / 36; bnb4 measured separately)

Verifiers: raw / lin / layer exit heads at each k (hspec.exit), the bnb4 4-bit copy (the
current middle model) and the target itself.

  python scripts/exp11_exit.py --heads /scratch-shared/$USER/hspec_lagft/exit_heads.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.exit import ExitHead, block_outcome, block_outcome_gated, head_name, load_heads  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import free, load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mid", default="bnb4:Qwen/Qwen3-8B", help="reference middle model ('' to skip)")
    ap.add_argument("--heads", default="", help="trained heads file (train_exit.py)")
    ap.add_argument("--layers", nargs="+", type=int, default=[12, 18, 24, 30])
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--gates", nargs="*", type=float, default=[],
                    help="also score each verifier gated at these top-1 probability thresholds")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    L = target.config.num_hidden_layers
    bs = draft.block_size

    heads = {head_name("raw", k): ExitHead(target, "raw", k) for k in args.layers}
    if args.heads:
        heads.update(load_heads(target, args.heads))
    cost = {n: (h.k + (1 if h.kind == "layer" else 0)) / L for n, h in heads.items()}

    # trajectories, drafted blocks, and each verifier's teacher-forced predictions
    trajs = []
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            seq = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids
            out = target(seq, output_hidden_states=True)
            feats = extract_context_feature(out.hidden_states, draft.target_layer_ids)
            preds, p1 = {"target": out.logits[0].argmax(-1)}, {"target": None}
            for n, h in heads.items():
                lg = h.logits(out.hidden_states[h.k])[0].float()
                pr = torch.softmax(lg, -1)
                preds[n], p1[n] = lg.argmax(-1), pr.max(-1).values
                del lg, pr
            x = seq[0]
            blocks = []
            for s in range(n0, len(x) - bs - 1, args.stride):
                dl = draft_logits_lagged(draft, target, feats, x, s, 0, bs)
                blocks.append((s, dl.argmax(-1)))
            trajs.append({"d": d, "n0": n0, "x": x, "preds": preds, "p1": p1, "blocks": blocks})
            del out, feats
    print(f"{len(trajs)} trajectories", flush=True)
    if args.mid:
        mid = load_mid(args.mid)
        for tr in trajs:
            lg = mid(tr["x"][None]).logits[0].float()
            tr["preds"]["bnb4"], tr["p1"]["bnb4"] = lg.argmax(-1), torch.softmax(lg, -1).max(-1).values
            del lg
        del mid
        free()

    names = ["target"] + (["bnb4"] if args.mid else []) + sorted(heads, key=lambda n: (heads[n].k, n))
    acc = defaultdict(list)
    for tr in trajs:
        x, n0, d = tr["x"], tr["n0"], tr["d"]
        truth_next = x[n0:]                                   # tokens at positions n0..T-1
        for n in names:
            pred = tr["preds"][n][n0 - 1 : -1]                # predictions for positions n0..T-1
            dis = (pred != truth_next).float()
            acc[(n, d, "eps")] += dis.tolist()
            if tr["p1"][n] is not None:
                c = tr["p1"][n][n0 - 1 : -1] >= 0.9
                acc[(n, d, "conf90")] += c.float().tolist()
                acc[(n, d, "dis90")] += dis[c].tolist()
            e_all = tr["preds"][n]
            for s, dt in tr["blocks"]:
                t = x[s + 1 : s + bs].tolist()
                e = e_all[s : s + bs - 1].tolist()             # prediction at s+j is for s+j+1
                tb = int(x[s + bs]) if s + bs < len(x) else None
                eb = int(e_all[s + bs - 1]) if tb is not None else None
                dv, lk, pf = block_outcome(dt.tolist(), t, e, tb, eb)
                acc[(n, d, "delivered")].append(dv)
                acc[(n, d, "perfect")].append(pf)
                acc[(n, d, "leak")].append(lk)
                if tr["p1"][n] is None:
                    continue
                pp = tr["p1"][n][s : s + bs - 1].tolist()
                pb = float(tr["p1"][n][s + bs - 1]) if tb is not None else None
                for thr in args.gates:
                    g = f"{n}@{thr}"
                    dv, lk, pf = block_outcome_gated(dt.tolist(), t, e, pp, thr, tb, eb, pb)
                    acc[(g, d, "delivered")].append(dv)
                    acc[(g, d, "perfect")].append(pf)
                    acc[(g, d, "leak")].append(lk)

    gated = [f"{n}@{thr}" for n in names if n != "target" for thr in args.gates]
    rows = []
    for n in names + gated:
        for d in args.datasets + ["ALL"]:
            ds = args.datasets if d == "ALL" else [d]
            g = lambda key: sum((acc[(n, dd, key)] for dd in ds), [])  # noqa: E731
            dv, pf = g("delivered"), g("perfect")
            base_n = n.split("@")[0]
            rows.append({"verifier": n, "dataset": d, "cost": cost.get(base_n, 1.0 if n == "target" else float("nan")),
                         "delivered": mean(dv), "eff": sum(dv) / max(sum(pf), 1), "leak": mean(g("leak")),
                         "eps": mean(g("eps")), "conf90": mean(g("conf90")) if g("conf90") else float("nan"),
                         "dis@90": mean(g("dis90")) if g("dis90") else float("nan"), "blocks": len(dv)})
    print_table([r for r in rows if r["dataset"] == "ALL"],
                ["verifier", "cost", "delivered", "eff", "leak", "eps", "conf90", "dis@90", "blocks"],
                "middle verifier candidates on DFlash blocks (ALL datasets)")
    print_table([r for r in rows if r["dataset"] != "ALL"],
                ["verifier", "dataset", "delivered", "eff", "leak", "eps"], "per dataset")
    save_json("exp11_exit", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
