"""Experiment 37: where do cheap correctors disagree with the target? (margin-aware context, offline)

The multi-round refresh loop is limited by the corrector's token agreement with the target
(0.80 / 0.85 / 0.97 for Qwen3-0.6B / 1.7B / a 4-bit copy -> 13.5 / 18.6 / 33.8 accepted per verify).
Theory (acceptance flips only at low target margin) suggests cheap correctors fail mainly where the
target itself is undecided. On target-greedy trajectories, per generated position:

  target margin   p_top1 - p_top2 of the target's next-token distribution
  agreement       each corrector's argmax == the target's

Tables:
  1. disagreement rate and share of all disagreements per target-margin bin, per corrector
  2. escalation frontier: use the cheap corrector where ITS OWN margin >= m (known at inference) and
     the expensive one (4-bit copy) elsewhere; agreement achieved vs share of positions escalated

  python scripts/exp37_margin.py --cheap Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B --copy bnb4:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

BINS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.01]


def margin(logits):
    p = torch.softmax(logits.float(), -1)
    top = p.topk(2, -1).values
    return top[:, 0] - top[:, 1], logits.argmax(-1)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16", help="only to generate the trajectories")
    ap.add_argument("--cheap", nargs="+", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"])
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default="exp37_margin")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    models = {m.split("/")[-1]: load_target(m) for m in args.cheap}
    models["copy4"] = load_mid(args.copy)
    names = list(models)
    tm, ok, mg = [], {k: [] for k in names}, {k: [] for k in names}
    for d in args.datasets:
        for pr in load_prompts(d, args.n):
            ids = encode(tok, pr)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            r = slice(n0 - 1, len(x) - 1)
            m_t, a_t = margin(target(x[None]).logits[0, r])
            tm.append(m_t)
            for k, mod in models.items():
                m_k, a_k = margin(mod(x[None]).logits[0, r])
                ok[k].append(a_k == a_t)
                mg[k].append(m_k)
        print(f"[{d}] done", flush=True)
    tm = torch.cat(tm)
    ok = {k: torch.cat(v) for k, v in ok.items()}
    mg = {k: torch.cat(v) for k, v in mg.items()}
    N = len(tm)

    rows = []
    for lo, hi in zip(BINS, BINS[1:]):
        sel = (tm >= lo) & (tm < hi)
        row = {"target_margin": f"{lo:.2f}-{min(hi, 1):.2f}", "share_pos": float(sel.float().mean())}
        for k in names:
            dis = ~ok[k]
            row[f"dis_{k}"] = float(dis[sel].float().mean()) if sel.any() else float("nan")
            row[f"share_dis_{k}"] = float((dis & sel).sum() / max(int(dis.sum()), 1))
        rows.append(row)
    cols = ["target_margin", "share_pos"] + [f"dis_{k}" for k in names] + [f"share_dis_{k}" for k in names]
    print_table(rows, cols, f"disagreement with the target by the target's margin ({N} positions); "
                            "overall agreement: " + ", ".join(f"{k} {float(ok[k].float().mean()):.3f}" for k in names))

    frows = []
    for k in [n for n in names if n != "copy4"]:
        for m in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01):
            esc = mg[k] < m                          # escalate where the cheap corrector is unsure
            agree = torch.where(esc, ok["copy4"], ok[k]).float().mean()
            frows.append({"cheap": k, "escalate_if_margin_below": m, "share_escalated": float(esc.float().mean()),
                          "agreement": float(agree)})
    print_table(frows, ["cheap", "escalate_if_margin_below", "share_escalated", "agreement"],
                "escalation frontier: cheap corrector where its own margin >= m, 4-bit copy elsewhere")
    save_json(args.out, {"args": vars(args), "by_margin": rows, "frontier": frows,
                         "agreement": {k: float(ok[k].float().mean()) for k in names}})


if __name__ == "__main__":
    main()
