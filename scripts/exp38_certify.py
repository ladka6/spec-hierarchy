"""Experiment 38: certified acceptance. Can a full-depth low-precision copy certify tokens by its own margin?

Lossless speculation is capped by the best cheap model's agreement a: about 1/(1-a) accepted tokens per
target pass (0.97 for the 4-bit copy -> ~33). Idea: where the copy's own top-1 margin is large, its
argmax equals the target's with near certainty, so those tokens need no target check; the target is
called only at low-margin positions. Not lossless, but with a calibrated bound on the undetected
error rate (mismatches per token <= eps).

On target-greedy trajectories (teacher-forced: exact for the copy up to its first mismatch), for each
corrector and each logit-gap threshold m (certify where the corrector's top1-top2 logit gap >= m):
  coverage      share of positions certified
  err/token     certified positions where the corrector's argmax != the target's (undetected errors)
  tok/target    positions per target call, if the target runs once per uncertain position
  lossless ref  positions per corrector-target mismatch (the 1/(1-a) cap)
Calibration: thresholds chosen on half of the responses so that a 95% upper bound on err/token <= eps,
then evaluated on the other half (overall and per dataset). Per-position arrays are saved for re-analysis.

  python scripts/exp38_certify.py --correctors bnb4:Qwen/Qwen3-8B hf:Qwen/Qwen3-1.7B hf:Qwen/Qwen3-0.6B
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
from hspec.utils import RESULTS, print_table, save_json  # noqa: E402

GRID = [x / 4 for x in range(0, 81)]          # logit-gap thresholds 0 .. 20
EPS = [1e-2, 5e-3, 2e-3, 1e-3]


def gap_and_arg(logits):
    lg = logits.float()
    top = lg.topk(2, -1).values
    p = torch.softmax(lg, -1).topk(2, -1).values
    return top[:, 0] - top[:, 1], p[:, 0] - p[:, 1], lg.argmax(-1)


def upper(k, n, conf=0.95):
    """One-sided upper confidence bound on a binomial rate (Clopper-Pearson)."""
    if n == 0:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(conf, k + 1, n - k)) if k < n else 1.0
    except ImportError:
        return (k + 3) / n                     # rough fallback


def stats(gap, ok, m):
    n = len(gap)
    cert = gap >= m
    bad = int((cert & ~ok).sum())
    unc = int((~cert).sum())
    return {"coverage": float(cert.float().mean()), "err_per_tok": bad / max(n, 1), "bad": bad,
            "tok_per_target": n / max(unc, 1), "n": n}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16", help="only to generate the trajectories")
    ap.add_argument("--correctors", nargs="+",
                    default=["bnb4:Qwen/Qwen3-8B", "hf:Qwen/Qwen3-1.7B", "hf:Qwen/Qwen3-0.6B"])
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default="exp38_certify")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    trajs = []                                  # (dataset, response index, full ids, n0)
    for d in args.datasets:
        for pr in load_prompts(d, args.n):
            ids = encode(tok, pr)
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            trajs.append((d, len(trajs), x, ids.shape[1]))
        print(f"[{d}] trajectories done", flush=True)
    del draft

    rec = {"ds": [], "resp": [], "t_gap": [], "t_pm": []}
    for d, r, x, n0 in trajs:
        g, pm, a = gap_and_arg(target(x[None]).logits[0, n0 - 1 : len(x) - 1])
        rec["ds"] += [d] * len(g)
        rec["resp"].append(torch.full_like(g, r, dtype=torch.long))
        rec["t_gap"].append(g.cpu())
        rec["t_pm"].append(pm.cpu())
        rec.setdefault("t_arg", []).append(a)
    t_args = rec.pop("t_arg")
    resp = torch.cat(rec["resp"]).cpu()
    N = len(resp)
    names, gaps, oks, pms = [], {}, {}, {}
    for spec in args.correctors:
        try:
            mod = load_mid(spec)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {spec}: {e}", flush=True)
            continue
        k = spec.split(":", 1)[0] + ":" + spec.split("/")[-1]
        gl, ol, pl = [], [], []
        for (d, r, x, n0), ta in zip(trajs, t_args):
            g, pm, a = gap_and_arg(mod(x[None]).logits[0, n0 - 1 : len(x) - 1])
            gl.append(g.cpu())
            pl.append(pm.cpu())
            ol.append((a == ta).cpu())
        names.append(k)
        gaps[k], oks[k], pms[k] = torch.cat(gl), torch.cat(ol), torch.cat(pl)
        del mod
        torch.cuda.empty_cache()
        print(f"[{k}] agreement {float(oks[k].float().mean()):.4f} over {N} positions", flush=True)

    ds = rec["ds"]
    calib = resp % 2 == 1
    test = ~calib

    # 1. mismatch rate by the corrector's own logit gap (the certificate)
    bins = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 1e9]
    rows = []
    for k in names:
        for lo, hi in zip(bins, bins[1:]):
            sel = (gaps[k] >= lo) & (gaps[k] < hi)
            nb = int(sel.sum())
            mis = int((sel & ~oks[k]).sum())
            rows.append({"corrector": k, "own_gap": f"{lo}-{hi if hi < 1e8 else 'inf'}", "share": nb / N,
                         "mismatch": mis / max(nb, 1), "n_mismatch": mis})
    print_table(rows, ["corrector", "own_gap", "share", "mismatch", "n_mismatch"],
                f"mismatch with the target by the corrector's own top1-top2 logit gap ({N} positions)")

    # 2. frontier on all positions (descriptive)
    frows = []
    for k in names:
        lossless = N / max(int((~oks[k]).sum()), 1)
        for m in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15):
            s = stats(gaps[k], oks[k], m)
            frows.append({"corrector": k, "gap_min": m, **{c: s[c] for c in
                          ("coverage", "err_per_tok", "bad", "tok_per_target")}, "lossless_ref": lossless})
    print_table(frows, ["corrector", "gap_min", "coverage", "err_per_tok", "bad", "tok_per_target", "lossless_ref"],
                "certify where own gap >= gap_min; target called at every other position")

    # 3. calibrated: pick the smallest threshold whose 95% upper bound on err/token <= eps (calib half)
    crows = []
    for k in names:
        for eps in EPS:
            m_sel = None
            for m in GRID:
                s = stats(gaps[k][calib], oks[k][calib], m)
                if upper(s["bad"], s["n"]) <= eps:
                    m_sel = m
                    break
            if m_sel is None:
                crows.append({"corrector": k, "eps": eps, "gap_min": None})
                continue
            s = stats(gaps[k][test], oks[k][test], m_sel)
            row = {"corrector": k, "eps": eps, "gap_min": m_sel, "coverage": s["coverage"],
                   "test_err_per_tok": s["err_per_tok"], "tok_per_target": s["tok_per_target"],
                   "lossless_ref": int(test.sum()) / max(int((~oks[k][test]).sum()), 1)}
            for d in args.datasets:
                sel = test & torch.tensor([x == d for x in ds])
                sd = stats(gaps[k][sel], oks[k][sel], m_sel)
                row[f"tpt_{d}"] = sd["tok_per_target"]
                row[f"err_{d}"] = sd["err_per_tok"]
            crows.append(row)
    print_table(crows, ["corrector", "eps", "gap_min", "coverage", "test_err_per_tok", "tok_per_target", "lossless_ref"],
                "calibrated on half the responses (95% upper bound on err/token <= eps), evaluated on the other half")
    print_table([r for r in crows if r.get("gap_min") is not None],
                ["corrector", "eps"] + [f"tpt_{d}" for d in args.datasets] + [f"err_{d}" for d in args.datasets],
                "per dataset (test half): tokens per target call and undetected errors per token")

    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save({"ds": ds, "resp": resp, "t_gap": torch.cat(rec["t_gap"]), "t_pm": torch.cat(rec["t_pm"]),
                "gap": gaps, "ok": oks, "pm": pms}, RESULTS / f"{args.out}_positions.pt")
    save_json(args.out, {"args": vars(args), "by_gap": rows, "frontier": frows, "calibrated": crows,
                         "agreement": {k: float(oks[k].float().mean()) for k in names}})


if __name__ == "__main__":
    main()
