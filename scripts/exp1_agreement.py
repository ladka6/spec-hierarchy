"""Experiment 1: how often does a cheap middle model agree with the target?

Generates greedy target trajectories (fast, via vanilla DFlash, which is lossless under
greedy decoding), then teacher-forces each trajectory through the target and every
candidate middle model in a single forward pass and compares argmax predictions.

Reported per (mid, dataset):
  beta          P(mid argmax == target argmax), per token
  top5          P(target argmax in mid top-5)
  E[acc|P]      mean tokens the target would accept from a window of P mid-greedy tokens,
                measured from real agreement runs (no independence assumption)
  iid[acc|P]    the same under the i.i.d. model beta(1-beta^P)/(1-beta)
  P*(r)         optimal window from the cost model for cost ratios r = c_T / c

Usage (A100):
  python scripts/exp1_agreement.py --n 30 --datasets gsm8k humaneval
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import free, load_draft, load_mid, load_target, load_tokenizer, same_hidden_space  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import RESULTS, mean, print_table, save_json, slug  # noqa: E402

WINDOWS = [4, 8, 16, 32, 64, 128]
RATIOS = [10, 30, 100, 300]
MAX_RUN = 1024


def get_trajectories(args, target, draft, tok, dataset):
    path = RESULTS / f"traj_{slug(args.target)}_{dataset}_n{args.n}_m{args.max_new}.pt"
    if path.exists():
        return torch.load(path)
    prompts = load_prompts(dataset, args.n)
    stops = stop_ids(target, tok)
    trajs = []
    for i, p in enumerate(prompts):
        ids = encode(tok, p)
        r = two_stage_generate(draft, target, target, target, ids, args.max_new, stops)
        trajs.append({"ids": r.output_ids[0].cpu(), "n_input": r.num_input_tokens})
        print(f"[traj {dataset}] {i + 1}/{len(prompts)}  {r.num_output_tokens} tok  "
              f"tau={mean(r.rounds):.2f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save(trajs, path)
    return trajs


@torch.inference_mode()
def argmax_topk(model, ids, n_input, k=5):
    """Predictions for positions n_input .. len-1 (i.e. for each response token)."""
    out = model(ids.view(1, -1).to(model.device), use_cache=False)
    logits = out.logits[0, n_input - 1 : -1].float()
    return logits.argmax(-1).cpu(), logits.topk(k, dim=-1).indices.cpu()


def run_lengths(agree: torch.Tensor) -> torch.Tensor:
    """For each start index i: number of consecutive True values starting at i."""
    runs = torch.zeros(len(agree), dtype=torch.long)
    cur = 0
    for i in range(len(agree) - 1, -1, -1):
        cur = cur + 1 if agree[i] else 0
        runs[i] = min(cur, MAX_RUN)
    return runs


def p_star(beta: float, ratio: float) -> float:
    eps = max(1.0 - beta, 1e-6)
    return math.log1p(eps * ratio) / eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mids", nargs="+", default=[
        "bnb4:Qwen/Qwen3-8B", "bnb8:Qwen/Qwen3-8B",
        "hf:Qwen/Qwen3-4B", "hf:Qwen/Qwen3-1.7B", "hf:Qwen/Qwen3-0.6B",
    ])
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)

    trajs = {d: get_trajectories(args, target, draft, tok, d) for d in args.datasets}
    del draft
    free()

    # target predictions (teacher forced), plus sanity: does it reproduce the trajectory?
    tgt_pred = {}
    consistency = {}
    for d, ts in trajs.items():
        tgt_pred[d] = [argmax_topk(target, t["ids"], t["n_input"])[0] for t in ts]
        same = [(p == t["ids"][t["n_input"]:]).float().mean().item() for p, t in zip(tgt_pred[d], ts)]
        consistency[d] = mean(same)
        print(f"[sanity] {d}: teacher-forced target argmax matches its own trajectory "
              f"on {consistency[d]:.4f} of tokens")

    rows = []
    for spec in args.mids:
        mid = load_mid(spec)
        compat = same_hidden_space(target, mid)
        for d, ts in trajs.items():
            agrees, top5s, runs = [], [], []
            for t, tp in zip(ts, tgt_pred[d]):
                mp, mk = argmax_topk(mid, t["ids"], t["n_input"])
                agree = mp == tp
                agrees.append(agree)
                top5s.append((mk == tp[:, None]).any(-1))
                runs.append(run_lengths(agree))
            agree = torch.cat(agrees).float()
            runs = torch.cat(runs).float()
            beta = agree.mean().item()
            row = {
                "mid": spec, "dataset": d, "same_hidden_space": compat,
                "tokens": int(agree.numel()), "beta": beta,
                "top5": torch.cat(top5s).float().mean().item(),
            }
            for p in WINDOWS:
                row[f"E[acc|{p}]"] = runs.clamp(max=p).mean().item()
                row[f"iid[acc|{p}]"] = beta * (1 - beta ** p) / max(1 - beta, 1e-9)
            for r in RATIOS:
                row[f"P*(r={r})"] = p_star(beta, r)
            rows.append(row)
            print(f"[{spec} | {d}] beta={beta:.4f} top5={row['top5']:.4f} "
                  f"E[acc|32]={row['E[acc|32]']:.2f} E[acc|64]={row['E[acc|64]']:.2f}", flush=True)
        del mid
        free()

    print_table(rows, ["mid", "dataset", "beta", "top5", "E[acc|16]", "iid[acc|16]",
                       "E[acc|64]", "iid[acc|64]", "P*(r=30)", "P*(r=100)"],
                "middle-model agreement with target")
    save_json("exp1_agreement", {"args": vars(args), "consistency": consistency, "rows": rows})


if __name__ == "__main__":
    main()
