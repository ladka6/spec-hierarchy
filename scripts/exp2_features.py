"""Experiment 2: does the DFlash drafter still work when conditioned on the middle model's
hidden states instead of the target's?

The drafter is trained on bf16 target features. In the three-stage design the target is
not run between checks, so the drafter has to be fed the middle model's features. Only
middle models with the target's width/depth can be used (e.g. a quantized copy).

Configurations (all greedy, same prompts):
  T/T   features: target, verifier: target   (vanilla DFlash)
  M>T   features: mid,    verifier: target   (isolates the feature-substitution loss)
  M/M   features: mid,    verifier: mid      (stage 1 of the three-stage pipeline)

Also checks our re-implementation against dflash's own spec_generate on a few prompts.

Usage:
  python scripts/exp2_features.py --n 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import free, load_draft, load_mid, load_target, load_tokenizer, same_hidden_space  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--mids", nargs="+", default=["bnb4:Qwen/Qwen3-8B", "bnb8:Qwen/Qwen3-8B"])
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--ref-check", type=int, default=3, help="prompts to compare with dflash.spec_generate")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    prompts = {d: load_prompts(d, args.n) for d in args.datasets}

    # --- sanity: our loop == reference implementation ---
    ref = []
    for p in prompts[args.datasets[0]][: args.ref_check]:
        ids = encode(tok, p)
        ours = two_stage_generate(draft, target, target, target, ids, args.max_new, stops)
        theirs = draft.spec_generate(target=target, input_ids=ids, max_new_tokens=args.max_new,
                                     stop_token_ids=stops, temperature=0.0)
        same = torch.equal(ours.output_ids[0].cpu(), theirs[0].cpu())
        ref.append(same)
        print(f"[ref-check] identical to dflash.spec_generate: {same} "
              f"(ours {ours.output_ids.shape[1]} vs ref {theirs.shape[1]} tokens)")

    rows, per_prompt = [], []
    baseline = {}
    for d in args.datasets:
        taus, tps = [], []
        for i, p in enumerate(prompts[d]):
            r = two_stage_generate(draft, target, target, target, encode(tok, p), args.max_new, stops)
            baseline[(d, i)] = r.generated.cpu()
            taus.append(mean(r.rounds))
            tps.append(r.tok_per_s)
        rows.append({"mid": "-", "config": "T/T", "dataset": d, "tau": mean(taus), "tok/s": mean(tps)})
        print(f"[T/T | {d}] tau={mean(taus):.3f}", flush=True)

    for spec in args.mids:
        mid = load_mid(spec)
        if not same_hidden_space(target, mid):
            print(f"skip {spec}: hidden space differs from target, drafter cannot use its features")
            free(mid)
            continue
        for d in args.datasets:
            for name, feat, ver in (("M>T", mid, target), ("M/M", mid, mid)):
                taus, tps, match = [], [], []
                for i, p in enumerate(prompts[d]):
                    r = two_stage_generate(draft, target, ver, feat, encode(tok, p), args.max_new, stops)
                    taus.append(mean(r.rounds))
                    tps.append(r.tok_per_s)
                    b = baseline[(d, i)]
                    g = r.generated.cpu()
                    k = min(len(b), len(g))
                    match.append(torch.equal(b[:k], g[:k]) and len(b) == len(g))
                    per_prompt.append({"mid": spec, "config": name, "dataset": d, "i": i,
                                       "tau": taus[-1], "rounds": r.rounds})
                rows.append({"mid": spec, "config": name, "dataset": d, "tau": mean(taus),
                             "tok/s": mean(tps), "same_output_as_T/T": mean(match)})
                print(f"[{name} | {spec} | {d}] tau={mean(taus):.3f} "
                      f"same-output={mean(match):.2f}", flush=True)
        free(mid)

    print_table(rows, ["mid", "config", "dataset", "tau", "tok/s", "same_output_as_T/T"],
                "drafter acceptance (tau = tokens per round incl. bonus)")
    save_json("exp2_features", {"args": vars(args), "ref_check": ref, "rows": rows,
                                "per_prompt": per_prompt})


if __name__ == "__main__":
    main()
