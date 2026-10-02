"""Middle-model candidates for a three-model PEARL pipeline, measured in vLLM (batch 1).

  traj   greedy trajectories of the target (token ids), the reference for agreement
  agree  per-token top-1 disagreement e of a candidate with the target on those trajectories
         (teacher forced: prompt_logprobs on prompt + target output; only the output part counts)
  hedge  where the disagreements are: rank of the target's token in the candidate's top-5 and the
         candidate's top-1/top-2 margin at every position -> how many rollbacks side branches at
         low-margin positions would catch (h) and how many positions they cost (r)

Speed of a candidate (plain and with DFlash) comes from vllm_dflash.py --model <candidate>.

  python scripts/vllm_middle.py traj  --model Qwen/Qwen3-8B --prompts p.json --out traj.json
  python scripts/vllm_middle.py agree --model Qwen/Qwen3-8B-AWQ --traj traj.json --out e.json
"""

from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["traj", "agree", "hedge"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts")
    ap.add_argument("--traj")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.85)

    if args.mode == "traj":
        prompts = json.load(open(args.prompts))
        tok = llm.get_tokenizer()
        rows = []
        for p in prompts:
            ids = tok(p["text"], add_special_tokens=False)["input_ids"]
            o = llm.generate([{"prompt_token_ids": ids}], SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
                             use_tqdm=False)
            rows.append({"dataset": p["dataset"], "prompt": ids, "output": list(o[0].outputs[0].token_ids)})
        json.dump(rows, open(args.out, "w"))
        print(f"wrote {len(rows)} trajectories", flush=True)
        return

    rows = json.load(open(args.traj))
    if args.mode == "hedge":
        return hedge(llm, rows, args)
    dis = tot = 0
    per = {}
    for r in rows:
        seq = r["prompt"] + r["output"]
        o = llm.generate([{"prompt_token_ids": seq}], SamplingParams(max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        plp = o[0].prompt_logprobs
        n0 = len(r["prompt"])
        d = n = 0
        for i in range(n0, len(seq)):
            lp = plp[i]
            if lp is None or seq[i] not in lp:
                continue
            n += 1
            d += int(lp[seq[i]].rank != 1)
        dis += d
        tot += n
        a = per.setdefault(r["dataset"], [0, 0])
        a[0] += d
        a[1] += n
    res = {"model": args.model, "e": dis / max(tot, 1), "positions": tot,
           "e_per_dataset": {k: v[0] / max(v[1], 1) for k, v in per.items()}}
    print(json.dumps(res), flush=True)
    json.dump(res, open(args.out, "w"))


def hedge(llm, rows, args):
    ranks, margins = [], []
    for r in rows:
        seq = r["prompt"] + r["output"]
        o = llm.generate([{"prompt_token_ids": seq}], SamplingParams(max_tokens=1, prompt_logprobs=5), use_tqdm=False)
        plp = o[0].prompt_logprobs
        for i in range(len(r["prompt"]), len(seq)):
            lp = plp[i]
            if not lp:
                continue
            by_rank = {v.rank: v.logprob for v in lp.values()}
            if 1 not in by_rank or 2 not in by_rank:
                continue
            rk = lp[seq[i]].rank if seq[i] in lp else 99
            ranks.append(rk)
            margins.append(by_rank[1] - by_rank[2])
    n = len(ranks)
    dis = [i for i in range(n) if ranks[i] != 1]
    e = len(dis) / n
    cov = {k: sum(ranks[i] <= k for i in dis) / max(len(dis), 1) for k in (2, 3, 5)}
    curve = []
    for tau in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        flagged = [m < tau for m in margins]
        r_ = sum(flagged) / n
        h2 = sum(flagged[i] and ranks[i] <= 2 for i in dis) / max(len(dis), 1)
        h3 = sum(flagged[i] and ranks[i] <= 3 for i in dis) / max(len(dis), 1)
        curve.append({"margin_lt": tau, "flagged_share": r_, "h_top2": h2, "h_top3": h3})
    res = {"model": args.model, "positions": n, "e": e, "coverage_of_disagreements": cov, "curve": curve}
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
