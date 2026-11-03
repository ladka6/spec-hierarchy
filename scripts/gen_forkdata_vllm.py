"""Fork-branched drafter training data with vLLM (for exp104, the mixture block drafter).

For every training prompt: the target's greedy response (as gen_traindata_vllm.py), plus, at positions where the
target's top-2 log-prob gap is small (a fork), the target's greedy continuation after its 2nd-choice token. So every
anchor whose block covers a fork has the real alternative block the target would have written down the other branch.

Record format (a superset of gen_traindata shards, so other scripts can still read the "ids" / "n_prompt" part):
  {"ids": int32 tensor prompt + greedy response, "n_prompt": int,
   "forks": [(pos, alt_token, gap, cont), ...]}
  pos   absolute index in ids of the token that the fork replaces (ids[pos] is the greedy token)
  alt   the target's 2nd-choice token at pos
  gap   log p(top1) - log p(top2) at pos (nats)
  cont  int32 tensor: the target's greedy continuation after ids[:pos] + [alt] (--cont tokens)

Prompts: the prompts.pt file built by `gen_traindata_vllm.py prompts` (training splits only).

  python scripts/gen_forkdata_vllm.py --file /scratch-shared/$USER/prompts.pt --out /scratch-shared/$USER/forkdata
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch


def top2(lp_dict):
    """vLLM per-token logprob dict -> [(token, logprob)] sorted best first (top-2)."""
    items = sorted(((t, float(v.logprob)) for t, v in lp_dict.items()), key=lambda z: -z[1])
    return items[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="prompts.pt from gen_traindata_vllm.py prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--gap", type=float, default=1.0,
                    help="fork if log p1 - log p2 < gap (1.0 nats: the 2nd choice has at least 37%% of the 1st)")
    ap.add_argument("--max-forks", type=int, default=16, help="forks kept per sequence (spread at random)")
    ap.add_argument("--cont", type=int, default=16, help="tokens of greedy continuation after the 2nd choice")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    blob = torch.load(args.file)
    ids = blob["prompt_ids"]
    if args.limit:
        ids = ids[: args.limit]
    llm = LLM(model=blob["target"], max_model_len=max(len(x) for x in ids) + args.max_new + args.cont + 8,
              gpu_memory_utilization=0.90, enable_prefix_caching=True)
    sp_main = SamplingParams(temperature=0.0, max_tokens=args.max_new, logprobs=2)
    sp_cont = SamplingParams(temperature=0.0, max_tokens=args.cont)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per = (len(ids) + args.shards - 1) // args.shards
    tot_seq = tot_fork = tot_pos = tot_cand = 0
    for k in range(args.shards):
        chunk = ids[k * per : (k + 1) * per]
        if not chunk:
            continue
        res = llm.generate([TokensPrompt(prompt_token_ids=x) for x in chunk], sp_main)
        recs, branch_prompts, owners = [], [], []
        for p, r in zip(chunk, res):
            o = r.outputs[0]
            resp = list(o.token_ids)
            if len(resp) < 24 or o.logprobs is None:
                continue
            cands = []
            for j, lpd in enumerate(o.logprobs[: len(resp)]):
                t2 = top2(lpd)
                tot_pos += 1
                if len(t2) < 2 or t2[0][0] != resp[j]:
                    continue                   # greedy token must be the top-1 (vLLM greedy, should always hold)
                gap = t2[0][1] - t2[1][1]
                if gap < args.gap:
                    cands.append((len(p) + j, int(t2[1][0]), float(gap)))
            tot_cand += len(cands)
            if len(cands) > args.max_forks:
                cands = sorted(rng.sample(cands, args.max_forks))
            full = p + resp
            ri = len(recs)
            recs.append({"ids": torch.tensor(full, dtype=torch.int32), "n_prompt": len(p), "forks": []})
            for pos, alt, gap in cands:
                branch_prompts.append(TokensPrompt(prompt_token_ids=full[:pos] + [alt]))
                owners.append((ri, pos, alt, gap))
        if branch_prompts:
            bres = llm.generate(branch_prompts, sp_cont)
            for (ri, pos, alt, gap), br in zip(owners, bres):
                cont = list(br.outputs[0].token_ids)
                if len(cont) < 4:
                    continue
                recs[ri]["forks"].append((pos, alt, gap, torch.tensor(cont, dtype=torch.int32)))
        nf = sum(len(r["forks"]) for r in recs)
        fn = out / f"fork_shard{k}.pt"
        torch.save(recs, fn)
        tot_seq += len(recs)
        tot_fork += nf
        print(f"[gen] shard {k}: {len(recs)} sequences, {nf} forks -> {fn}", flush=True)
    print(f"[done] {tot_seq} sequences, {tot_fork} forks kept ({tot_fork / max(tot_seq, 1):.1f} per sequence); "
          f"{tot_cand} of {tot_pos} response positions were forks ({tot_cand / max(tot_pos, 1):.1%})", flush=True)


if __name__ == "__main__":
    main()
