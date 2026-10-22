"""Experiment 66 (vLLM version): greedy agreement of off-the-shelf quantized checkpoints with the bf16 target.

Same quantity as exp66_copyagree.py, but every model runs in vLLM with its own fast kernels (AWQ / GPTQ /
compressed-tensors / torchao), so checkpoints that the HF stack cannot load are covered.

  predict  one model: top-1 prediction at every position of the held-out sequences (prompt_logprobs), saved as .pt
  stats    compare each saved model with the saved target: q, P(run>=L), cond_k (see exp66_copyagree.py)

  python scripts/exp66_vllm_agree.py predict --model Qwen/Qwen3-8B --teacher .../teacher.pt --out DIR
  python scripts/exp66_vllm_agree.py stats --out DIR --target Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

LENS = (2, 5, 10, 20, 33, 50, 100, 200)


def slug(m: str) -> str:
    return m.replace("/", "__")


def predict(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    seqs = torch.load(args.teacher)["val"]
    if args.n_val:
        seqs = seqs[: args.n_val]
    max_len = max(len(d["ids"]) for d in seqs)
    llm = LLM(model=args.model, max_model_len=max_len + 16, gpu_memory_utilization=0.85, enforce_eager=True,
              enable_prefix_caching=False)
    prompts = [TokensPrompt(prompt_token_ids=d["ids"].long().tolist()) for d in seqs]
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1))
    preds = []
    for d, o in zip(seqs, outs):
        lo = max(d["n_prompt"], 1)
        top = []
        for i in range(lo, len(d["ids"])):                       # prediction for token i given tokens < i
            lp = o.prompt_logprobs[i]
            top.append(min(lp.items(), key=lambda kv: kv[1].rank)[0])
        preds.append(torch.tensor(top, dtype=torch.long))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "preds": preds,
                "data": [d["ids"][max(d["n_prompt"], 1):].long() for d in seqs]}, Path(args.out) / f"{slug(args.model)}.pt")
    print(f"[predict] {args.model}: {sum(len(p) for p in preds)} positions", flush=True)


def runs(agree):
    r = [0] * (len(agree) + 1)
    for i in range(len(agree) - 1, -1, -1):
        r[i] = r[i + 1] + 1 if agree[i] else 0
    return torch.tensor(r[:-1])


def stats(args):
    out = Path(args.out)
    tgt = torch.load(out / f"{slug(args.target)}.pt")
    data_greedy = sum(int((p == d).sum()) for p, d in zip(tgt["preds"], tgt["data"]))
    n = sum(len(p) for p in tgt["preds"])
    print(f"[stats] target {args.target} (vLLM) = stored greedy text at {data_greedy / n:.3f} of {n} positions")
    rows, conds = [], {}
    for f in sorted(out.glob("*.pt")):
        cur = torch.load(f)
        if cur["model"] == args.target:
            continue
        reach, starts = {L: 0 for L in LENS}, {L: 0 for L in LENS}
        num, den = torch.zeros(64), torch.zeros(64)
        agree_all = []
        for a, b in zip(cur["preds"], tgt["preds"]):
            ag = (a == b)
            agree_all.append(ag)
            r = runs(ag.tolist())
            left = torch.arange(len(ag), 0, -1)
            for L in LENS:
                ok = left >= L
                starts[L] += int(ok.sum())
                reach[L] += int((r[ok] >= L).sum())
            for k in range(1, 65):
                ok = left >= k
                den[k - 1] += int((r[ok] >= k - 1).sum())
                num[k - 1] += int((r[ok] >= k).sum())
        q = float(torch.cat(agree_all).float().mean())
        rows.append({"copy": cur["model"], "q": q, **{f"P(run>={L})": reach[L] / max(starts[L], 1) for L in LENS}})
        conds[cur["model"]] = (num / den.clamp(min=1)).tolist()
    w = max(len(r["copy"]) for r in rows) if rows else 10
    print(f"\n{'copy':<{w}}  q      " + "  ".join(f"run>={L:<3}" for L in LENS))
    for r in rows:
        print(f"{r['copy']:<{w}}  {r['q']:.3f}  " + "  ".join(f"{r[f'P(run>={L})']:.3f}   " for L in LENS))
    ks = [1, 2, 4, 8, 16, 32, 64]
    print(f"\ncond_k\n{'copy':<{w}}  " + "  ".join(f"k{k:<4}" for k in ks))
    for m, c in conds.items():
        print(f"{m:<{w}}  " + "  ".join(f"{c[k - 1]:.3f}" for k in ks))
    (out / "summary.json").write_text(json.dumps({"rows": rows, "cond": conds}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["predict", "stats"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    predict(args) if args.mode == "predict" else stats(args)


if __name__ == "__main__":
    main()
