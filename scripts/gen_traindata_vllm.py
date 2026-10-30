"""Generate drafter training data with vLLM (much faster than gen_traindata.py's HF generate loop).

Same prompts (training splits only: GSM8K train, MBPP train, Alpaca; no overlap with the eval sets) and the same
record format ({"ids": int32 tensor prompt + response, "n_prompt": int}), so every script that reads
gen_traindata.py shards reads these too.

Two steps, so the vLLM environment does not need `datasets`:
  prompts  (hspec venv, CPU is enough, e.g. on the login node): build and tokenize the prompts -> prompts.pt
  gen      (vllm venv, 1 GPU): greedy responses -> shard files in --out

  python scripts/gen_traindata_vllm.py prompts --n-gsm8k 7400 --n-alpaca 15000 --file /scratch-shared/$USER/prompts.pt
  python scripts/gen_traindata_vllm.py gen --file /scratch-shared/$USER/prompts.pt --out /scratch-shared/$USER/hspec_lagft/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_prompts(args):
    from transformers import AutoTokenizer

    from gen_traindata import training_prompts
    from hspec.data import format_prompt

    tok = AutoTokenizer.from_pretrained(args.target)
    prompts = training_prompts(args.n_gsm8k, args.n_alpaca)
    ids = [tok(format_prompt(tok, p), add_special_tokens=False)["input_ids"] for p in prompts]
    ids = [x for x in ids if len(x) <= args.max_prompt]
    torch.save({"target": args.target, "prompt_ids": ids}, args.file)
    print(f"saved {len(ids)} tokenized prompts to {args.file}")


def generate(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    blob = torch.load(args.file)
    ids = blob["prompt_ids"]
    llm = LLM(model=blob["target"], max_model_len=max(len(x) for x in ids) + args.max_new + 8,
              gpu_memory_utilization=0.90, enable_prefix_caching=False)
    eos = llm.get_tokenizer().eos_token_id
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per = (len(ids) + args.shards - 1) // args.shards
    total = 0
    for k in range(args.shards):
        chunk = ids[k * per : (k + 1) * per]
        if not chunk:
            continue
        res = llm.generate([TokensPrompt(prompt_token_ids=x) for x in chunk], sp)
        recs = []
        for p, r in zip(chunk, res):
            resp = list(r.outputs[0].token_ids)
            if len(resp) < 8:
                continue
            recs.append({"ids": torch.tensor(p + resp, dtype=torch.int32), "n_prompt": len(p)})
        fn = out / f"shard{k}{args.tag}.pt"
        torch.save(recs, fn)
        total += len(recs)
        print(f"[gen] shard {k}: {len(recs)} sequences, {sum(len(r['ids']) for r in recs)} tokens -> {fn}",
              flush=True)
    print(f"[done] {total} sequences (eos id {eos})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prompts")
    a.add_argument("--target", default="Qwen/Qwen3-8B")
    a.add_argument("--n-gsm8k", type=int, default=7400)
    a.add_argument("--n-alpaca", type=int, default=15000)
    a.add_argument("--max-prompt", type=int, default=1024)
    a.add_argument("--file", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--file", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-new", type=int, default=384)
    g.add_argument("--shards", type=int, default=4)
    g.add_argument("--tag", default="")
    args = ap.parse_args()
    build_prompts(args) if args.cmd == "prompts" else generate(args)


if __name__ == "__main__":
    main()
