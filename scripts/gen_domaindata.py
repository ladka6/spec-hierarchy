"""Per-domain drafter data: Qwen3-8B greedy responses, one directory per domain (train + val shards).

Domains (training splits only; the usual evaluation sets gsm8k-test, math500, humaneval, mt-bench are not used):
  math    gsm8k train questions, non-thinking mode
  think   the same gsm8k questions, thinking mode (long reasoning traces)
  code    MBPP (all splits; MBPP is not an evaluation set here)
  chat    Alpaca instructions

  python scripts/gen_domaindata.py --domain math --n 1100 --out /scratch-shared/$USER/hspec_domains
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import load_target, load_tokenizer  # noqa: E402


def prompts_for(domain, n, seed=0):
    from datasets import load_dataset

    rng = random.Random(seed)
    if domain in ("math", "think"):
        out = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="train")]
    elif domain == "code":
        out = []
        for split in ("train", "validation", "test", "prompt"):
            out += [r["text"] for r in load_dataset("google-research-datasets/mbpp", "full", split=split)]
    elif domain == "chat":
        out = [r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
               for r in load_dataset("tatsu-lab/alpaca", split="train")]
    else:
        raise ValueError(domain)
    rng.shuffle(out)
    return out[:n]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--domain", required=True, choices=["math", "think", "code", "chat"])
    ap.add_argument("--n", type=int, default=1100)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=0, help="0 = 384, or 1024 for think")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    max_new = args.max_new or (1024 if args.domain == "think" else 384)

    prompts = prompts_for(args.domain, args.n)
    tok = load_tokenizer(args.target)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_target(args.target)
    eos = model.generation_config.eos_token_id
    eos = set([eos] if isinstance(eos, int) else eos)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=args.domain == "think") for p in prompts]
    records = []
    bs = args.batch if args.domain != "think" else max(1, args.batch // 2)
    for b in range(0, len(texts), bs):
        enc = tok(texts[b : b + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for row, am in zip(gen, enc["attention_mask"]):
            n_pad = int((am == 0).sum())
            n_prompt = int(am.sum())
            seq = row[n_pad:].tolist()
            resp = seq[n_prompt:]
            cut = next((i for i, t in enumerate(resp) if t in eos), None)
            if cut is not None:
                resp = resp[: cut + 1]
            if len(resp) < 32:
                continue
            records.append({"ids": torch.tensor(seq[:n_prompt] + resp, dtype=torch.int32), "n_prompt": n_prompt})
        print(f"[{args.domain}] {min(b + bs, len(texts))}/{len(texts)} prompts, {len(records)} kept", flush=True)
    out = Path(args.out) / args.domain
    out.mkdir(parents=True, exist_ok=True)
    torch.save(records[: args.val], out / "val.pt")
    torch.save(records[args.val :], out / "train.pt")
    print(f"[{args.domain}] saved {min(len(records), args.val)} val / {max(0, len(records) - args.val)} train to {out}")


if __name__ == "__main__":
    main()
