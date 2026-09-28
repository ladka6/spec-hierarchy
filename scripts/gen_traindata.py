"""Generate drafter training data: target-greedy responses to training prompts.

Prompts come from training splits only (no overlap with the evaluation sets gsm8k-test,
math500, humaneval, mt-bench): gsm8k train, MBPP train, Alpaca.

  python scripts/gen_traindata.py --shard 0 --nshards 4 --out /scratch-shared/$USER/hspec_lagft/data
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import format_prompt  # noqa: E402
from hspec.models import load_target, load_tokenizer  # noqa: E402


def training_prompts(n_gsm8k=2000, n_alpaca=2600, seed=0):
    from datasets import load_dataset

    out = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="train")]
    rng = random.Random(seed)
    rng.shuffle(out)
    out = out[:n_gsm8k]
    out += [r["text"] for r in load_dataset("google-research-datasets/mbpp", "full", split="train")]
    alp = [r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
           for r in load_dataset("tatsu-lab/alpaca", split="train")]
    rng.shuffle(alp)
    out += alp[:n_alpaca]
    rng.shuffle(out)
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = training_prompts()[args.shard :: args.nshards]
    tok = load_tokenizer(args.target)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_target(args.target)
    eos = model.generation_config.eos_token_id
    eos = set([eos] if isinstance(eos, int) else eos)
    texts = [format_prompt(tok, p) for p in prompts]
    records = []
    for b in range(0, len(texts), args.batch):
        enc = tok(texts[b : b + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for row, am in zip(gen, enc["attention_mask"]):
            n_pad = int((am == 0).sum())
            n_prompt = int(am.sum())
            seq = row[n_pad:].tolist()
            resp = seq[n_prompt:]
            cut = next((i for i, t in enumerate(resp) if t in eos), None)
            if cut is not None:
                resp = resp[: cut + 1]
            if len(resp) < 8:
                continue
            records.append({"ids": torch.tensor(seq[:n_prompt] + resp, dtype=torch.int32), "n_prompt": n_prompt})
        print(f"[shard {args.shard}] {min(b + args.batch, len(texts))}/{len(texts)} prompts, "
              f"{sum(len(r['ids']) for r in records)} tokens", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(records, out / f"shard{args.shard}.pt")
    print(f"saved {len(records)} sequences to {out / f'shard{args.shard}.pt'}")


if __name__ == "__main__":
    main()
