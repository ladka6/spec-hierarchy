"""Greedy responses of one target to a fixed prompt set, as a shard for the tap / feature measurements.

Prompts come from the benchmark sets the harness already uses (hspec.data.load_prompts: the DFlash benchmark
formats), the same first N of each set for every target, so pairs of different families are compared on the same
prompts. A set that is not in the local HF cache is skipped (jobs run offline) and reported.

  python scripts/gen_pairdata.py --target meta-llama/Llama-3.1-8B-Instruct --out /scratch-shared/$USER/hspec_taps/llama31_8b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import format_prompt, load_prompts  # noqa: E402
from hspec.models import load_target, load_tokenizer  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=150, help="prompts per dataset")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--min-new", type=int, default=64, help="drop shorter responses")
    args = ap.parse_args()

    prompts = []
    for ds in args.datasets:
        try:
            p = load_prompts(ds, args.n)
        except Exception as e:  # not cached / offline
            print(f"[gen] {ds}: skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
            continue
        print(f"[gen] {ds}: {len(p)} prompts", flush=True)
        prompts += p
    assert prompts, "no prompt set could be loaded"

    tok = load_tokenizer(args.target)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_target(args.target)
    eos = model.generation_config.eos_token_id or tok.eos_token_id
    eos = set([eos] if isinstance(eos, int) else eos)
    texts = [format_prompt(tok, p) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))          # similar lengths per batch
    records = []
    for b in range(0, len(order), args.batch):
        batch = [texts[i] for i in order[b : b + args.batch]]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for row, am in zip(gen, enc["attention_mask"]):
            n_pad, n_prompt = int((am == 0).sum()), int(am.sum())
            seq = row[n_pad:].tolist()
            resp = seq[n_prompt:]
            cut = next((i for i, t in enumerate(resp) if t in eos), None)
            if cut is not None:
                resp = resp[: cut + 1]
            if len(resp) >= args.min_new:
                records.append({"ids": torch.tensor(seq[:n_prompt] + resp, dtype=torch.int32), "n_prompt": n_prompt})
        print(f"[gen] {min(b + args.batch, len(order))}/{len(order)} prompts, {len(records)} kept, "
              f"{sum(len(r['ids']) - r['n_prompt'] for r in records)} response tokens", flush=True)
    assert records, "no response long enough"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(records, out / "shard0.pt")
    print(f"[gen] saved {len(records)} sequences to {out / 'shard0.pt'}")


if __name__ == "__main__":
    main()
