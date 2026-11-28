"""Are the stored responses the target's own greedy tokens, as every tau assumes?

tau counts a drafted token as accepted when it equals the stored next token. That is the verifier's decision only
if the stored token is the argmax of the target loaded here (bf16, transformers) given the stored prefix. The lag
fine-tuning data were generated with vLLM, the pair data with batched, left-padded generate; both can differ from
the unbatched forward at near-ties. This reports the share of response positions where they agree, and how far
into a block of 16 the first disagreement falls on average (the cap it puts on a perfect drafter's tau).

  python scripts/audit_data_greedy.py --data /scratch-shared/$USER/hspec_lagft/data
  python scripts/audit_data_greedy.py --target meta-llama/Llama-3.1-8B-Instruct --data /scratch-shared/$USER/hspec_taps/llama31_8b
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import load_target  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--seqs", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--block", type=int, default=16)
    args = ap.parse_args()
    data = []
    for f in sorted(Path(args.data).expanduser().glob("shard*.pt")):
        data += torch.load(f)
    data = [d for d in data if len(d["ids"]) <= args.max_len]
    random.Random(0).shuffle(data)                           # the first 48 are the validation sequences of train_lag.py
    target = load_target(args.target)
    agree = n = 0
    perfect, blocks, bad_seqs = 0.0, 0, 0
    for d in data[: args.seqs]:
        ids = d["ids"].long().cuda()
        lo = int(d["n_prompt"])
        pred = target(ids[None]).logits[0].argmax(-1)        # pred[p] = the target's token for position p + 1
        ok = (pred[lo - 1 : -1] == ids[lo:]).tolist()        # response positions lo .. T - 1
        agree += sum(ok)
        n += len(ok)
        bad_seqs += not all(ok)
        for s in range(0, len(ok) - args.block, args.block): # a perfect drafter's accepted length + 1, block by block
            run = ok[s + 1 : s + args.block]
            perfect += 1 + (run.index(False) if False in run else len(run))
            blocks += 1
    print(f"{args.target} on {args.data}: {min(args.seqs, len(data))} sequences, {n} response tokens")
    print(f"stored token = target argmax at {agree / max(n, 1):.4f} of the positions "
          f"({n - agree} disagreements, in {bad_seqs} sequences)")
    print(f"tau of a drafter that always proposes the target's own token: {perfect / max(blocks, 1):.3f} "
          f"of {args.block} ({blocks} blocks)")


if __name__ == "__main__":
    main()
