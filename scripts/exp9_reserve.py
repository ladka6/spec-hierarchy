"""Measured feature-ready reserve ablation on one, two, or three CUDA devices.

Example (three GPUs, explicit placement):
  python scripts/exp9_reserve.py --target-device cuda:0 --mid-device cuda:1 \
      --draft-device cuda:2 --reserves 0 1 2 4 8 --n 5 --max-new 256

The middle and draft workers serialize when assigned to the same device. No simulated
latency or assumed batched branch costs are used. Every run is checked against target AR.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import encode, load_prompts, stop_ids
from hspec.models import load_draft, load_mid, load_target, load_tokenizer
from hspec.pipeline import ar_generate
from hspec.reserve import ReserveConfig
from hspec.reserve_torch import TorchReserveBackend, reserve_generate


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--mid", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--target-device", default="cuda:0")
    ap.add_argument("--mid-device", default="cuda:1")
    ap.add_argument("--draft-device", default="cuda:1")
    ap.add_argument("--reserves", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    ap.add_argument("--fork-threshold", type=float, default=0.6)
    ap.add_argument("--max-ahead", type=int, default=64)
    ap.add_argument("--target-window", type=int, default=32)
    ap.add_argument("--target-fallback", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-sync", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k"])
    ap.add_argument("--prompt", action="append", help="literal prompt; replaces dataset loading")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", action="store_true", help="save per-call timing events")
    ap.add_argument("--output", type=Path, default=Path("results/exp9_reserve.json"))
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if min(args.n, args.max_new, args.repeats) < 1:
        ap.error("n, max-new, and repeats must be positive")
    if not 0 <= args.fork_threshold <= 1:
        ap.error("fork-threshold must lie in [0, 1]")
    configs = [(f"reserve-{r}", ReserveConfig(r, args.max_ahead, args.target_window,
                                              args.target_fallback)) for r in args.reserves]
    if not args.skip_sync:
        configs.insert(0, ("sync", ReserveConfig(0, args.max_ahead, args.target_window,
                                               False, synchronous=True)))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    target = load_target(args.target, args.target_device)
    mid = load_mid(args.mid, args.mid_device)
    draft = load_draft(args.draft, args.draft_device)
    tokenizer = load_tokenizer(args.target)
    stops = stop_ids(target, tokenizer)
    prompts = ({"literal": args.prompt} if args.prompt else
               {d: load_prompts(d, args.n, args.seed) for d in args.datasets})
    backend = TorchReserveBackend(draft, target, mid, max_length=1,
                                  reserve_size=max(args.reserves), fork_threshold=args.fork_threshold)
    cuda_devices = sorted({str(d) for d in backend.devices.values() if d.type == "cuda"})
    first = encode(tokenizer, next(iter(prompts.values()))[0], args.target_device)
    for _, cfg in configs:
        reserve_generate(draft, target, mid, first, min(16, args.max_new), stops, cfg,
                         backend=backend, fork_threshold=args.fork_threshold)

    records = []
    metadata = dict(args=vars(args) | {"output": str(args.output)}, torch=torch.__version__,
                    cuda=torch.version.cuda, placement=backend.resources,
                    devices={d: torch.cuda.get_device_name(d) for d in cuda_devices},
                    timing="measured_wall_clock", records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(args.output)

    for dataset, texts in prompts.items():
        for i, text in enumerate(texts):
            ids = encode(tokenizer, text, args.target_device)
            device = backend.devices["target"]
            # Existing AR timing synchronizes the current CUDA device.
            with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                ref = ar_generate(target, ids, args.max_new, stops)
            for repeat in range(args.repeats):
                order = list(configs)
                rng.shuffle(order)
                for name, cfg in order:
                    for d in cuda_devices:
                        torch.cuda.synchronize(d)
                        torch.cuda.reset_peak_memory_stats(d)
                    r = reserve_generate(draft, target, mid, ids, args.max_new, stops, cfg,
                                         backend=backend, fork_threshold=args.fork_threshold)
                    match = torch.equal(r.output_ids, ref.output_ids)
                    # Prefill chooses the first token and is outside decode_time.
                    decoded = max(r.num_output_tokens - 1, 0)
                    record = dict(dataset=dataset, i=i, repeat=repeat, config=name,
                                  match=match, tokens=r.num_output_tokens, timed_tokens=decoded,
                                  seconds=r.decode_time,
                                  confirmed_tok_s=decoded / r.decode_time if r.decode_time else 0,
                                  allocated_gpu_seconds_per_token=(len(cuda_devices) * r.decode_time / decoded
                                                                   if decoded else None),
                                  peak_allocated_bytes={d: torch.cuda.max_memory_allocated(d) for d in cuda_devices},
                                  metrics=r.reserve_stats)
                    if args.trace:
                        record["trace"] = r.reserve_trace
                    records.append(record)
                    save()
                    print(f"{dataset}/{i} repeat={repeat} {name}: "
                          f"{record['confirmed_tok_s']:.2f} tok/s match={match}", flush=True)
                    if not match:
                        raise RuntimeError(f"greedy mismatch in {name}; results saved to {args.output}")


if __name__ == "__main__":
    main()
