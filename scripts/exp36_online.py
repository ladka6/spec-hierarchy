"""Experiment 36: a drafter that learns during generation (per-request online adaptation), offline.

Every verify pass gives exact supervision (the target's tokens and features) that DFlash discards.
Here the drafter gets small LoRA adapters that are trained ONLINE within each response: every
`--update-every` verified tokens, a few gradient steps on blocks whose labels are already verified.
Adapters are reset for every new prompt (per-request learning, nothing carried across prompts).

Walks along target-greedy trajectories (the verified text), anchors every `--stride` positions:
  frozen   tau of the original drafter (accepted tokens per block, leading correct + 1)
  online   tau of the adapting drafter, evaluated BEFORE it trains on anything at or after the anchor
Reported overall, per dataset, and per position bucket in the response (does acceptance drift, and
does online learning help more later in long responses?).

  python scripts/exp36_online.py
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import block_loss, draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(x.dtype)

    def reset(self):
        with torch.no_grad():
            self.A.normal_(0, 0.01)
            self.B.zero_()


def add_lora(draft, r, alpha, names):
    mods = []
    for layer in draft.layers:
        for parent, attr in ((layer.self_attn, "q_proj"), (layer.self_attn, "k_proj"), (layer.self_attn, "v_proj"),
                             (layer.self_attn, "o_proj"), (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"),
                             (layer.mlp, "down_proj")):
            if attr not in names:
                continue
            m = LoRA(getattr(parent, attr), r, alpha)
            setattr(parent, attr, m)
            mods.append(m)
    return mods


def lead(hits):
    n = 0
    while n < len(hits) and hits[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=1024, help="long responses: learning needs room")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--update-every", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2, help="gradient steps per update")
    ap.add_argument("--train-blocks", type=int, default=16, help="verified blocks per gradient step")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--targets", nargs="+", default=["o_proj", "down_proj"], help="LoRA'd projections")
    ap.add_argument("--out", default="exp36_online")
    args = ap.parse_args()
    torch.manual_seed(0)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    target.requires_grad_(False)
    draft = load_draft(args.draft)
    draft.requires_grad_(False)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    loras = add_lora(draft, args.rank, args.alpha, args.targets)
    lparams = [p for m in loras for p in (m.A, m.B)]
    print(f"LoRA on {args.targets}: {sum(p.numel() for p in lparams) / 1e6:.2f} M trainable params", flush=True)

    def set_enabled(on):
        for m in loras:
            m.scale = (args.alpha / args.rank) if on else 0.0

    recs = []
    for d in args.datasets:
        for pr in load_prompts(d, args.n):
            ids = encode(tok, pr)
            n0 = ids.shape[1]
            with torch.no_grad():
                set_enabled(False)
                x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
                feats = extract_context_feature(target(x[None], output_hidden_states=True,
                                                       logits_to_keep=1).hidden_states, draft.target_layer_ids)
            for m in loras:
                m.reset()
            opt = torch.optim.Adam(lparams, lr=args.lr)
            rng = random.Random(0)
            last_update = n0
            for s in range(n0, len(x) - bs - 1, args.stride):
                # online update: train on blocks whose labels are already verified (anchor + 15 <= s)
                if s - last_update >= args.update_every and s - n0 > bs + 2:
                    set_enabled(True)
                    pool = list(range(n0, s - bs))
                    for _ in range(args.steps):
                        blocks = [(a, 0) for a in rng.sample(pool, min(args.train_blocks, len(pool)))]
                        opt.zero_grad(set_to_none=True)
                        loss, _ = block_loss(draft, target, feats[:, :s].detach(), x[:s], blocks, bs)
                        loss.backward()
                        opt.step()
                    last_update = s
                truth = x[s + 1 : s + bs]
                with torch.no_grad():
                    set_enabled(False)
                    t_frozen = 1 + lead((draft_logits_lagged(draft, target, feats, x, s, 0, bs).argmax(-1) == truth).tolist())
                    set_enabled(True)
                    t_online = 1 + lead((draft_logits_lagged(draft, target, feats, x, s, 0, bs).argmax(-1) == truth).tolist())
                recs.append({"ds": d, "pos": s - n0, "frozen": t_frozen, "online": t_online})
            print(f"[{d}] {len(recs)} anchors, response {len(x) - n0} tokens", flush=True)

    def row(name, rs):
        return {"group": name, "anchors": len(rs), "frozen": mean(r["frozen"] for r in rs),
                "online": mean(r["online"] for r in rs),
                "gain": mean(r["online"] for r in rs) / max(mean(r["frozen"] for r in rs), 1e-9) - 1}

    rows = [row("all", recs)] + [row(d, [r for r in recs if r["ds"] == d]) for d in args.datasets]
    buckets = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 10**9)]
    rows += [row(f"pos {a}-{b}", [r for r in recs if a <= r["pos"] < b]) for a, b in buckets
             if any(a <= r["pos"] < b for r in recs)]
    print_table(rows, ["group", "anchors", "frozen", "online", "gain"],
                f"per-request online adaptation (LoRA r{args.rank} on {args.targets}, update every "
                f"{args.update_every} tokens, {args.steps} steps): tau = accepted tokens per block, T=0")
    save_json(args.out, {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
