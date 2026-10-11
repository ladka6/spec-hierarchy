"""Experiment 39: does the frozen target compute useful lookahead at mask slots? (probe test)

Idea under test: in the verify pass, append a few mask slots after the verified prefix; the target's
full-depth states at those slots ("its own lookahead") are given to the block drafter as context.
This only makes sense if those states carry information about future tokens beyond what the drafter
already has (the target's state at the last verified position + the anchor token).

Per anchor s (x_s = anchor / bonus token, known; target has processed x_0 .. x_{s-1}):
  A  base       h(s-1) and emb(x_s)                                  (what DFlash has, final layer)
  B  + slots    A + target states at m mask slots placed at positions s .. s+m-1 after x_0..x_{s-1}
  O  oracle     A + target states at the TRUE tokens x_s .. x_{s+m-1}   (real context, upper bound)
Probes (same architecture for every arm, trained on train sequences, evaluated on held-out ones)
predict x_{s+k}, k = 1..K, through the target's frozen LM head. Reported per offset k: top-1 accuracy,
and the share of the A -> O gap that the slots close: (B - A) / (O - A).
Also zero-shot: the target's own logits at slot s+k-1 vs x_{s+k} (no probe), per mask token.

  python scripts/exp39_lookahead.py --data .../hspec_lagft/data
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import load_target, load_tokenizer  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


class Probe(nn.Module):
    """Each 4096-d state -> shared 512-d projection (+ slot-index embedding) -> MLP -> K hidden states."""

    def __init__(self, n_in, H, K, d=512, width=2048):
        super().__init__()
        self.inp = nn.Linear(H, d)
        self.slot = nn.Parameter(torch.zeros(n_in, d))
        self.mlp = nn.Sequential(nn.LayerNorm(n_in * d), nn.Linear(n_in * d, width), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(width, K * H))
        self.K, self.H = K, H

    def forward(self, z):                        # z [b, n_in, H]
        u = self.inp(z) + self.slot
        return self.mlp(u.flatten(1)).view(-1, self.K, self.H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-seqs", type=int, default=2400)
    ap.add_argument("--val-seqs", type=int, default=300)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--m", type=int, default=8, help="mask slots")
    ap.add_argument("--K", type=int, default=8, help="offsets probed")
    ap.add_argument("--mask-tokens", nargs="+", default=["<|endoftext|>", "<|fim_pad|>"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="exp39_lookahead")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer(args.target)
    target = load_target(args.target, dev)
    target.requires_grad_(False)
    H = target.config.hidden_size
    pairs = [(t, tok.convert_tokens_to_ids(t)) for t in args.mask_tokens]
    pairs = [(t, i) for t, i in pairs if isinstance(i, int) and i != tok.unk_token_id]
    args.mask_tokens = [t for t, _ in pairs]
    mids = [i for _, i in pairs]
    print("mask tokens", list(zip(args.mask_tokens, mids)), flush=True)
    emb = target.model.embed_tokens
    head = target.lm_head

    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
        if len(data) >= 3 * args.n_seqs:
            break
    data = [d for d in data if len(d["ids"]) <= args.max_len and len(d["ids"]) - d["n_prompt"] > args.m + args.K + 4]
    rng.shuffle(data)
    data = data[: args.n_seqs]
    print(f"{len(data)} sequences", flush=True)

    feats = {"A": [], "S": [], "O": [], "y": [], "val": []}
    zs = {mt: torch.zeros(args.K) for mt in args.mask_tokens}
    zs10 = {mt: torch.zeros(args.K) for mt in args.mask_tokens}
    zn = 0
    with torch.inference_mode():
        for i, d in enumerate(data):
            ids = d["ids"].long().to(dev)
            T = len(ids)
            h = target.model(ids[None]).last_hidden_state[0]           # [T, H], final (normed) layer
            lo, hi = max(d["n_prompt"], 1), T - args.K - 1
            for s in rng.sample(range(lo, hi), min(args.anchors, hi - lo)):
                y = ids[s + 1 : s + 1 + args.K]
                slot_states = None
                for j, (mt, mid) in enumerate(zip(args.mask_tokens, mids)):
                    x = torch.cat([ids[:s], torch.full((args.m,), mid, device=dev)])
                    hs = target.model(x[None]).last_hidden_state[0, s : s + args.m]   # slots at s .. s+m-1
                    lg = head(hs)                                                   # slot s+k-1 -> x_{s+k}
                    kk = min(args.K, args.m)
                    pred = lg[:kk].argmax(-1)
                    zs[mt][:kk] += (pred == y[:kk]).float().cpu()
                    zs10[mt][:kk] += (lg[:kk].topk(10, -1).indices == y[:kk, None]).any(-1).float().cpu()
                    if j == 0:
                        slot_states = hs
                zn += 1
                feats["A"].append(torch.stack([h[s - 1], emb(ids[s])]).to(torch.bfloat16).cpu())
                feats["S"].append(slot_states.to(torch.bfloat16).cpu())
                feats["O"].append(h[s : s + args.m].to(torch.bfloat16).cpu())
                feats["y"].append(y.cpu())
                feats["val"].append(i < args.val_seqs)
            if (i + 1) % 200 == 0:
                print(f"[features] {i + 1}/{len(data)} sequences", flush=True)

    zrows = [{"mask_token": mt, **{f"top1@{k + 1}": float(zs[mt][k] / zn) for k in range(min(args.K, args.m))},
              **{f"top10@{k + 1}": float(zs10[mt][k] / zn) for k in (0, 3, min(args.K, args.m) - 1)}}
             for mt in args.mask_tokens]
    print_table(zrows, list(zrows[0]), f"zero-shot: target logits at mask slot s+k-1 vs x_(s+k) ({zn} anchors)")

    A = torch.stack(feats["A"])
    S = torch.stack(feats["S"])
    O = torch.stack(feats["O"])
    Y = torch.stack(feats["y"])
    val = torch.tensor(feats["val"])
    arms = {"A_base": A, "B_slots": torch.cat([A, S], 1), "O_oracle": torch.cat([A, O], 1)}
    W = head.weight                                                     # frozen LM head
    res = {}
    for name, Z in arms.items():
        torch.manual_seed(0)
        probe = Probe(Z.shape[1], H, args.K).to(dev)
        opt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=0.05)
        tr = (~val).nonzero().squeeze(1)
        va = val.nonzero().squeeze(1)

        def run(idx, train):
            hits = torch.zeros(args.K)
            lead = 0.0
            for b in range(0, len(idx), 256):
                bi = idx[b : b + 256]
                z = Z[bi].to(dev).float()
                y = Y[bi].to(dev)
                with torch.set_grad_enabled(train):
                    out = probe(z)                                       # [b, K, H]
                    logits = out.to(W.dtype) @ W.T
                    loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                else:
                    ok = logits.argmax(-1) == y
                    hits += ok.float().sum(0).cpu()
                    lead += float(ok.long().cumprod(1).sum())
            return hits / max(len(idx), 1), lead / max(len(idx), 1)

        best = None
        for ep in range(args.epochs):
            run(tr[torch.randperm(len(tr))], True)
            acc, lead = run(va, False)
            print(f"[{name} epoch {ep + 1}] val acc@1 {acc[0]:.3f} acc@4 {acc[3]:.3f} acc@{args.K} {acc[-1]:.3f} "
                  f"leading-correct {lead:.3f}", flush=True)
            if best is None or lead > best[1]:
                best = (acc, lead)
        res[name] = best
        del probe, opt
        torch.cuda.empty_cache()

    rows = []
    for k in range(args.K):
        a, b, o = (float(res[n][0][k]) for n in ("A_base", "B_slots", "O_oracle"))
        rows.append({"offset": k + 1, "A_base": a, "B_slots": b, "O_oracle": o,
                     "gap_closed": (b - a) / (o - a) if o - a > 1e-3 else float("nan")})
    rows.append({"offset": "lead", **{n: res[n][1] for n in res},
                 "gap_closed": (res["B_slots"][1] - res["A_base"][1]) / max(res["O_oracle"][1] - res["A_base"][1], 1e-6)})
    print_table(rows, ["offset", "A_base", "B_slots", "O_oracle", "gap_closed"],
                f"probe top-1 accuracy for x_(s+k) on held-out sequences (m={args.m} slots, mask '{args.mask_tokens[0]}')")
    save_json(args.out, {"args": vars(args), "zero_shot": zrows, "probe": rows})


if __name__ == "__main__":
    main()
