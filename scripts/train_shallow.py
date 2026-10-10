"""Train the shallow-feature adapter (hspec/shallow.py): target's first k layers + adapter as the
drafter's feature source for recently drafted positions. Target and drafter frozen.

Loss = drafter CE on depth-lagged blocks (real features up to s-g, adapter features for the
last g positions; g uniform in 1..max-lag) + lam_feat * (1 - cos) to the real feature slices
+ lam_pred * CE of the adapter's next-token prediction against the target's greedy token.

Validation (held-out sequences, lags 4 / 8 / 16): tau (accepted tokens per block, greedy) with
  real     the target's real features for the lagged positions (ceiling)
  shallow  only the exact shallow slices, deep slices zeroed (no adapter, floor)
  adapter  shallow + adapter
plus the adapter's token agreement with the target (the copy's was 0.969 at T=0) and mean cosine.

  python scripts/train_shallow.py --exit 12 --adapter-layers 2 --data .../hspec_lagft/data --out .../k12_a2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.fusion import accepted  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.shallow import ShallowAdapter, adapter_aux, assemble, drafter_block_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exit", type=int, default=12, help="number of target layers run in the refresh pass")
    ap.add_argument("--adapter-layers", type=int, default=2, help="trainable layers from target k.. (0 = MLP)")
    ap.add_argument("--max-lag", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=16, help="blocks per sequence")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lam-feat", type=float, default=1.0)
    ap.add_argument("--lam-pred", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-drafter", action="store_true", help="also fine-tune the drafter (fp32 master weights)")
    ap.add_argument("--draft-lr", type=float, default=3e-5)
    ap.add_argument("--p-zero", type=float, default=0.25,
                    help="with --train-drafter: share of blocks with real features only (keeps normal drafting)")
    ap.add_argument("--freeze-adapter", action="store_true",
                    help="keep the adapter at its zero init (missing slices = h_k): drafter-only control")
    ap.add_argument("--copy", default=None,
                    help="full-depth low-bit copy as the source instead of shallow layers + adapter "
                         "(load_mid spec, e.g. rtn2:Qwen/Qwen3-8B); the adapter is not used")
    ap.add_argument("--copy-device", default="cuda:1")
    ap.add_argument("--small", default=None,
                    help="small same-family model (e.g. Qwen/Qwen3-0.6B) + mapper as the source (proposal 1)")
    ap.add_argument("--n-sel", type=int, default=6, help="small-model layers fed to the mapper")
    ap.add_argument("--ridge-seqs", type=int, default=400, help="training sequences for the ridge init")
    ap.add_argument("--map-hidden", type=int, default=2048)
    ap.add_argument("--map-lr", type=float, default=1e-4)
    ap.add_argument("--freeze-map", action="store_true", help="keep the mapper at its ridge init")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.copy or args.small:
        args.freeze_adapter = True
    tag = (args.copy.split(":")[0] if args.copy else args.small.split("/")[-1] + ("_ridge" if args.freeze_map else "")
           if args.small else f"k{args.exit}_a{args.adapter_layers}") + \
        ("_joint" if args.train_drafter else "") + ("_noadapt" if args.freeze_adapter and not (args.copy or args.small) else "")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
    data = [d for d in data if len(d["ids"]) <= args.max_len]
    rng.shuffle(data)
    val, train = data[: args.val_seqs], data[args.val_seqs :]
    if args.max_train:
        train = train[: args.max_train]
    print(f"[{tag}] {len(train)} train / {len(val)} val sequences", flush=True)

    target = load_target(args.target)
    target.requires_grad_(False)
    draft = load_draft(args.draft)
    draft.requires_grad_(False)
    if args.train_drafter:
        draft.float()
        draft.requires_grad_(True)
        draft.train()
    lids = list(draft.target_layer_ids)
    copy = None
    if args.copy:
        from hspec.models import load_mid
        copy = load_mid(args.copy, args.copy_device)
        copy.requires_grad_(False)
    small = mapper = None
    if args.small:
        from hspec.smallmap import SmallMapper, ridge_fit, small_inputs, small_layers
        small = load_target(args.small, args.device)
        small.requires_grad_(False)
        sel = small_layers(small.config.num_hidden_layers, args.n_sel)
        with torch.no_grad():
            W_r, b_r = ridge_fit(small, target, lids, sel, train[: args.ridge_seqs], args.device,
                                 log=lambda m: print(f"[{tag}] {m}", flush=True))
        mapper = SmallMapper(W_r.shape[0], W_r.shape[1], args.map_hidden).to(args.device)
        with torch.no_grad():
            mapper.lin.weight.copy_(W_r.T)
            mapper.lin.bias.copy_(b_r)
        mapper.requires_grad_(not args.freeze_map)
        print(f"[{tag}] small model layers {sel}, mapper {sum(p.numel() for p in mapper.parameters()) / 1e6:.0f} M "
              f"params", flush=True)
    if copy is not None or small is not None:   # unused placeholder: all slices "exact", no layers
        args.exit, args.adapter_layers = target.config.num_hidden_layers, 0
    adapter = ShallowAdapter(target, lids, args.exit, args.adapter_layers).to(args.device)
    if args.freeze_adapter:
        adapter.requires_grad_(False)
    params = [p for p in adapter.parameters() if p.requires_grad]
    dparams = [p for p in draft.parameters() if p.requires_grad]
    print(f"[{tag}] exact slices: {[l for j, l in enumerate(lids) if j not in adapter.missing]}, "
          f"predicted: {[lids[j] for j in adapter.missing]}; adapter {sum(p.numel() for p in params) / 1e6:.0f} M "
          f"params", flush=True)
    mparams = [p for p in mapper.parameters() if p.requires_grad] if mapper is not None else []
    groups = ([{"params": params, "lr": args.lr}] if params else []) + \
        ([{"params": dparams, "lr": args.draft_lr}] if dparams else []) + \
        ([{"params": mparams, "lr": args.map_lr}] if mparams else [])
    params = params + mparams
    if not groups:
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)   # placeholder: evaluation only
    elif args.device == "cpu":
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=0.0, betas=(0.9, 0.95))
    print(f"[{tag}] trainable: adapter {sum(p.numel() for p in params) / 1e6:.0f} M, "
          f"drafter {sum(p.numel() for p in dparams) / 1e6:.0f} M", flush=True)
    total = int(len(train) * args.epochs / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    bs = draft.block_size
    rotary = target.model.rotary_emb
    H_t = target.config.hidden_size

    def target_pass(ids):
        with torch.no_grad():
            out = target(ids[None], output_hidden_states=True)
            hs = out.hidden_states
            return hs, extract_context_feature(hs, lids), out.logits[0].argmax(-1)

    ridge_cache = {}

    def run_adapter(hs, T, ids=None):
        """(missing slices, final pre-norm state, assembled features); copy mode: ([], copy's greedy
        predictions, copy features)."""
        if small is not None:
            with torch.no_grad():
                x, sp = small_inputs(small, ids, sel)
            ridge_cache["x"] = x
            return [], sp, mapper(x)[None]
        if copy is not None:
            with torch.no_grad():
                co = copy(ids.to(args.copy_device)[None], output_hidden_states=True)
                cf = extract_context_feature(co.hidden_states, lids).to(args.device)
                return [], co.logits[0].argmax(-1).to(args.device), cf
        pos = torch.arange(T, device=args.device)[None]
        missing, final = adapter(hs[args.exit], rotary, pos)
        return missing, final, assemble(hs, lids, args.exit, missing)

    def blocks_of(d, n, r, lag=None):
        T = len(d["ids"])
        out = []
        for _ in range(n):
            g = lag if lag is not None else r.randint(1, args.max_lag)
            if lag is None and args.train_drafter and r.random() < args.p_zero:
                g = 0
            lo, hi = max(d["n_prompt"], g + 1), T - 2
            if hi <= lo:
                continue
            out.append((r.randrange(lo, hi), g))
        return out

    history = []

    def evaluate(step):
        adapter.eval()
        draft.eval()
        res = {}
        with torch.no_grad():
            for lag in (4, 8, 16):
                vrng = random.Random(123)
                taus = {"real": [], "shallow": [], "adapter": []}
                agree, cos = [], []
                for d in val:
                    ids = d["ids"].long().to(args.device)
                    blocks = blocks_of(d, 8, vrng, lag)
                    if not blocks:
                        continue
                    hs, real, tp = target_pass(ids)
                    missing, final, alt = run_adapter(hs, len(ids), ids)
                    rows = torch.tensor(sorted({t for s, g in blocks for t in range(s - g, s)}), device=ids.device)
                    if copy is not None or small is not None:
                        # "shallow" column: copy features again (copy mode) / the ridge-init map (small mode)
                        zero = alt if small is None else (ridge_cache["x"] @ W_r + b_r)[None]
                        agree.append(float((final[rows] == tp[rows]).float().mean()))
                        cos.append(float(torch.nn.functional.cosine_similarity(
                            alt[0, rows].float(), real[0, rows].float(), dim=-1).mean()))
                    else:
                        zero = assemble(hs, lids, args.exit, [torch.zeros_like(m) for m in missing])
                        fl, _, ag = adapter_aux(adapter, target, missing, final, hs, tp, rows)
                        agree.append(float(ag))
                        cos.append(1 - float(fl))
                    for name, src in (("real", real), ("shallow", zero), ("adapter", alt)):
                        _, correct = drafter_block_loss(draft, target, real, src, ids, blocks, bs, args.gamma)
                        taus[name] += [accepted(r) for r in correct if None not in r]
                res[lag] = {k: sum(v) / max(len(v), 1) for k, v in taus.items()}
                res[lag]["agree"] = sum(agree) / len(agree)
                res[lag]["cos"] = sum(cos) / len(cos)
        adapter.train()
        if args.train_drafter:
            draft.train()
        history.append({"step": step, "val": res})
        msg = " | ".join(f"lag{g}: tau real {v['real']:.2f} shallow {v['shallow']:.2f} adapter {v['adapter']:.2f}"
                         f" agree {v['agree']:.3f} cos {v['cos']:.3f}" for g, v in res.items())
        print(f"[{tag} step {step}] {msg}", flush=True)

    evaluate(0)
    if not (params or dparams):
        print(f"[{tag}] nothing to train (frozen evaluation only)", flush=True)
        return
    step, t0 = 0, time.time()
    order = list(range(len(train)))
    while step < total:
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if step >= total:
                break
            opt.zero_grad(set_to_none=True)
            acc = {"draft": 0.0, "feat": 0.0, "pred": 0.0, "agree": 0.0}
            n = 0
            for j in order[i : i + args.batch]:
                d = train[j]
                ids = d["ids"].long().to(args.device)
                blocks = blocks_of(d, args.blocks, rng)
                if not blocks:
                    continue
                hs, real, tp = target_pass(ids)
                missing, final, alt = run_adapter(hs, len(ids), ids)
                ld, _ = drafter_block_loss(draft, target, real, alt, ids, blocks, bs, args.gamma)
                rows = torch.tensor(sorted({t for s, g in blocks for t in range(s - g, s)}), device=ids.device)
                if len(rows) and small is not None and not args.freeze_map:
                    lf = torch.stack([1 - torch.nn.functional.cosine_similarity(
                        alt[0, rows, j * H_t : (j + 1) * H_t].float(), real[0, rows, j * H_t : (j + 1) * H_t].float(),
                        dim=-1).mean() for j in range(len(lids))]).mean()
                    lp = torch.zeros((), device=ids.device)
                    ag = (final[rows] == tp[rows]).float().mean()
                elif len(rows) and not args.freeze_adapter:
                    lf, lp, ag = adapter_aux(adapter, target, missing, final, hs, tp, rows)
                else:
                    lf = lp = ag = torch.zeros((), device=ids.device)
                loss = ld + args.lam_feat * lf + args.lam_pred * lp
                (loss / args.batch).backward()
                for k, v in (("draft", ld), ("feat", lf), ("pred", lp), ("agree", ag)):
                    acc[k] += float(v)
                n += 1
            torch.nn.utils.clip_grad_norm_(params + dparams, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[{tag} step {step}/{total}] " + " ".join(f"{k} {v / max(n, 1):.3f}" for k, v in acc.items())
                      + f" {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0:
                evaluate(step)
    evaluate(step)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(args), "state": adapter.state_dict()}, out / "adapter.pt")
    if mapper is not None:
        torch.save({"args": vars(args), "layers": sel, "state": mapper.state_dict()}, out / "mapper.pt")
    if args.train_drafter:
        draft.to(torch.bfloat16).save_pretrained(out / "drafter")
    (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    print(f"[{tag}] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
