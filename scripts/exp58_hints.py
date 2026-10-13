"""Experiment 58: the copy's features as gated hints at the restart's MASK positions (hspec/hints.py).

Offline, on the exp55 teacher sequences (the target's greedy text; round-1 drafts = the drafter's top-1 with real
features, stored in teacher.pt). Per anchor s, as exp56 (kv=target): copy over [x_s, y] on the target's prefix KV ->
copy features for s .. s+15 and copy top-1. Restarts:
  pc     at the copy's first disagreement j1: anchor a = s+1+j1 with the copy's token
  c1,c2  at the two most likely stop positions of the drafter: anchor a = s+i with the drafted token
Context features < a as in exp27/exp56; NEW: the copy's features at positions >= a enter as hints (HintMLP).
Train: the HintMLP only (drafter frozen), cross-entropy of every restart block whose prefix is correct against the
target's tokens, position weights gamma^(k-1). Evaluate: tau for block1 / +pc / +c2 / +pc+c2 with plain restarts and
with hints, on held-out sequences, and the accepted length of the restart part alone.

  python scripts/exp58_hints.py --copy bnb4:Qwen/Qwen3-8B --teacher .../teacher.pt --out .../hints_bnb4
  python scripts/exp58_hints.py --copy bridge:.../bridge_mlp ...       (bridged copy shares the target's weights)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import matched  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from hspec import bridges as B  # noqa: E402
from hspec.hints import HintMLP, hinted_logits  # noqa: E402
from hspec.models import load_draft, load_mid, load_target  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

POLICIES = {"block1": [], "pc": ["pc"], "c2": ["c1", "c2"], "pc+c2": ["pc", "c1", "c2"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", required=True, help="load_mid spec, or bridge:<dir> (attached to the target itself)")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=0, help="0: all")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--out", default="")
    ap.add_argument("--save", default="", help="results json name")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tdata = torch.load(args.teacher)
    train, val = tdata["train"][: args.n_train], tdata["val"]
    if args.n_val:
        val = val[: args.n_val]
    target = load_target(args.target, dev)
    target.requires_grad_(False)
    draft = load_draft(args.draft, dev)
    draft.requires_grad_(False)
    bridged = args.copy.startswith("bridge:")
    if bridged:
        B.load_bridged(target, args.copy.split(":", 1)[1])
        copy = target
    else:
        copy = load_mid(args.copy, dev)
        copy.requires_grad_(False)
    bs, lids, D = draft.block_size, list(draft.target_layer_ids), draft.block_size - 1
    hint = HintMLP(target.config.hidden_size, bs, args.width).to(dev)
    w = args.gamma ** torch.arange(D, device=dev, dtype=torch.float32)
    print(f"[hints] copy {args.copy}: HintMLP {sum(p.numel() for p in hint.parameters()) / 1e6:.1f}M params, "
          f"{len(train)} train / {len(val)} val sequences", flush=True)

    @torch.no_grad()
    def prepare(d):
        """Per anchor: everything the restarts need (no grad: target, copy, round-1 drafts)."""
        x = d["ids"].long().to(dev)
        B.BRIDGE["on"] = False
        tout = target(x[None], output_hidden_states=True, logits_to_keep=1, use_cache=True)
        B.BRIDGE["on"] = True
        feats = extract_context_feature(tout.hidden_states, lids)
        tcache = tout.past_key_values
        del tout
        items = []
        for k, s in enumerate(d["anchors"]):
            y = d["t_idx"][k, :, 0].long().to(dev)
            conf = d["t_lp"][k, :, 0].float().exp().tolist()
            z = torch.cat([x[: s + 1], y])
            cache = DynamicCache(ddp_cache_data=[(l.keys[:, :, :s], l.values[:, :, :s]) for l in tcache.layers])
            po = copy(z[None, s:], past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=bs)
            pz = extract_context_feature(po.hidden_states, lids).to(feats.dtype)          # positions s .. s+15
            ctop = po.logits[0].argmax(-1)                                                 # for s+1 .. s+16
            del po, cache
            full = torch.cat([feats[:, :s], pz], 1)
            truth = x[s + 1 : s + 2 * bs].tolist()
            yl = y.tolist()
            post, run = {}, 1.0
            for i in range(1, D + 1):
                run *= conf[i - 1]
                post[i] = run * (1.0 - conf[i]) if i < D else run
            cpos = sorted(post, key=lambda i: -post[i])[:2]
            dis = [j for j in range(D) if int(ctop[j]) != yl[j]]
            blocks = {}                     # name -> (ids, anchor, chain prefix)
            if dis:
                j1 = dis[0]
                zc = torch.cat([z[: s + 1 + j1], ctop[j1 : j1 + 1]])
                blocks["pc"] = (zc, s + 1 + j1, yl[:j1] + [int(ctop[j1])])
            for r, i in enumerate(cpos):
                blocks[f"c{r + 1}"] = (z, s + i, yl[:i])
            items.append({"s": s, "x": x, "y": yl, "full": full, "pz": pz, "seen": z[s : s + bs], "truth": truth,
                          "blocks": blocks})
        return items

    def block_logits(it, names, use_hint):
        """Logits per block name; pc and the conf restarts differ in the anchor token, so one call per ids."""
        out = {}
        groups = defaultdict(list)
        for n in names:
            groups[id(it["blocks"][n][0])].append(n)
        for ns in groups.values():
            ids = it["blocks"][ns[0]][0]
            L = hinted_logits(draft, target, it["full"], ids, [it["blocks"][n][1] for n in ns], bs,
                              hint if use_hint else None, it["pz"], it["seen"], it["s"] + bs - 1)
            out.update(zip(ns, L))
        return out

    def valid(it, n):
        pre = it["blocks"][n][2]
        return pre == it["truth"][: len(pre)] and len(it["truth"]) > len(pre)

    def evaluate(step):
        hint.eval()
        rec = {m: defaultdict(list) for m in ("plain", "hint")}
        part = {m: defaultdict(list) for m in ("plain", "hint")}
        with torch.no_grad():
            for d in val:
                for it in prepare(d):
                    for mode in ("plain", "hint"):
                        L = block_logits(it, list(it["blocks"]), mode == "hint")
                        chains = {n: it["blocks"][n][2] + L[n].argmax(-1).tolist() for n in L}
                        for pol, parts in POLICIES.items():
                            cs = [it["y"]] + [chains[n] for n in parts if n in chains]
                            rec[mode][pol].append(max(matched(c, it["truth"]) for c in cs) + 1)
                        for n in chains:
                            if valid(it, n):
                                pre = len(it["blocks"][n][2])
                                part[mode][n[:2] if n == "pc" else "conf"].append(
                                    matched(chains[n][pre:], it["truth"][pre:]))
        hint.train()
        rows = []
        for pol in POLICIES:
            rows.append({"policy": pol, "tau_plain": mean(rec["plain"][pol]), "tau_hint": mean(rec["hint"][pol])})
        for n in ("pc", "conf"):
            rows.append({"policy": f"restart part, {n} (valid prefix)", "tau_plain": mean(part["plain"][n]),
                         "tau_hint": mean(part["hint"][n])})
        print_table(rows, list(rows[0]), f"[eval step {step}] copy {args.copy}: tau with plain restarts vs with copy "
                                         f"hints (held-out); restart part = tokens accepted after the anchor")
        return rows

    rows0 = evaluate(0)
    opt = torch.optim.AdamW(hint.parameters(), lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    t0, step, nb = time.time(), 0, 0
    rng = random.Random(1)
    run_loss = []
    for _ in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for j in order:
            step += 1
            losses = []
            for it in prepare(train[j]):
                names = [n for n in it["blocks"] if valid(it, n)]
                if not names:
                    continue
                L = block_logits(it, names, True)
                for n in names:
                    a = it["blocks"][n][1]
                    lab = it["x"][a + 1 : a + bs]
                    k = len(lab)
                    ce = F.cross_entropy(L[n][:k].float(), lab, reduction="none")
                    losses.append((ce * w[:k]).sum() / w[:k].sum())
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            nb += len(losses)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hint.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss.append(float(loss))
            if step % 50 == 0:
                print(f"[train {step}/{total}] loss {mean(run_loss[-50:]):.4f} ({nb} blocks) "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 and step < total:
                evaluate(step)
    rows = evaluate(total)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        torch.save({"state": hint.state_dict(), "width": args.width, "bs": bs, "copy": args.copy},
                   Path(args.out) / "hint.pt")
        print(f"[hints] saved {args.out}", flush=True)
    if args.save:
        save_json(args.save, {"args": vars(args), "before": rows0, "after": rows})


if __name__ == "__main__":
    main()
