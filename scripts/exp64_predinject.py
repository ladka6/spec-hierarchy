"""Experiment 64: DSpine-style predecessor injection fed by the cheap copy (hspec/predinject.py), rough numbers.

Round: the DFlash drafter drafts block 1 from real features (teacher.pt top-1 = y); the copy (4-bit, on the target's
prefix KV) reads [x_s, y]; a SECOND drafter pass over the same block (same anchor, same real-feature context) gets,
after every layer, a message about the predecessor of each position from one of the sources in predinject.VARIANTS.
Every variant is trained in the same loop (its own zero-init injection modules + its own drafter LoRA) with
cross-entropy of the second pass on the target's tokens; the "none" variant (LoRA only) separates the fine-tune from
the information. Evaluated on held-out sequences: accepted length of the second-pass chain, of block 1, and of the
better of the two (a 2-chain tree), and top-1 accuracy by block position.

  python scripts/exp64_predinject.py --teacher .../teacher.pt --copy bnb4:Qwen/Qwen3-8B
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

from dflash.model import _output_head, extract_context_feature  # noqa: E402
from exp27_refresh import matched  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from hspec import predinject as P  # noqa: E402
from hspec.lagtrain import pack  # noqa: E402
from hspec.models import load_draft, load_mid, load_target  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--teacher", required=True, help="exp55 teacher.pt (sequences, anchors, drafter top-32)")
    ap.add_argument("--variants", nargs="+", default=list(P.VARIANTS))
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--inj-rank", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--save", default="exp64_predinject")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tdata = torch.load(args.teacher)
    train, val = tdata["train"][: args.n_train], tdata["val"]
    if args.n_val:
        val = val[: args.n_val]
    target = load_target(args.target, dev)
    target.requires_grad_(False)
    copy = load_mid(args.copy, dev)
    copy.requires_grad_(False)
    draft = load_draft(args.draft, dev)
    draft.requires_grad_(False)
    bs, lids, D = draft.block_size, list(draft.target_layer_ids), draft.block_size - 1
    H = target.config.hidden_size
    nL = len(draft.layers)
    assert nL == len(lids), "depth-aligned injection assumes one tap layer per drafter layer"
    emb = target.get_input_embeddings()
    lora = P.add_multilora(draft, args.variants, args.lora_rank)
    inj = {v: P.make_injectors(v, H, nL, args.inj_rank, dev) for v in args.variants}
    opts, scheds = {}, {}
    total = args.epochs * len(train)
    for v in args.variants:
        ps = lora[v] + (list(inj[v].parameters()) if inj[v] is not None else [])
        opts[v] = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
        scheds[v] = torch.optim.lr_scheduler.LambdaLR(
            opts[v], lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    w = args.gamma ** torch.arange(D, device=dev, dtype=torch.float32)
    POS = sorted({1, max(1, D // 4), D // 2, (3 * D) // 4, D})   # 1, 4, 8, 12, 15 for block 16
    print(f"[exp64] variants {args.variants}; {len(train)} train / {len(val)} val sequences; "
          f"LoRA {sum(p.numel() for p in lora[args.variants[0]]) / 1e6:.1f}M per variant", flush=True)

    @torch.no_grad()
    def prepare(d):
        x = d["ids"].long().to(dev)
        tout = target(x[None], output_hidden_states=True, logits_to_keep=1, use_cache=True)
        feats = extract_context_feature(tout.hidden_states, lids)
        tcache = tout.past_key_values
        del tout
        anchors = list(d["anchors"])
        y_all = d["t_idx"][:, :, 0].long().to(dev)                     # [A, 15] first-pass top-1
        tl, ti = d["t_lp"].to(dev).float(), d["t_idx"].long().to(dev)    # [A, 15, 32]
        tok_msgs = {k: [] for k in ("self", "copy_soft", "oracle")}
        feat_msgs = []
        for k, s in enumerate(anchors):
            y = y_all[k]
            z = torch.cat([x[s : s + 1], y])                              # positions s .. s+15
            cache = DynamicCache(ddp_cache_data=[(l.keys[:, :, :s], l.values[:, :, :s]) for l in tcache.layers])
            po = copy(z[None], past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=bs)
            pz = extract_context_feature(po.hidden_states, lids)[0]        # [16, 5H] positions s .. s+15
            lc = po.logits[0].float()                                      # predicts s+1 .. s+16
            del po, cache
            e_anchor = emb(x[s : s + 1]).float()                           # predecessor of row 1 = the anchor
            cv, ci = torch.log_softmax(lc[: D - 1], -1).topk(args.topk, -1)
            tok_msgs["self"].append(torch.cat([e_anchor, P.soft_embed(emb, tl[k, : D - 1], ti[k, : D - 1])]))
            tok_msgs["copy_soft"].append(torch.cat([e_anchor, P.soft_embed(emb, cv, ci)]))
            tok_msgs["oracle"].append(emb(x[s : s + D]).float())          # true tokens at s .. s+14
            feat_msgs.append(pz[:D].float().view(D, nL, H))                # copy features at s .. s+14
        noise, pos, mask, starts = pack(draft, target, feats, x, [(s, 0) for s in anchors], bs)
        rows = torch.cat([torch.arange(st, st + D, device=dev) for st in starts])
        tok = {k: torch.cat(v) for k, v in tok_msgs.items()}               # [A*15, H]
        fm = torch.cat(feat_msgs)                                           # [A*15, nL, H]
        truth = torch.stack([x[s + 1 : s + 1 + D] for s in anchors])       # [A, 15]
        return {"x": x, "feats": feats, "noise": noise, "pos": pos, "mask": mask, "starts": starts, "rows": rows,
                "tok": tok, "fm": fm, "truth": truth, "y": y_all}

    def msgs_for(v, it):
        if v == "none":
            return None
        if v in ("self", "copy_soft", "oracle"):
            return [it["tok"][v]] * nL
        if v == "copy_feat":
            return [it["fm"][:, l] for l in range(nL)]
        return [torch.cat([it["tok"]["copy_soft"], it["fm"][:, l]], -1) for l in range(nL)]   # copy_both

    def second_pass(v, it):
        P.ACTIVE["v"] = v
        hid = P.injected_forward(draft, it["feats"], it["noise"], it["pos"], it["mask"], it["rows"],
                                 msgs_for(v, it), inj[v])[0]
        P.ACTIVE["v"] = None
        rows = torch.stack([hid[st : st + D] for st in it["starts"]])
        return draft.compute_logits(rows, _output_head(target))         # [A, 15, V]

    def evaluate(step):
        for v in args.variants:
            if inj[v] is not None:
                inj[v].eval()
        rec = defaultdict(list)
        acc = defaultdict(lambda: torch.zeros(D))
        n_blocks = 0
        with torch.no_grad():
            for d in val:
                it = prepare(d)
                truth = it["truth"].tolist()
                y = it["y"].tolist()
                n_blocks += len(truth)
                for k in range(len(truth)):
                    rec["block1"].append(matched(y[k], truth[k]) + 1)
                acc["block1"] += (it["y"] == it["truth"]).float().sum(0).cpu()
                for v in args.variants:
                    pred = second_pass(v, it).argmax(-1)
                    acc[v] += (pred == it["truth"]).float().sum(0).cpu()
                    pl = pred.tolist()
                    for k in range(len(truth)):
                        t2 = matched(pl[k], truth[k]) + 1
                        rec[v].append(t2)
                        rec[v + "|tree2"].append(max(t2, matched(y[k], truth[k]) + 1))
        for v in args.variants:
            if inj[v] is not None:
                inj[v].train()
        rows = [{"variant": "block1 (first pass)", "tau": mean(rec["block1"]), "tau_tree2": float("nan"),
                 **{f"acc{p}": float(acc['block1'][p - 1] / n_blocks) for p in POS}}]
        for v in args.variants:
            rows.append({"variant": v, "tau": mean(rec[v]), "tau_tree2": mean(rec[v + "|tree2"]),
                         **{f"acc{p}": float(acc[v][p - 1] / n_blocks) for p in POS}})
        print_table(rows, list(rows[0]), f"[eval step {step}] second drafter pass with predecessor injection "
                                         f"(copy {args.copy}); tau = accepted length of the chain, tree2 = better of "
                                         f"block 1 and the second pass; acc = top-1 accuracy at block position")
        return rows

    rows0 = evaluate(0)
    t0, step, hist = time.time(), 0, defaultdict(list)
    rng = random.Random(1)
    for _ in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for j in order:
            step += 1
            it = prepare(train[j])
            for v in args.variants:
                L = second_pass(v, it).float()
                ce = F.cross_entropy(L.reshape(-1, L.shape[-1]), it["truth"].reshape(-1), reduction="none")
                loss = (ce.view(-1, D) * w).sum() / (w.sum() * ce.view(-1, D).shape[0])
                opts[v].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in opts[v].param_groups for p in g["params"]], 1.0)
                opts[v].step()
                scheds[v].step()
                hist[v].append(float(loss))
            if step % 50 == 0:
                print(f"[train {step}/{total}] " + " ".join(f"{v} {mean(hist[v][-50:]):.3f}" for v in args.variants)
                      + f"  {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 and step < total:
                evaluate(step)
    rows = evaluate(total)
    save_json(args.save, {"args": vars(args), "before": rows0, "after": rows})


if __name__ == "__main__":
    main()
