"""Experiment 45: how much MoE verification does a block diffusion drafter waste on tokens that get rejected?

DFlash always verifies its full block (anchor + 15 drafted tokens). On an MoE target the verify pass loads
the union of the experts all 16 tokens route to. Only the anchor and the accepted prefix are needed; the
rejected tail is verified for nothing, and if incoherent drafts scatter across experts, it is also the
expensive part.

Phase 1 (Qwen3-8B + its DFlash drafter): on target-greedy sequences, at anchors every --stride tokens,
draft a block and verify it (which drafted tokens match the target), keep the drafter's top-1 prob.
Phase 2 (Qwen3-30B-A3B, same tokenizer): route [anchor + drafted block] after the cached prefix, and,
for comparison, [anchor + the TRUE next 15 tokens]; record the experts every token selects per layer.
Acceptance comes from the 8B pair (a proxy: the 30B has no DFlash drafter here); routing from the 30B.

Reported:
  1. per block: accepted length, experts per layer for the full block vs for the needed tokens (anchor +
     accepted prefix), and the share of expert loads / of all weights read that serve only rejected tokens
  2. new experts per layer brought by drafted position k, accepted vs rejected tokens, vs true tokens
  3. union growth along the block: drafted block vs true continuation (do drafts scatter experts?)
  4. new experts by drafter confidence

  python scripts/exp45_expert_waste.py --data .../hspec_lagft/data
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp42_moe_cost import router_hooks  # noqa: E402

from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

CONF_BINS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01]


def sliced(cache, s):
    return DynamicCache(ddp_cache_data=[(l.keys[:, :, :s], l.values[:, :, :s]) for l in cache.layers])


@torch.inference_mode()
def phase1(args, data):
    target = load_target(args.target8b)
    draft = load_draft(args.draft)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    rng = random.Random(1)
    blocks, seqs = [], []
    for d in data:
        x = d["ids"].long().cuda()
        T = len(x)
        out = target(x[None], output_hidden_states=True, use_cache=True)
        feats = extract_context_feature(out.hidden_states, lids)
        full_logits, cache = out.logits[0], out.past_key_values
        lo, hi = max(d["n_prompt"], 1), T - bs
        if hi <= lo:
            continue
        seqs.append(x.cpu())
        for s in sorted(rng.sample(range(lo, hi), min(args.anchors, hi - lo))):
            dl = draft_logits_lagged(draft, target, feats, x, s, 0, bs).float()
            dtok = dl.argmax(-1)
            vl = target(dtok[None], past_key_values=sliced(cache, s + 1), use_cache=True).logits[0]
            tgt = torch.cat([full_logits[s : s + 1].argmax(-1), vl[:-1].argmax(-1)])      # target's choice at s+1..
            blocks.append({"seq": len(seqs) - 1, "s": s, "draft": dtok.cpu(),
                           "match": (dtok == tgt).cpu(), "dprob": torch.softmax(dl, -1).max(-1).values.cpu()})
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()
    return blocks, seqs


@torch.inference_mode()
def phase2(args, blocks, seqs):
    model = AutoModelForCausalLM.from_pretrained(args.moe, dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation="sdpa").eval()
    dev = model.get_input_embeddings().weight.device
    store, hooks = router_hooks(model)
    res = [None] * len(blocks)
    for si, x in enumerate(seqs):
        mine = [i for i, b in enumerate(blocks) if b["seq"] == si]
        if not mine:
            continue
        ids = x.to(dev)
        cache = model(ids[None], use_cache=True).past_key_values      # whole sequence once, sliced per anchor
        for i in mine:
            b, s = blocks[i], blocks[i]["s"]
            rout = {}
            for name, toks in (("draft", torch.cat([ids[s : s + 1], b["draft"].to(dev)])),
                               ("true", ids[s : s + 1 + len(b["draft"])])):
                store.clear()
                model(toks[None], past_key_values=sliced(cache, s), use_cache=True)
                rout[name] = torch.stack([ix.cpu() for ix in store])  # [layers, 16, top_k]
            res[i] = rout
        del cache
    for h in hooks:
        h.remove()
    cfg = AutoConfig.from_pretrained(args.moe)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res, cfg


def union_size(ix):
    """ix [layers, n, k] -> mean over layers of the number of distinct experts."""
    if ix.shape[1] == 0:
        return 0.0
    return sum(len(torch.unique(ix[l])) for l in range(ix.shape[0])) / ix.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target8b", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--moe", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-seqs", type=int, default=150)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="exp45_expert_waste")
    ap.add_argument("--save", default="", help="save per-block routing (int16) + acceptance + confidence here")
    args = ap.parse_args()

    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
        if len(data) >= 4 * args.n_seqs:
            break
    data = [d for d in data if len(d["ids"]) <= args.max_len and len(d["ids"]) - d["n_prompt"] > 40]
    random.Random(0).shuffle(data)
    data = data[: args.n_seqs]

    blocks, seqs = phase1(args, data)
    print(f"phase 1: {len(blocks)} blocks, mean accepted {sum(int(b['match'].long().cumprod(0).sum()) for b in blocks) / len(blocks):.2f}",
          flush=True)
    routs, cfg = phase2(args, blocks, seqs)
    H, L = cfg.hidden_size, cfg.num_hidden_layers
    attn = H * cfg.head_dim * (2 * cfg.num_attention_heads + 2 * cfg.num_key_value_heads)
    non_expert = L * (attn + H * cfg.num_experts) + cfg.vocab_size * H
    expert = 3 * H * cfg.moe_intermediate_size

    rows, pos_acc, pos_rej, pos_true, conf_new = [], {}, {}, {}, {}
    grow_d = torch.zeros(16)
    grow_t = torch.zeros(16)
    tot = {"A": 0.0, "u_all": 0.0, "u_need": 0.0, "u_true": 0.0, "w_all": 0.0, "w_need": 0.0}
    for b, r in zip(blocks, routs):
        A = int(b["match"].long().cumprod(0).sum())                # accepted drafted tokens
        d, t = r["draft"], r["true"]                               # [L, 16, k]
        u_all, u_need = union_size(d), union_size(d[:, : A + 1])   # anchor + accepted prefix are needed
        tot["A"] += A
        tot["u_all"] += u_all
        tot["u_need"] += u_need
        tot["u_true"] += union_size(t)
        tot["w_all"] += non_expert + L * u_all * expert
        tot["w_need"] += non_expert + L * u_need * expert
        for n in range(1, 17):
            grow_d[n - 1] += union_size(d[:, :n])
            grow_t[n - 1] += union_size(t[:, :n])
        for k in range(1, 16):                                     # drafted token k sits at block index k
            new_d = union_size(d[:, : k + 1]) - union_size(d[:, :k])
            new_t = union_size(t[:, : k + 1]) - union_size(t[:, :k])
            (pos_acc if k <= A else pos_rej).setdefault(k, []).append(new_d)
            pos_true.setdefault(k, []).append(new_t)
            c = float(b["dprob"][k - 1])
            for lo, hi in zip(CONF_BINS, CONF_BINS[1:]):
                if lo <= c < hi:
                    conf_new.setdefault((lo, hi), []).append((new_d, k <= A))
    n = len(blocks)

    def m(xs):
        return sum(xs) / len(xs) if xs else None

    summary = {"blocks": n, "mean_accepted": tot["A"] / n, "experts_full_block": tot["u_all"] / n,
               "experts_needed": tot["u_need"] / n, "experts_true_block": tot["u_true"] / n,
               "expert_loads_wasted": 1 - tot["u_need"] / tot["u_all"],
               "weights_read_wasted": 1 - tot["w_need"] / tot["w_all"]}
    print_table([summary], list(summary), "MoE verification of DFlash blocks (16 tokens): experts per layer and waste "
                                          "(expert loads / weights read that serve only rejected tokens)")
    prow = [{"k": k, "new_if_accepted": m(pos_acc.get(k, [])), "n_acc": len(pos_acc.get(k, [])),
             "new_if_rejected": m(pos_rej.get(k, [])), "n_rej": len(pos_rej.get(k, [])),
             "new_true_text": m(pos_true[k])} for k in range(1, 16)]
    print_table(prow, list(prow[0]), "new experts per layer brought by drafted position k (accepted vs rejected), "
                                     "vs the true continuation at the same position")
    grow = [{"tokens": i + 1, "draft_block": float(grow_d[i] / n), "true_block": float(grow_t[i] / n)} for i in range(16)]
    print_table(grow, list(grow[0]), "experts per layer needed to verify the first n tokens: drafted vs true text")
    crow = [{"drafter_conf": f"{lo}-{min(hi, 1)}", "tokens": len(v), "new_experts": m([x for x, _ in v]),
             "accepted_share": m([float(a) for _, a in v])} for (lo, hi), v in sorted(conf_new.items())]
    print_table(crow, list(crow[0]), "new experts per layer and acceptance by drafter confidence")
    if args.save:
        torch.save({"A": torch.tensor([int(b["match"].long().cumprod(0).sum()) for b in blocks]),
                    "match": torch.stack([b["match"] for b in blocks]),
                    "dprob": torch.stack([b["dprob"] for b in blocks]),
                    "draft_rout": torch.stack([r["draft"] for r in routs]).to(torch.int16),   # [n, L, 16, k]
                    "true_rout": torch.stack([r["true"] for r in routs]).to(torch.int16),
                    "non_expert": non_expert, "expert": expert, "layers": L}, args.save)
        print(f"saved per-block routing to {args.save}", flush=True)
    save_json(args.out, {"args": vars(args), "summary": summary, "by_position": prow, "growth": grow,
                         "by_confidence": crow})


if __name__ == "__main__":
    main()
