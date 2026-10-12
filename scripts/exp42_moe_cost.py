"""Experiment 42: is DFlash's 16-token block wrong for MoE targets? (verification cost vs draft length)

Block drafters assume verification is nearly free per extra token (dense model, batch 1, memory-bound).
For an MoE target every verified token routes to its own experts, so the union of experts that must
be loaded grows with the number of tokens verified (Limits of Speculation in MoE, 2609.22156).

Part 1 (GPU): verify-pass time for n = 1 .. 16 query tokens after a cached prefix, and the mean number of
distinct experts activated per layer, for an MoE target and for the dense Qwen3-8B (reference).
Part 2 (CPU, if --blocks): with DFlash acceptance data (exp41 --save-blocks: per block, which of the 15
drafted positions match the target, and the drafter's top-1 probability), throughput of
  fixed L        verify the anchor + the first L drafted tokens (DFlash: L = 15)
  confidence c   verify drafted tokens while the drafter's top-1 prob >= c (online, per block)
  oracle         verify exactly the accepted prefix (upper bound)
under each target's measured cost curve; drafter time = --t-draft-frac x the 1-token verify time.

  python scripts/exp42_moe_cost.py --models Qwen/Qwen3-30B-A3B Qwen/Qwen3-8B --data ... --blocks blocks.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.utils import RESULTS, print_table, save_json  # noqa: E402

NS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]


def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def router_hooks(model):
    """Capture the selected experts of every MoE router (module '...mlp.gate'): indices [tokens, top_k]."""
    store, hooks = [], []
    if not getattr(model.config, "num_experts", None):
        return store, hooks
    k = model.config.num_experts_per_tok

    def hook(m, i, o):
        store.append(o[2].detach() if isinstance(o, tuple) else o.detach().topk(k, -1).indices)

    for name, mod in model.named_modules():
        if name.endswith("mlp.gate"):
            hooks.append(mod.register_forward_hook(hook))
    return store, hooks


@torch.inference_mode()
def cost_curve(model_id, seqs, prefix_len, reps):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation="sdpa").eval()
    dev = model.get_input_embeddings().weight.device
    store, hooks = router_hooks(model)
    times = {n: [] for n in NS}
    experts = {n: [] for n in NS}
    for ids in seqs:
        ids = ids[: prefix_len + max(NS)].long().to(dev)
        out = model(ids[None, :prefix_len], use_cache=True)
        full = out.past_key_values
        for n in NS:
            q = ids[None, prefix_len : prefix_len + n]
            for r in range(reps + 1):
                cache = DynamicCache(ddp_cache_data=[(l.keys, l.values) for l in full.layers])
                store.clear()
                sync()
                t0 = time.perf_counter()
                model(q, past_key_values=cache, use_cache=True)
                sync()
                dt = time.perf_counter() - t0
                if r > 0:                                    # first rep = warm-up
                    times[n].append(dt * 1000)
                if r == reps and store:
                    act = [len(torch.unique(ix)) for ix in store]
                    experts[n].append(sum(act) / len(act))
    for h in hooks:
        h.remove()
    rows = []
    t1 = statistics.median(times[1])
    for n in NS:
        t = statistics.median(times[n])
        rows.append({"model": model_id.split("/")[-1], "tokens": n, "verify_ms": t, "rel_to_1": t / t1,
                     "experts_per_layer": statistics.mean(experts[n]) if experts[n] else None})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def interp(curve, n):
    xs = sorted(curve)
    if n <= xs[0]:
        return curve[xs[0]]
    for a, b in zip(xs, xs[1:]):
        if a <= n <= b:
            return curve[a] + (curve[b] - curve[a]) * (n - a) / (b - a)
    return curve[xs[-1]] * n / xs[-1]


def policies(blocks, curve, t_draft):
    match, dprob = blocks["match"].bool(), blocks["dprob"].float()
    nb, Lmax = match.shape
    A = match.long().cumprod(1).sum(1)                      # accepted drafted tokens per block (exact)
    res = []

    def run(name, L):                                       # L: tensor [nb] of drafted tokens verified
        tok = (torch.minimum(A, L) + 1).float()
        tim = torch.tensor([t_draft + interp(curve, int(l) + 1) for l in L])
        res.append({"policy": name, "mean_verified": float(L.float().mean()), "tau": float(tok.mean()),
                    "tok_per_ms": float(tok.sum() / tim.sum())})

    for L in (1, 2, 3, 4, 6, 8, 10, 12, 15):
        run(f"fixed L={L}", torch.full((nb,), L))
    for c in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        run(f"confidence>={c}", (dprob >= c).long().cumprod(1).sum(1).clamp(min=1))
    run("oracle (L = accepted)", A.clamp(min=1))
    base = next(r for r in res if r["policy"] == f"fixed L={Lmax}")["tok_per_ms"]
    for r in res:
        r["vs_dflash"] = r["tok_per_ms"] / base
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-8B"])
    ap.add_argument("--data", default="")
    ap.add_argument("--n-seqs", type=int, default=12)
    ap.add_argument("--prefix-len", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--blocks", default="", help="exp41 --save-blocks file")
    ap.add_argument("--curves", default="", help="reuse a saved cost-curve json instead of measuring")
    ap.add_argument("--t-draft-frac", type=float, default=0.2, help="drafter time / 1-token verify time")
    ap.add_argument("--bytes-model", action="store_true",
                    help="replace measured MoE times by weights read per verify (memory-bound kernels): "
                         "non-expert params + layers x measured experts/layer x expert params")
    ap.add_argument("--out", default="exp42_moe_cost")
    args = ap.parse_args()

    if args.curves:
        rows = json.loads(Path(args.curves).read_text())["cost"]
    else:
        data = []
        for f in sorted(Path(args.data).glob("shard*.pt")):
            data += torch.load(f)
            if len(data) > 20 * args.n_seqs:
                break
        data = [d["ids"] for d in data if len(d["ids"]) >= args.prefix_len + max(NS)]
        random.Random(0).shuffle(data)
        seqs = data[: args.n_seqs]
        rows = []
        for m in args.models:
            rows += cost_curve(m, seqs, args.prefix_len, args.reps)
            print(f"[{m}] measured", flush=True)
    print_table(rows, ["model", "tokens", "verify_ms", "rel_to_1", "experts_per_layer"],
                "verify-pass time after a cached prefix vs number of query tokens (median over sequences x reps)")
    out = {"args": vars(args), "cost": rows}

    if args.bytes_model:
        from transformers import AutoConfig
        brows = []
        for m in args.models:
            name = m.split("/")[-1]
            mr = [r for r in rows if r["model"] == name]
            if not mr or mr[0]["experts_per_layer"] is None:
                continue
            c = AutoConfig.from_pretrained(m)
            H, L, hd = c.hidden_size, c.num_hidden_layers, c.head_dim
            attn = H * hd * (2 * c.num_attention_heads + 2 * c.num_key_value_heads)
            non_expert = L * (attn + H * c.num_experts) + c.vocab_size * H          # + LM head
            expert = 3 * H * c.moe_intermediate_size
            for r in mr:
                gp = (non_expert + L * r["experts_per_layer"] * expert) / 1e9
                brows.append({**r, "verify_ms": gp, "rel_to_1": None, "params_read_B": gp})
            base = brows[-len(mr)]["params_read_B"]
            for r in brows[-len(mr):]:
                r["rel_to_1"] = r["params_read_B"] / base
        print_table(brows, ["model", "tokens", "params_read_B", "rel_to_1", "experts_per_layer"],
                    "memory-bound cost model: billions of weights read per verify pass (time units below)")
        rows = [r for r in rows if r["model"] not in {b["model"] for b in brows}] + brows
        out["bytes_cost"] = brows

    if args.blocks:
        blocks = torch.load(args.blocks)
        for model in dict.fromkeys(r["model"] for r in rows):
            curve = {r["tokens"]: r["verify_ms"] for r in rows if r["model"] == model}
            res = policies(blocks, curve, args.t_draft_frac * curve[1])
            print_table(res, ["policy", "mean_verified", "tau", "tok_per_ms", "vs_dflash"],
                        f"{model}: throughput with DFlash-8B acceptance data ({len(blocks['match'])} blocks), "
                        f"drafter = {args.t_draft_frac} x 1-token verify")
            out[f"policies_{model}"] = res
    save_json(args.out, out)


if __name__ == "__main__":
    main()
