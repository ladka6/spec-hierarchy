"""Experiment 82: which parts of the target's computation do the in-flight states need? (no training)

Chained round as in exp80 / exp81: block 1 at anchor s matches n tokens (oracle cut); block 2 at anchor s+n needs
features for the in-flight positions s .. s+n-1. Here those features come from the target itself, run on the
in-flight tokens only, every layer attending to its REAL KV cache over the verified context, but with parts of the
computation removed (a removed sub-block = identity on the residual stream). DFlash is the original, frozen; its 5
tap features are taken from the ablated stack (outputs of layers 1, 9, 17, 25, 33).

Variants (layers 0 .. 33, the last one DFlash reads):
  full             every attention + every MLP (sanity check: must equal real)
  taps             only layers 1, 9, 17, 25, 33 (= exp81 at initialisation)
  attn_all         every attention, MLP only in the tap layers
  mlp_all          every MLP, attention only in the tap layers
  no_mlp           every attention, no MLP at all
  every2 / every4  every 2nd / 4th layer (+ the tap layers)
  taps_sens        tap layers + 4-7 + 20-23 (the most sensitive blocks for L25 / L33, exp78)
Reported per variant: tokens per verification over all rounds (chained n + m2 + 1 vs normal n + 1), block 2 first
token right, centred cosine at L25 / L33 against the real states, and the weights read per round (attention 42M /
MLP 151M parameters per Qwen3-8B layer).

  python scripts/exp82_ablate.py --cache seqs.pt --stops 300
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp81_tap_layers import cache_kv  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def variants(last, taps):
    all_l = set(range(last + 1))
    t = set(taps)
    return {
        "full": (all_l, all_l),
        "taps": (t, t),
        "attn_all": (all_l, t),
        "mlp_all": (t, all_l),
        "no_mlp": (all_l, set()),
        "every2": (set(range(0, last + 1, 2)) | t, set(range(0, last + 1, 2)) | t),
        "every4": (set(range(0, last + 1, 4)) | t, set(range(0, last + 1, 4)) | t),
        "taps_sens": (t | {4, 5, 6, 7, 20, 21, 22, 23}, t | {4, 5, 6, 7, 20, 21, 22, 23}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stops", type=int, default=300)
    ap.add_argument("--save", default="exp82_ablate")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    val = seqs[: max(int(len(seqs) * 0.2), 1)]
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H, K = target.config.hidden_size, len(lids)
    last = max(lids)
    layers = target.model.layers
    n_h, n_kv = target.config.num_attention_heads, target.config.num_key_value_heads
    emb = target.model.embed_tokens
    attn_p = sum(p.numel() for p in layers[0].self_attn.parameters())
    mlp_p = sum(p.numel() for p in layers[0].mlp.parameters())
    V = variants(last, lids)

    @torch.no_grad()
    def target_pass(x):
        out = target(x[None], output_hidden_states=True, use_cache=True, logits_to_keep=1)
        f = extract_context_feature(out.hidden_states, lids)[0]
        kv = {li: cache_kv(out.past_key_values, li) for li in range(last + 1)}
        return f, kv

    def attn_block(li, h, pos, kv, s):
        layer = layers[li]
        at = layer.self_attn
        n = h.shape[1]
        hd = at.head_dim
        x = layer.input_layernorm(h)
        q = at.q_norm(at.q_proj(x).view(1, n, n_h, hd)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(1, n, n_kv, hd)).transpose(1, 2)
        v = at.v_proj(x).view(1, n, n_kv, hd).transpose(1, 2)
        cos, sin = target.model.rotary_emb(v, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        ck, cv = kv[li]
        keys = torch.cat([ck[:, :, :s].to(k.dtype), k], 2).repeat_interleave(n_h // n_kv, 1)
        vals = torch.cat([cv[:, :, :s].to(v.dtype), v], 2).repeat_interleave(n_h // n_kv, 1)
        mask = torch.cat([torch.ones(n, s, dtype=torch.bool, device=dev),
                          torch.ones(n, n, dtype=torch.bool, device=dev).tril()], 1)[None, None]
        o = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask)
        return h + at.o_proj(o.transpose(1, 2).reshape(1, n, n_h * hd))

    def mlp_block(li, h):
        layer = layers[li]
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def ablated(kv, x, s, n, attn_set, mlp_set):
        """[n, K*H] tap features of the in-flight positions from the ablated target stack"""
        h = emb(x[s : s + n][None])
        pos = torch.arange(s, s + n, device=dev)[None]
        taps = {}
        for li in range(last + 1):
            if li in attn_set:
                h = attn_block(li, h, pos, kv, s)
            if li in mlp_set:
                h = mlp_block(li, h)
            if li in lids:
                taps[li] = h[0]
        return torch.cat([taps[li] for li in lids], -1)

    vr = random.Random(7)
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 8, len(ids) - 3 * bs)]
    vr.shuffle(cand)
    by = {}
    for i, s in cand[: args.stops]:
        by.setdefault(i, []).append(s)

    # mean of the real tap features (for centred cosines)
    acc, cnt = torch.zeros(K * H, device=dev, dtype=torch.float64), 0
    with torch.inference_mode():
        for i in list(by)[:20]:
            f, _ = target_pass(val[i][0].to(dev))
            acc += f[val[i][1] :].double().sum(0)
            cnt += f.shape[0] - val[i][1]
    mean = (acc / cnt).float()
    i25, i33 = lids.index(25), lids.index(33)

    recs = []
    with torch.inference_mode():
        for done, (i, anchors) in enumerate(by.items()):
            ids, lo = val[i]
            x = ids.to(dev)
            f, kv = target_pass(x)
            for s in anchors:
                y1 = blocks_logits(draft, target, f[None, :s], x, [(s, 0)], bs)[0].argmax(-1).tolist()
                n = matched(y1, x[s + 1 :].tolist())
                r = {"n": n}
                if n == 0:
                    r.update({k: 0 for k in ["real"] + list(V)})
                    recs.append(r)
                    continue
                a = s + n
                truth = x[a + 1 :].tolist()
                lg = blocks_logits(draft, target, f[None, :a], x, [(a, 0)], bs)[0]
                r["real"] = matched(lg.argmax(-1).tolist(), truth)
                real = f[s:a].float()
                for name, (aset, mset) in V.items():
                    e = ablated(kv, x, s, n, aset, mset)
                    fz = torch.cat([f[:s], e.to(f.dtype)], 0)[None]
                    lg = blocks_logits(draft, target, fz, x, [(a, 0)], bs)[0]
                    r[name] = matched(lg.argmax(-1).tolist(), truth)
                    ef = e.float()
                    for t, ti in (("25", i25), ("33", i33)):
                        sl = slice(ti * H, (ti + 1) * H)
                        r[f"{name}_cos{t}"] = float(F.cosine_similarity(ef[:, sl] - mean[sl], real[:, sl] - mean[sl],
                                                                        dim=-1).mean())
                recs.append(r)
            if done % 10 == 0:
                print(f"[run] {done}/{len(by)} sequences, {len(recs)} rounds, {time.time() - t0:.0f}s", flush=True)

    k = len(recs)
    pos_r = [r for r in recs if r["n"] > 0]
    normal = sum(r["n"] + 1 for r in recs) / k
    rows = [{"variant": "none (normal DFlash)", "attn layers": 0, "mlp layers": 0, "GB bf16": 0.0,
             "first token right": "-", "centred cos L25": "-", "centred cos L33": "-",
             "tokens / verification": normal, "vs normal": 1.0},
            {"variant": "real (verifier)", "attn layers": "-", "mlp layers": "-", "GB bf16": "-",
             "first token right": sum(r["real"] >= 1 for r in pos_r) / len(pos_r), "centred cos L25": 1.0,
             "centred cos L33": 1.0, "tokens / verification": sum(r["n"] + r["real"] + 1 for r in recs) / k,
             "vs normal": sum(r["n"] + r["real"] + 1 for r in recs) / k / normal}]
    for name, (aset, mset) in V.items():
        t = sum(r["n"] + r[name] + 1 for r in recs) / k
        rows.append({"variant": name, "attn layers": len(aset), "mlp layers": len(mset),
                     "GB bf16": 2 * (len(aset) * attn_p + len(mset) * mlp_p) / 1e9,
                     "first token right": sum(r[name] >= 1 for r in pos_r) / len(pos_r),
                     "centred cos L25": sum(r[f"{name}_cos25"] for r in pos_r) / len(pos_r),
                     "centred cos L33": sum(r[f"{name}_cos33"] for r in pos_r) / len(pos_r),
                     "tokens / verification": t, "vs normal": t / normal})
    print_table(rows, list(rows[0]), f"in-flight states from an ablated target (real cache), {k} rounds, oracle cut, "
                                     f"frozen DFlash; attention {attn_p / 1e6:.0f}M / MLP {mlp_p / 1e6:.0f}M per layer")
    save_json(args.save, {"args": vars(args), "rows": rows, "records": recs})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
