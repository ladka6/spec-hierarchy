"""Experiment 56: one wide refresh step instead of several deep ones (offline, lossless).

exp29: iterating drafter -> copy -> redraft raises tau (11.3 at 1 loop .. 19.0 at 6, 4-bit copy, T=1) but every loop
adds a drafter and a copy pass, so the best rule reaches 1.28x over DFlash. At batch 1 width is nearly free:
verifying a 64-node tree costs ~ a 16-token block (14.2 vs 13.3 ms, vLLM) and restart blocks batch into one drafter
call. So: ONE copy pass, then many restarts in one batched drafter call, all verified as one tree.

Per anchor s: block 1 y (real features); one copy pass over [x_s, y] -> copy features and the copy's top-3 next
tokens at every position; its disagreements with y at positions j1 < j2 < j3.
Restart chains (each = kept prefix + corrected/kept token + 15 drafted tokens with copy features):
  pc       correction top-1 at j1                      (exp27's pc)
  w2, w3   corrections top-2 / top-3 at j1             (more than one guess for the token the copy doubts)
  d2, d3   top-1 corrections at j1, j2 (, j3)          (the copy's later doubts, in case j1 was its mistake)
  cK       restarts at the K most likely stop positions of the drafter (exp27's confK)
Policies = unions of these; tau = longest chain prefix matching the target trajectory + 1; nodes = tree size.
Projected speed (vLLM component costs, as exp29): round = drafter + copy + restart call + verify(nodes); DFlash =
drafter + verify(16).

  python scripts/exp56_wide.py --copy bnb4:Qwen/Qwen3-8B --temp 1.0
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched, tree_nodes  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

VTAB = [(1, 12.7), (16, 13.3), (32, 14.1), (64, 14.2), (128, 15.8), (256, 19.0)]   # vLLM verify ms (exp29)


def v(q):
    for (a, ta), (b, tb) in zip(VTAB, VTAB[1:]):
        if q <= b:
            return ta + (tb - ta) * (q - a) / (b - a)
    return VTAB[-1][1]


POLICIES = {
    "block1": [], "pc": ["pc"], "w2": ["pc", "w2"], "w3": ["pc", "w2", "w3"], "d2": ["pc", "d2"],
    "d3": ["pc", "d2", "d3"], "w2+d2": ["pc", "w2", "d2"], "pc+c2": ["pc", "c1", "c2"],
    "w2+c2": ["pc", "w2", "c1", "c2"], "w2+d2+c2": ["pc", "w2", "d2", "c1", "c2"],
    "w3+d3+c4": ["pc", "w2", "w3", "d2", "d3", "c1", "c2", "c3", "c4"],
}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", required=True, help="load_mid spec, e.g. bnb4:Qwen/Qwen3-8B or lora:<dir>")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--drafter-ms", type=float, default=2.6)
    ap.add_argument("--copy-ms", nargs="+", type=float, default=[1.5, 3.0, 4.5])
    ap.add_argument("--restart-ms", type=float, default=1.0, help="the batched restart drafter call")
    ap.add_argument("--kv", choices=["own", "target"], default="own",
                    help="own: the copy computes the whole prefix with its own weights; target: the copy attends to "
                         "the TARGET's KV cache for the committed prefix and only computes the drafted block itself")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp56_wide")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    copy = load_mid(args.copy)
    stops = stop_ids(target, tok)
    bs = draft.block_size
    D = bs - 1
    rec = defaultdict(list)
    agree = []
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            if args.temp > 0:
                x = target.generate(ids, max_new_tokens=args.max_new, do_sample=True, temperature=args.temp,
                                    top_p=1.0, top_k=0, eos_token_id=stops)[0]
            else:
                x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            tout = target(x[None], output_hidden_states=True, logits_to_keep=1, use_cache=True)
            feats = extract_context_feature(tout.hidden_states, draft.target_layer_ids)
            tcache = tout.past_key_values
            del tout

            def prefix_cache(n):
                return DynamicCache(ddp_cache_data=[(l.keys[:, :, :n], l.values[:, :, :n]) for l in tcache.layers])
            for s in range(n0, len(x) - 2 * bs - 1, args.stride):
                truth = x[s + 1 : s + 2 * bs].tolist()
                pr = torch.softmax(draft_logits_lagged(draft, target, feats, x, s, 0, bs).float(), -1)
                conf, y = pr.max(-1)
                y, conf = y.tolist(), conf.tolist()
                z = torch.cat([x[: s + 1], torch.tensor(y, device=x.device)])
                if args.kv == "own":
                    po = copy(z[None], output_hidden_states=True, logits_to_keep=bs)
                    pz = extract_context_feature(po.hidden_states, draft.target_layer_ids).to(feats.dtype)[:, s : s + bs]
                else:
                    po = copy(z[None, s:], past_key_values=prefix_cache(s), use_cache=True, output_hidden_states=True,
                              logits_to_keep=bs)
                    pz = extract_context_feature(po.hidden_states, draft.target_layer_ids).to(feats.dtype)
                top3 = po.logits[0].float().topk(3, -1).indices.tolist()          # predictions for s+1 .. s+16
                del po
                # the target's own next tokens on the same drafted inputs: how often the copy agrees with them
                tl = target(z[None, s:], past_key_values=prefix_cache(s), use_cache=True, logits_to_keep=bs).logits[0]
                agree.append(float((tl.argmax(-1).cpu() == torch.tensor([t[0] for t in top3])).float().mean()))
                full = torch.cat([feats[:, :s], pz], 1)
                dis = [k for k in range(D) if top3[k][0] != y[k]][:3]

                def restart(jj, t):
                    zc = torch.cat([z[: s + jj + 1], torch.tensor([t], device=z.device)])
                    L = blocks_logits(draft, target, full, zc, [(s + jj + 1, 0)], bs)
                    return y[:jj] + [t] + L[0].argmax(-1).tolist()

                ch = {}
                if dis:
                    j1 = dis[0]
                    ch["pc"] = restart(j1, top3[j1][0])
                    ch["w2"] = restart(j1, top3[j1][1])
                    ch["w3"] = restart(j1, top3[j1][2])
                    for n_, jj in zip(("d2", "d3"), dis[1:]):
                        ch[n_] = restart(jj, top3[jj][0])
                post, run = {}, 1.0
                for i in range(1, D + 1):
                    run *= conf[i - 1]
                    post[i] = run * (1.0 - conf[i]) if i < D else run
                cpos = sorted(post, key=lambda i: -post[i])[:4]
                mix = torch.cat([feats[:, :s], pz[:, :D]], 1)
                Lc = blocks_logits(draft, target, mix, z, [(s + i, 0) for i in cpos], bs)
                for r, i in enumerate(cpos):
                    ch[f"c{r + 1}"] = y[:i] + Lc[r].argmax(-1).tolist()
                for pol, parts in POLICIES.items():
                    chains = [y] + [ch[k] for k in parts if k in ch]
                    best = max(matched(c, truth) for c in chains)
                    rec[pol].append((best + 1, tree_nodes(chains), len(chains) - 1))
            print(f"[{d}] {len(rec['block1'])} anchors", flush=True)

    base_tau = mean(r[0] for r in rec["block1"])
    base_ms = args.drafter_ms + v(16)
    rows = []
    for pol in POLICIES:
        tau = mean(r[0] for r in rec[pol])
        nodes = mean(r[1] for r in rec[pol])
        row = {"policy": pol, "tau": tau, "nodes": nodes, "restarts": mean(r[2] for r in rec[pol])}
        for cm in args.copy_ms:
            ms = base_ms if pol == "block1" else args.drafter_ms + cm + args.restart_ms + v(nodes + 1)
            row[f"x@copy{cm:g}ms"] = (tau / ms) / (base_tau / base_ms)
        rows.append(row)
    print(f"copy-target next-token agreement on drafted blocks (kv={args.kv}): {mean(agree):.4f}", flush=True)
    print_table(rows, list(rows[0]), f"one wide refresh step, copy {args.copy}, kv={args.kv}, T={args.temp}: accepted per verify, "
                                     f"tree nodes, projected speed vs DFlash (vLLM costs, drafter {args.drafter_ms} ms, "
                                     f"restart call {args.restart_ms} ms)")
    save_json(args.out, {"args": vars(args), "rows": rows, "copy_target_agreement": mean(agree)})


if __name__ == "__main__":
    main()
