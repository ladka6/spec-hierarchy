"""Experiment 29: k refresh iterations before one target verify (offline).

Round at anchor s (real target features up to s):
  iteration 0   DFlash drafts block 1: chain C = y_1 .. y_15
  iteration i   the cheap copy runs over x_s + C: its first disagreement j with C and its token there
                (the correction), plus its features for positions s .. s+j; DFlash redrafts after
                the correction from those features: C <- C[:j] + [correction] + 15 new tokens
  verify        the target checks once:
                  chain  only the final chain C (linear verification, no tree)
                  tree   every chain produced in the round, merged into a prefix tree
Accepted = longest prefix matching the target trajectory, + 1 bonus. Lossless either way.

Also reports how often the copy's next token equals the target's (per-token agreement on the
trajectory), which caps how long a chain the iterations can build.

Cost model at the end: round = (k+1) drafter passes + k copy passes + one verify of the chain / tree,
with measured stage costs (vLLM: drafter 2.6 ms, copy 3-6 ms, verify table; our HF + CUDA graphs:
drafter 3.7 ms, copy 15.5-17.7 ms, verify 22.0-23.6 ms), against plain DFlash (one drafter pass +
verify of 16).

  python scripts/exp29_iterate.py --copy bnb4:Qwen/Qwen3-8B --ks 1 2 3 4
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import _make_cache, extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched, tree_nodes  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import crop, two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

VLLM_V = [(1, 12.7), (16, 13.3), (32, 14.1), (64, 14.2), (128, 15.8), (256, 19.0)]
HF_V = [(1, 20.8), (16, 22.0), (32, 22.8), (48, 23.3), (64, 23.6), (128, 25.5), (256, 29.0)]


def interp(tab, q):
    for (q0, t0), (q1, t1) in zip(tab, tab[1:]):
        if q <= q1:
            return t0 + (t1 - t0) * (q - q0) / (q1 - q0)
    return tab[-1][1] * q / tab[-1][0]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--small", default=None,
                    help="estimated states instead of a copy: small same-family model + mapper (hspec/smallmap.py)")
    ap.add_argument("--mapper", default=None, help="trained mapper.pt (train_shallow --small); else a ridge fit")
    ap.add_argument("--data", default=None, help="training sequences for the ridge fit (no --mapper)")
    ap.add_argument("--ridge-seqs", type=int, default=400)
    ap.add_argument("--n-sel", type=int, default=6)
    ap.add_argument("--corr", default="model",
                    help="token inserted at the corrector's first disagreement j: model (corrector's argmax) | "
                         "oracle (the target's true token: diagnostic) | none (no insert: restart from the draft's "
                         "own token at j-1) | conf:<p> (corrector's token only if its prob >= p, else none)")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp29_iterate")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    log("target loaded")
    draft = load_draft(args.draft)
    log("drafter loaded")
    mapper = None
    if args.small:
        from hspec.smallmap import SmallMapper, ridge_fit, small_layers
        dev = next(target.parameters()).device
        copy = load_target(args.small)
        if args.mapper:
            ck = torch.load(args.mapper, map_location=dev)
            sel = ck["layers"]
            w = ck["state"]["lin.weight"]
            mapper = SmallMapper(w.shape[1], w.shape[0], ck["args"]["map_hidden"]).to(dev)
            mapper.load_state_dict(ck["state"])
        else:
            import random as _r
            from pathlib import Path as _P
            seqs = []
            for f in sorted(_P(args.data).glob("shard*.pt")):
                seqs += torch.load(f)
            seqs = [q for q in seqs if len(q["ids"]) <= 1024]
            _r.Random(0).shuffle(seqs)
            sel = small_layers(copy.config.num_hidden_layers, args.n_sel)
            W_r, b_r = ridge_fit(copy, target, list(draft.target_layer_ids), sel, seqs[48 : 48 + args.ridge_seqs],
                                 dev, log=log)
            mapper = SmallMapper(W_r.shape[0], W_r.shape[1]).to(dev)
            mapper.lin.weight.copy_(W_r.T)
            mapper.lin.bias.copy_(b_r)
        mapper.eval()
        log(f"small model {args.small} + {'trained mapper' if args.mapper else 'ridge map'} (layers {sel})")
    else:
        copy = load_mid(args.copy)
        log("copy loaded")
    stops = stop_ids(target, tok)
    bs = draft.block_size
    D = bs - 1
    K = max(args.ks)
    span = D + K * bs                                  # longest chain the round can build

    rec = defaultdict(list)                            # k -> [(tau_chain, tau_tree, nodes, chain_len)]
    traj = []                                          # per anchor: [(chain_len, accepted, full_agree, conf)] per iteration
    base, agree = [], []
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            if args.temp > 0:
                x = target.generate(ids, max_new_tokens=args.max_new, do_sample=True, temperature=args.temp,
                                    top_p=1.0, top_k=0, eos_token_id=stops)[0]
            else:
                x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            draft.target_layer_ids)
            # per-token agreement of the copy with the trajectory (generated part)
            cl = copy(x[None]).logits[0, n0 - 1 : -1].argmax(-1)
            agree.append(float((cl == x[n0:]).float().mean()))
            # copy KV cache over the true text, extended anchor by anchor (iterations only add new tokens)
            ccache = _make_cache(copy.config)
            c_len = 0
            for s in range(n0, len(x) - span - 2, args.stride):
                if c_len < s:
                    copy(x[None, c_len:s], position_ids=torch.arange(c_len, s, device=x.device)[None],
                         past_key_values=ccache, use_cache=True, logits_to_keep=1)
                    c_len = s
                truth = x[s + 1 : s + 1 + span].tolist()
                lg = draft_logits_lagged(draft, target, feats, x, s, 0, bs)
                C = lg.argmax(-1).tolist()
                base.append(matched(C, truth) + 1)
                chains = [C]
                pc0 = torch.softmax(lg.float(), -1).max(-1).values
                conf0 = float(pc0.mean())
                # (chain_len, accepted, full_agree, mean_conf, | extra features for learned stopping rules:)
                # j (first copy disagreement, -1 at iteration 0), min conf of the new block,
                # copy's prob. of its correction token, mean copy prob. over the chain, dataset
                states = [(len(C), matched(C, truth) + 1, False, conf0, -1, float(pc0.min()), float("nan"),
                           float("nan"), d)]
                for it in range(1, K + 1):
                    z = torch.cat([x[: s + 1], torch.tensor(C, device=x.device)])
                    new = z[s:]                                       # x_s + chain, positions s .. s+len(C)
                    co = copy(new[None], position_ids=torch.arange(s, s + len(new), device=x.device)[None],
                              past_key_values=ccache, use_cache=True, output_hidden_states=True,
                              logits_to_keep=len(new))
                    if mapper is not None:     # estimated target states from the small model
                        hc = torch.cat([co.hidden_states[i][0] for i in sel], -1).float()
                        pz = mapper(hc)[None].to(feats.dtype)
                    else:
                        pz = extract_context_feature(co.hidden_states, draft.target_layer_ids).to(feats.dtype)
                    cpr = torch.softmax(co.logits[0].float(), -1).max(-1).values
                    cmean = float(cpr[:-1].mean())                   # over the chain being checked
                    pred = co.logits[0].argmax(-1).tolist()          # predictions for s+1 .. s+len(C)+1
                    del co
                    crop(ccache, s)
                    j = next((i for i in range(len(C)) if pred[i] != C[i]), len(C))
                    full = j == len(C)
                    mode = args.corr
                    if mode.startswith("conf:"):
                        mode = "model" if float(cpr[j]) >= float(mode[5:]) else "none"
                    if mode == "none" and j == 0:
                        mode = "model"                   # nothing to restart from: fall back to the corrector
                    if mode == "none":                   # keep C[:j], re-anchor on the draft's own C[j-1]
                        ctx = torch.cat([feats[:, :s], pz[:, :j]], 1)
                        L = blocks_logits(draft, target, ctx, z[: s + j + 1], [(s + j, 0)], bs)
                        C = (C[:j] + L[0].argmax(-1).tolist())[:span]
                    else:
                        tok_j = truth[j] if mode == "oracle" and j < len(truth) else pred[j]
                        ctx = torch.cat([feats[:, :s], pz[:, : j + 1]], 1)
                        zc = torch.cat([z[: s + j + 1], torch.tensor([tok_j], device=x.device)])
                        L = blocks_logits(draft, target, ctx, zc, [(s + j + 1, 0)], bs)
                        C = (C[:j] + [tok_j] + L[0].argmax(-1).tolist())[:span]
                    chains.append(C)
                    pl = torch.softmax(L[0].float(), -1).max(-1).values
                    states.append((len(C), matched(C, truth) + 1, full, float(pl.mean()), j, float(pl.min()),
                                   float(cpr[j]), cmean, d))
                    if it in args.ks:
                        rec[it].append((matched(C, truth) + 1, max(matched(c, truth) for c in chains) + 1,
                                        tree_nodes(chains), len(C)))
                traj.append(states)
            del ccache
            log(f"[{d}] {len(base)} anchors")

    tau0 = mean(base)
    rows = [{"k": 0, "tau_chain": tau0, "tau_tree": tau0, "chain_len": D, "tree_nodes": D}]
    for k in sorted(rec):
        v = rec[k]
        rows.append({"k": k, "tau_chain": mean(r[0] for r in v), "tau_tree": mean(r[1] for r in v),
                     "chain_len": mean(r[3] for r in v), "tree_nodes": mean(r[2] for r in v)})
    print(f"copy/target per-token agreement on the trajectory: {mean(agree):.4f}")
    print_table(rows, ["k", "tau_chain", "tau_tree", "chain_len", "tree_nodes"],
                f"k refresh iterations before one verify (T={args.temp}); k=0 = plain DFlash")

    # cost model
    costs = {"vLLM": (2.6, [3.0, 4.5, 6.0], VLLM_V), "HF+graphs": (3.7, [15.5, 17.7], HF_V)}
    crow = []
    for eng, (dd, ps, vt) in costs.items():
        r0 = tau0 / (dd + interp(vt, 16))
        for p in ps:
            for row in rows[1:]:
                k = row["k"]
                for mode in ("chain", "tree"):
                    q = (row["chain_len"] if mode == "chain" else row["tree_nodes"]) + 1
                    t = (k + 1) * dd + k * p + interp(vt, q)
                    crow.append({"engine": eng, "copy_ms": p, "k": k, "verify": mode,
                                 "round_ms": t, "x_vs_dflash": row[f"tau_{mode}"] / t / r0})
    print_table(crow, ["engine", "copy_ms", "k", "verify", "round_ms", "x_vs_dflash"],
                "estimated speedup over plain DFlash (measured stage costs)")
    # stopping rules: decide after each iteration whether to stop (iteration 0 = block 1 only, never stop
    # before 1 iteration). Cost per round: (k+1) drafter + k copy passes (each over ~16 new tokens with a
    # cached prefix) + one chain verify.
    def stop_at(st, rule):
        kmax = len(st) - 1
        for k in range(1, kmax + 1):
            ln, _, full, conf = st[k][:4]
            if rule[0] == "fixed" and k >= rule[1]:
                return k
            if rule[0] == "agree" and (full or k >= rule[1]):
                return k
            if rule[0] == "len" and (ln >= rule[1] or k >= rule[2]):
                return k
            if rule[0] == "conf" and (conf < rule[1] or k >= rule[2]):
                return k
            if rule[0] == "agree+len" and (full or ln >= rule[1] or k >= rule[2]):
                return k
        return kmax

    rules = [("fixed", k) for k in range(1, K + 1)] + [("agree", K)] + \
        [("len", L, K) for L in (32, 40, 48, 64)] + [("conf", c, K) for c in (0.5, 0.6, 0.7, 0.8)] + \
        [("agree+len", L, K) for L in (40, 48)]
    srows = []
    for eng, (dd, ps, vt) in costs.items():
        t_plain = dd + interp(vt, 16)
        for p in ps:
            for rule in rules:
                tok = tim = ks = 0.0
                for st in traj:
                    k = stop_at(st, rule)
                    tok += st[k][1]
                    ks += k
                    tim += (k + 1) * dd + k * p + interp(vt, st[k][0] + 1)
                plain = sum(st[0][1] for st in traj) / (len(traj) * t_plain)
                srows.append({"engine": eng, "copy_ms": p, "rule": "/".join(str(r) for r in rule),
                              "mean_k": ks / len(traj), "tau": tok / len(traj),
                              "x_vs_dflash": (tok / tim) / plain})
    best = {}
    for r in srows:
        key = (r["engine"], r["copy_ms"])
        if key not in best or r["x_vs_dflash"] > best[key]["x_vs_dflash"]:
            best[key] = r
    print_table(srows, ["engine", "copy_ms", "rule", "mean_k", "tau", "x_vs_dflash"],
                "stopping rules (fixed k / stop on full copy agreement / chain length budget / low drafter "
                "confidence)")
    print_table(list(best.values()), ["engine", "copy_ms", "rule", "mean_k", "tau", "x_vs_dflash"],
                "best rule per engine and copy cost")
    save_json(args.out, {"args": vars(args), "rows": rows, "costs": crow, "agree": mean(agree), "rules": srows,
                         "traj": traj})


if __name__ == "__main__":
    main()
