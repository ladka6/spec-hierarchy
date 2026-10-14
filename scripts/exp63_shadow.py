"""Experiment 63: shadow refresh, offline. Can the drafter + copy prepare the NEXT round while the target verifies?

Pipeline idea: one draft is always in flight at the target. During its verification the small models guess the
verification outcome (accepted length j and the target's token t at the stop) and prepare the next draft for the
likely outcomes, conditioned on the copy's features (the target's real features do not exist yet). On a hit the
prepared draft goes straight to the target; on a miss a plain DFlash block from the real features is the fallback.

Per anchor s (target trajectory known, copy on the target's prefix KV as exp56):
  in-flight chain   "block": the drafter's block from real features; "refreshed": copy-corrected restart of it
                    (methodology A). The copy runs over the in-flight chain once.
  outcome model     from the copy's own probabilities q over the chain c:
                      P(stop at j, token t) = prod_{i<j} q_i(c_i) * q_j(t),  t != c_j   (j = len(c): full accept)
                    -> hit@K: is the actual outcome (j*, t*) among the K most likely?
  prepared draft    for the ACTUAL outcome (what a hit delivers): anchor a = s+1+j* with token t*, context = real
                    features before s + copy features s .. a-1, copy KV for the in-flight positions. Depth d = number
                    of refresh loops (copy over the chain -> restart at its first disagreement, or extend at the end
                    when it agrees everywhere), up to --depth; chains may grow past 16 tokens (cap --max-len).
  baselines         the next round drafted AFTER verification from real features: plain block, and + one loop.
Event model (vLLM component costs as exp56) for policies (K outcome branches, depth d): two GPUs (shadow work in
parallel, round = max(verify, shadow)) and one GPU (round = verify + kappa * shadow, kappa = share not hidden).

  python scripts/exp63_shadow.py --copy bnb4:Qwen/Qwen3-8B --temp 0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp56_wide import v  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402

KS = [1, 2, 4, 8]


def sliced(cache, n):
    return DynamicCache(ddp_cache_data=[(l.keys[:, :, :n], l.values[:, :, :n]) for l in cache.layers])


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=63, help="cap on a prepared chain (verify cost is flat to ~64)")
    ap.add_argument("--drafter-ms", type=float, default=2.6)
    ap.add_argument("--copy-ms", type=float, default=3.0)
    ap.add_argument("--restart-ms", type=float, default=1.0, help="the batched restart drafter call (exp56)")
    ap.add_argument("--batch-extra", type=float, default=0.25, help="extra cost per additional branch in a batch")
    ap.add_argument("--kappa", nargs="+", type=float, default=[0.5, 0.77], help="one GPU: share of shadow work not hidden")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp63_shadow")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    copy = load_mid(args.copy)
    stops = stop_ids(target, tok)
    bs, lids = draft.block_size, list(draft.target_layer_ids)
    D = bs - 1
    recs = []

    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            if args.temp > 0:
                x = target.generate(ids, max_new_tokens=args.max_new, do_sample=True, temperature=args.temp,
                                    top_p=1.0, top_k=0, eos_token_id=stops)[0]
            else:
                x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            tout = target(x[None], output_hidden_states=True, logits_to_keep=1, use_cache=True)
            feats = extract_context_feature(tout.hidden_states, lids)
            tcache = tout.past_key_values
            del tout
            dt = feats.dtype

            def copy_run(cache, toks, start):
                """Copy over toks at positions start.. (cache holds positions < start). -> feats, logits, cache."""
                po = copy(toks[None], past_key_values=cache, use_cache=True, output_hidden_states=True,
                          logits_to_keep=len(toks))
                f = extract_context_feature(po.hidden_states, lids).to(dt)
                return f, po.logits[0].float(), po.past_key_values

            def draft_block(ctx, seq, a):
                """Drafter block (15 tokens) from anchor a of seq, context features ctx (positions < a used)."""
                return blocks_logits(draft, target, ctx, seq, [(a, 0)], bs)[0].argmax(-1).tolist()

            for s in range(n0, len(x) - 3 * bs - 1, args.stride):
                truth = x[s + 1 :].tolist()
                y = draft_logits_lagged(draft, target, feats, x, s, 0, bs).argmax(-1).tolist()
                rec = {"tau_block1": matched(y, truth) + 1}
                # in-flight chains: the plain block, and its copy-corrected restart (methodology A)
                zb = torch.cat([x[: s + 1], torch.tensor(y, device=x.device)])
                fb, lb, _ = copy_run(sliced(tcache, s), zb[s:], s)
                ctop = lb.argmax(-1).tolist()
                dis = [k for k in range(D) if ctop[k] != y[k]]
                if dis:
                    j1 = dis[0]
                    seq = torch.cat([zb[: s + 1 + j1], torch.tensor([ctop[j1]], device=x.device)])
                    full = torch.cat([feats[:, :s], fb], 1)
                    chain_r = y[:j1] + [ctop[j1]] + draft_block(full, seq, s + 1 + j1)
                else:
                    chain_r = y
                for kind, chain in (("block", y), ("refreshed", chain_r)):
                    c = torch.tensor(chain, device=x.device)
                    z = torch.cat([x[: s + 1], c])
                    if kind == "block":
                        fc, lc, cc = fb, lb, None
                    else:
                        fc, lc, cc = copy_run(sliced(tcache, s), z[s:], s)
                    L = len(chain)
                    jstar = matched(chain, truth)
                    tstar = truth[jstar] if jstar < len(truth) else -1
                    # outcome distribution from the copy: logq [L+1, V] predicts s+1 .. s+L+1
                    logq = torch.log_softmax(lc[: L + 1], -1)
                    cum = torch.cat([torch.zeros(1, device=lc.device),
                                     torch.cumsum(logq[torch.arange(L), c], 0)])          # cum[j] = sum_{i<j}
                    cand = []
                    for j in range(L + 1):
                        top = logq[j].topk(min(KS[-1] + 1, logq.shape[-1]))
                        for lp, t in zip(top.values.tolist(), top.indices.tolist()):
                            if j < L and t == chain[j]:
                                continue
                            cand.append((float(cum[j]) + lp, j, t))
                    cand.sort(reverse=True)
                    # greedy target: the copy's own greedy outcome first (first position where its argmax differs
                    # from the chain, with its argmax there), then the rest by probability
                    qa = logq.argmax(-1).tolist()
                    jg = next((j for j in range(L) if qa[j] != chain[j]), L)
                    greedy = [(0.0, jg, qa[jg])] + [o for o in cand if (o[1], o[2]) != (jg, qa[jg])]
                    for name, lst in (("rank", cand), ("grank", greedy)):
                        rec[f"{kind}_{name}"] = next((r for r, (_, j, t) in enumerate(lst[: KS[-1]])
                                                      if (j, t) == (jstar, tstar)), None)
                    rec[f"{kind}_len"] = L
                    rec[f"{kind}_tau"] = jstar + 1
                    rec[f"{kind}_p1"] = float(torch.tensor(cand[0][0]).exp())
                    if kind != "refreshed" or tstar < 0:
                        continue
                    # prepared draft for the ACTUAL outcome, from copy features (no real features after s yet)
                    a = s + 1 + jstar
                    tail = truth[jstar + 1 :]                                    # truth after the anchor a
                    seq = torch.cat([z[: a], torch.tensor([tstar], device=x.device)])
                    ctx = torch.cat([feats[:, :s], fc[:, : a - s]], 1)            # real < s, copy s .. a-1
                    ccache = sliced(cc, a)                                        # target KV < s + copy KV s .. a-1
                    Q = draft_block(ctx, seq, a)
                    rec["prep_tau"], rec["prep_len"] = [matched(Q, tail) + 1], [len(Q)]
                    for _d in range(args.depth):
                        qt = torch.cat([seq[a : a + 1], torch.tensor(Q, device=x.device)])
                        fq, lq, _ = copy_run(sliced(ccache, a), qt, a)
                        qtop = lq.argmax(-1).tolist()
                        k = next((k for k in range(len(Q)) if qtop[k] != Q[k]), len(Q))
                        if a + 1 + k + bs > len(x) or k + 1 + D > args.max_len:
                            rec["prep_tau"].append(rec["prep_tau"][-1])
                            rec["prep_len"].append(len(Q))
                            continue
                        seq2 = torch.cat([seq[: a + 1], torch.tensor(Q[:k] + [qtop[k]], device=x.device)])
                        ctx2 = torch.cat([ctx[:, :a], fq], 1)
                        Q = Q[:k] + [qtop[k]] + draft_block(ctx2, seq2, a + 1 + k)
                        rec["prep_tau"].append(matched(Q, tail) + 1)
                        rec["prep_len"].append(len(Q))
                    # baselines after verification: real features up to a (x == the verified tokens up to a)
                    R = draft_block(feats, x, a)
                    rec["real0_tau"] = matched(R, tail) + 1
                    fr, lr, _ = copy_run(sliced(tcache, a), torch.cat([x[a : a + 1], torch.tensor(R, device=x.device)]), a)
                    rtop = lr.argmax(-1).tolist()
                    k = next((k for k in range(D) if rtop[k] != R[k]), None)
                    if k is None:
                        rec["real1_tau"], rec["real1_len"] = rec["real0_tau"], D
                    else:
                        seq3 = torch.cat([x[: a + 1 + k], torch.tensor([rtop[k]], device=x.device)])
                        R1 = R[:k] + [rtop[k]] + draft_block(torch.cat([feats[:, :a], fr], 1), seq3, a + 1 + k)
                        rec["real1_tau"], rec["real1_len"] = matched(R1, tail) + 1, len(R1)
                recs.append(rec)
            print(f"[{dname}] {len(recs)} anchors", flush=True)

    # ---- outcome prediction
    if not recs:
        print("no anchors (sequences too short)", flush=True)
        return
    rows = []
    for kind in ("block", "refreshed"):
        for name, label in (("rank", "copy probability"), ("grank", "copy greedy first")):
            rr = [r[f"{kind}_{name}"] for r in recs]
            row = {"in-flight chain": kind, "ranking": label, "tau": mean(r[f"{kind}_tau"] for r in recs),
                   "len": mean(r[f"{kind}_len"] for r in recs), "p_top1": mean(r[f"{kind}_p1"] for r in recs)}
            for K in KS:
                row[f"hit@{K}"] = mean(float(q is not None and q < K) for q in rr)
            rows.append(row)
    RK = "refreshed_grank" if args.temp == 0 else "refreshed_rank"          # ranking used by the event model
    print_table(rows, list(rows[0]), f"outcome prediction from the copy's probabilities, copy {args.copy}, T={args.temp}")
    # calibration of the top-1 outcome probability (refreshed chain)
    cal = []
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        sel = [r for r in recs if lo <= r["refreshed_p1"] < hi]
        if sel:
            cal.append({"p_top1 bin": f"{lo:.1f}-{min(hi, 1):.1f}", "n": len(sel),
                        "mean p": mean(r["refreshed_p1"] for r in sel),
                        "hit@1": mean(float(r["refreshed_rank"] == 0) for r in sel)})
    if cal:
        print_table(cal, list(cal[0]), "calibration: predicted top-1 outcome probability vs how often it is the actual outcome")

    # ---- prepared draft quality (actual outcome, i.e. what a hit delivers)
    pr = [r for r in recs if "prep_tau" in r]
    q = [{"next-round draft": "after verification, real features, plain block", "tau": mean(r["real0_tau"] for r in pr),
          "len": float(D)},
         {"next-round draft": "after verification, real features, + 1 loop", "tau": mean(r["real1_tau"] for r in pr),
          "len": mean(r["real1_len"] for r in pr)}]
    for d in range(args.depth + 1):
        q.append({"next-round draft": f"prepared in the shadow, copy features, depth {d}",
                  "tau": mean(r["prep_tau"][d] for r in pr), "len": mean(r["prep_len"][d] for r in pr)})
    print_table(q, list(q[0]), "quality of the next round's draft (accepted length when verified)")

    # ---- event model
    Dm, Cm, ex = args.drafter_ms, args.copy_ms, args.batch_extra
    base_tau = mean(r["tau_block1"] for r in recs)
    base_ms = Dm + v(16)
    seq_tau = mean(r["refreshed_tau"] for r in recs)
    seq_ms = Dm + Cm + args.restart_ms + v(mean(r["refreshed_len"] for r in recs) + 1)   # methodology A, sequential
    ev = [{"policy": "DFlash", "hit": "", "tau": base_tau, "ms": base_ms, "x(2 GPU)": 1.0},
          {"policy": "refresh A, sequential", "hit": "", "tau": seq_tau, "ms": seq_ms,
           "x(2 GPU)": (seq_tau / seq_ms) / (base_tau / base_ms)}]
    for kk in args.kappa:
        ev[0][f"x(1 GPU, k={kk})"] = 1.0
        ev[1][f"x(1 GPU, k={kk})"] = ev[1]["x(2 GPU)"]
    for K in KS:
        for d in range(args.depth + 1):
            mult = 1 + ex * (K - 1)
            shadow = Cm + Dm * mult + d * (Cm + Dm) * mult                 # outcome copy pass + open K + d loops on K
            h = mean(float(r[RK] is not None and r[RK] < K) for r in pr)
            tau = h * mean(r["prep_tau"][d] for r in pr) + (1 - h) * mean(r["real0_tau"] for r in pr)
            vlen = h * v(mean(r["prep_len"][d] for r in pr) + 1) + (1 - h) * v(16)
            ms2 = max(vlen, shadow) + (1 - h) * Dm                         # miss: fallback block on the critical path
            row = {"policy": f"shadow K={K} d={d}", "hit": h, "tau": tau, "ms": ms2,
                   "x(2 GPU)": (tau / ms2) / (base_tau / base_ms)}
            for kk in args.kappa:
                ms1 = vlen + kk * shadow + (1 - h) * Dm
                row[f"x(1 GPU, k={kk})"] = (tau / ms1) / (base_tau / base_ms)
            ev.append(row)
    print_table(ev, list(ev[0]), f"event model, speed vs DFlash (drafter {Dm} ms, copy {Cm} ms, vLLM verify costs); "
                                 "hit = actual outcome among the K prepared branches")
    save_json(args.out, {"args": vars(args), "outcome": rows, "calibration": cal, "quality": q, "events": ev,
                         "records": recs})


if __name__ == "__main__":
    main()
