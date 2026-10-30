"""Experiment 72: free lookahead from the verify pass (offline, Colab-ready)

Idea: append m placeholder positions after the tokens the target verifies. Real tokens never attend to them (they come
last, causal mask), so verification is unchanged and lossless. The target's hidden states at the placeholders are its
own guess of the states at the next m positions, at almost no extra verify cost (verify time grows little with
length). A chained second drafting stage (DLoop-style) needs exactly these: features for positions the target has not
verified yet.

Minimal version: the target stays FROZEN; only m placeholder input embeddings are learned (prompt-tuning style,
16 x 4096 parameters), trained so the target's tap states at the placeholders match its real states at those
positions (1 - cosine per tap) plus cross-entropy of its LM head on the true next tokens.

Offline test, same protocol as exp71: anchor s, real context features up to p = s - m, and for the last m positions
p .. s-1 one of
  real       the target's real features (upper bound)
  removed    nothing: context ends at p
  la_trained the target's states at the m learned placeholders appended after x[:p]
  la_repeat  untrained placeholders: the last real token's embedding repeated (zero-training baseline)
then the DFlash block from s is scored against the target's greedy continuation (tau = matched + 1).
Also reported per placeholder offset j: cosine to the real state per tap, and top-1 accuracy of the target's LM head
(a free multi-token prediction).

Reference numbers from exp71 (same cache): real 8.44; removed 3.60 / 2.18 / 1.60 and copy features 8.42 / 8.38 / 8.30
for m = 4 / 8 / 15.

  python scripts/exp72_lookahead.py --cache /content/seqs.pt
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences, e.g. from exp69")
    ap.add_argument("--k", type=int, default=16, help="placeholders learned (eval uses the first m)")
    ap.add_argument("--m", nargs="+", type=int, default=[4, 8, 15])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--per-step", type=int, default=4, help="prefixes per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tap-w", nargs="+", type=float, default=[0.25, 0.25, 0.25, 1.0, 1.0])
    ap.add_argument("--ce-w", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--anchors", type=int, default=600, help="eval anchors per m")
    ap.add_argument("--save", default="exp72_lookahead")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    torch.manual_seed(0)

    seqs = [(ids, lo) for ids, lo in torch.load(args.cache) if len(ids) - lo >= args.k + 40]
    random.Random(0).shuffle(seqs)
    nv = int(len(seqs) * args.val_frac)
    val, train = seqs[:nv], seqs[nv:]
    print(f"[data] {len(train)} train / {len(val)} val sequences", flush=True)

    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p_ in target.parameters():
        p_.requires_grad_(False)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    nT, H, K = len(lids), target.config.hidden_size, args.k
    emb = target.get_input_embeddings()
    tap_w = torch.tensor(args.tap_w, device=dev)

    # learned placeholders, initialised at the mean token embedding (+ small noise so they differ)
    with torch.no_grad():
        mu = emb.weight.float().mean(0)
        sd = emb.weight.float().std()
    P = torch.nn.Parameter((mu[None].repeat(K, 1) + 0.02 * sd * torch.randn(K, H, device=dev)))
    opt = torch.optim.AdamW([P], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))

    def lookahead(x, p, ph):
        """Target pass over real tokens x[:p] followed by placeholder embeddings ph [k, H].
        Returns tap features [k, nT*H] and logits [k, V] at the placeholders."""
        e = torch.cat([emb(x[None, :p]), ph[None].to(emb.weight.dtype)], 1)
        out = target(inputs_embeds=e, output_hidden_states=True, use_cache=False, logits_to_keep=ph.shape[0])
        f = extract_context_feature(out.hidden_states, lids)[0, p:]
        return f, out.logits[0].float()

    @torch.no_grad()
    def real_feats(x):
        out = target(x[None], output_hidden_states=True, use_cache=False)
        return extract_context_feature(out.hidden_states, lids)

    # ---------------- train the placeholders (target frozen) ----------------
    target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    target.train()                                    # checkpointing is only active in train mode; Qwen3 has no dropout
    rng = random.Random(1)
    log = []
    for step in range(args.steps):
        ids, lo = train[rng.randrange(len(train))]
        x = ids.to(dev)
        rf = real_feats(x)[0]                         # [T, nT*H]
        opt.zero_grad(set_to_none=True)
        tot = {"loss": 0.0, "cos_deep": 0.0, "acc": 0.0}
        for _ in range(args.per_step):
            p = rng.randrange(lo + 8, len(x) - K - 1)
            f, lg = lookahead(x, p, P)
            cos = F.cosine_similarity(f.float().view(K, nT, H), rf[p : p + K].float().view(K, nT, H), dim=-1)  # [K, nT]
            l_tap = ((1 - cos) * tap_w).sum(-1).mean() / tap_w.sum()
            l_ce = F.cross_entropy(lg, x[p + 1 : p + K + 1])
            loss = (l_tap + args.ce_w * l_ce) / args.per_step
            loss.backward()
            tot["loss"] += float(loss)
            tot["cos_deep"] += float(cos[:, -2:].mean()) / args.per_step
            tot["acc"] += float((lg.argmax(-1) == x[p + 1 : p + K + 1]).float().mean()) / args.per_step
        torch.nn.utils.clip_grad_norm_([P], 1.0)
        opt.step()
        sched.step()
        log.append(tot)
        if step % 50 == 0 or step == args.steps - 1:
            w = log[-50:]
            print(f"[train] step {step}: loss {sum(r['loss'] for r in w)/len(w):.4f}  "
                  f"cos(L25,L33) {sum(r['cos_deep'] for r in w)/len(w):.3f}  "
                  f"lm-head acc {sum(r['acc'] for r in w)/len(w):.3f}  {time.time()-t0:.0f}s", flush=True)
    target.gradient_checkpointing_disable()
    target.eval()
    torch.save(P.detach().cpu(), f"{args.save}_placeholders.pt")

    # ---------------- evaluate on held-out sequences ----------------
    M = max(args.m)
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + M + 8, len(ids) - bs)]
    random.Random(0).shuffle(cand)
    by_seq = {}
    for i, s in sorted(cand[: args.anchors]):
        by_seq.setdefault(i, []).append(s)
    srcs = ("real", "removed", "la_trained", "la_repeat")
    tau = {(src, m): [] for src in srcs for m in args.m}
    cos_j = {src: torch.zeros(M, nT, device=dev) for src in ("la_trained", "la_repeat")}
    acc_j = {src: torch.zeros(M, device=dev) for src in ("la_trained", "la_repeat")}
    n_j = 0
    with torch.inference_mode():
        for n_done, (i, anchors) in enumerate(by_seq.items()):
            ids, lo = val[i]
            x = ids.to(dev)
            rf = real_feats(x)
            for s in anchors:
                truth = x[s + 1 :].tolist()
                for m in args.m:
                    p = s - m
                    la = {}
                    la["la_trained"], lg_t = lookahead(x, p, P.detach()[:m])
                    rep = emb.weight[x[p - 1]].float()[None].repeat(m, 1)
                    la["la_repeat"], lg_r = lookahead(x, p, rep)
                    for src in srcs:
                        if src == "removed":
                            L = blocks_logits(draft, target, rf[:, :p], x, [(s, m)], bs)[0]   # as exp71
                        elif src == "real":
                            L = blocks_logits(draft, target, rf[:, :s], x, [(s, 0)], bs)[0]
                        else:
                            fz = torch.cat([rf[:, :p], la[src][None].to(rf.dtype)], 1)
                            L = blocks_logits(draft, target, fz, x, [(s, 0)], bs)[0]
                        tau[(src, m)].append(matched(L.argmax(-1).tolist(), truth) + 1)
                    if m == M:
                        rm = rf[0, p : p + m].float().view(m, nT, H)
                        for src, lg in (("la_trained", lg_t), ("la_repeat", lg_r)):
                            cos_j[src] += F.cosine_similarity(la[src].float().view(m, nT, H), rm, dim=-1)
                            acc_j[src] += (lg.argmax(-1) == x[p + 1 : p + m + 1]).float()
                        n_j += 1
            if n_done % 20 == 0:
                print(f"[eval] {n_done}/{len(by_seq)} sequences, {time.time()-t0:.0f}s", flush=True)

    def mean(v):
        return sum(v) / max(len(v), 1)

    rows = [{"m": m, **{src: mean(tau[(src, m)]) for src in srcs}} for m in args.m]
    print_table(rows, ["m", *srcs], "tau when the last m context positions get: real target features / nothing / the "
                                    "target's states at m learned placeholders / at m untrained (repeat-last) placeholders")
    jrows = []
    for j in range(M):
        r = {"offset": j + 1}
        for src in ("la_trained", "la_repeat"):
            c = (cos_j[src][j] / max(n_j, 1)).tolist()
            r[f"{src}_cos_L{lids[-2]}"] = c[-2]
            r[f"{src}_cos_L{lids[-1]}"] = c[-1]
            r[f"{src}_lmhead_acc"] = float(acc_j[src][j] / max(n_j, 1))
        jrows.append(r)
    print_table(jrows, list(jrows[0]), f"per placeholder offset (m = {M}): cosine to the real state at the deep taps, "
                                       f"and top-1 accuracy of the target's own LM head on the true token")
    save_json(args.save, {"args": vars(args), "tau": rows, "per_offset": jrows,
                          "n_anchors": sum(len(v) for v in by_seq.values()), "train_log_tail": log[-50:]})
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
