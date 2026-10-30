"""Experiment 76: upper-layer completion, trained end to end. The drafter learns the layer-k states that the target's
own frozen upper layers turn into the right tokens (offline, Colab-ready)

exp73: the target's layers k..L-1 absorb errors in layer-k states. exp74: a LINEAR map from DFlash's states only
reaches cosine 0.63-0.80 and tau 5.6-6.5 < DFlash 8.5, because DFlash was never trained to produce such states.
Here the drafter is trained for it:

  drafter (DFlash, from its checkpoint) -> state at mask i (predicts x[s+i])
  [state ; embedding of the token at s+i-1 (anchor, or the drafter's own greedy guess)] -> projection P -> layer-k
  state for position s+i-1
  -> the target's FROZEN layers k..L-1 over [real prefix at layer k ; the B predicted states] (packed: each block
     sees its own prefix and itself causally) -> norm -> LM head -> cross-entropy on x[s+1 .. s+B]
The gradient flows through the frozen upper layers into P and the drafter. P starts from the exp74 ridge solution.
Loss = CE through the upper layers + --own-w * CE of the drafter's own head (it still supplies the previous-token
guesses). Position weights gamma^(i-1) as DFlash.

Fair control: the same fine-tune of DFlash with only its own head (same steps, data, lr, seed).

Reported on held-out anchors (tau out of 16): dflash_orig, control (fine-tuned DFlash), upper (tokens from the
target's head after completion), upper_own_head (the trained drafter's own head), plus the cost of the upper layers.
Break-even at batch 1 (exp73 cost model): tau ~ 9.4 for k = 32, ~ 10.2 for k = 28, vs DFlash ~ 8.5.

  python scripts/exp76_upper_e2e.py --cache /content/seqs.pt --k 28
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import _output_head, extract_context_feature  # noqa: E402

from hspec.lagtrain import pack  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences, e.g. from exp69")
    ap.add_argument("--k", type=int, default=28, help="first target layer the completion runs (layers k..L-1)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--blocks", type=int, default=8, help="anchors per sequence per step")
    ap.add_argument("--lr", type=float, default=1e-4, help="drafter")
    ap.add_argument("--lr-p", type=float, default=1e-4, help="projection P")
    ap.add_argument("--own-w", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--fit-anchors", type=int, default=1500, help="for the ridge initialisation of P")
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--anchors", type=int, default=500, help="eval anchors")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    args.save = args.save or f"exp76_upper_e2e_k{args.k}"
    dev = "cuda"
    t0 = time.time()

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    nv = int(len(seqs) * args.val_frac)
    val, train = seqs[:nv], seqs[nv:]
    target = load_target(args.target, dev)
    for p_ in target.parameters():
        p_.requires_grad_(False)
    head = _output_head(target)
    emb = target.get_input_embeddings()
    layers = target.model.layers
    L, H, K = len(layers), target.config.hidden_size, args.k
    neg = torch.finfo(torch.bfloat16).min

    probe = load_draft(args.draft, dev)
    lids, bs = list(probe.target_layer_ids), probe.block_size
    Hd = probe.config.hidden_size
    B = bs - 1
    del probe
    torch.cuda.empty_cache()
    w = torch.tensor([args.gamma ** i for i in range(B)], device=dev)

    @torch.no_grad()
    def target_pass(x):
        out = target(x[None], output_hidden_states=True, use_cache=False)
        return extract_context_feature(out.hidden_states, lids), out.hidden_states[K][0]

    def drafter_pass(draft, feats, x, anchors):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            noise, pos, mask, starts = pack(draft, target, feats, x, [(s, 0) for s in anchors], bs)
            hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                        past_key_values=None, use_cache=False)[0]
            hb = torch.stack([hid[st : st + B] for st in starts])                      # [A, B, Hd]
            ol = draft.compute_logits(hb, head).float()                                # own logits [A, B, V]
        return hb, ol

    def proj_input(hb, own_logits, x, anchors):
        prev = torch.cat([x[torch.tensor(anchors, device=dev)][:, None], own_logits.argmax(-1)[:, :-1]], 1)
        return torch.cat([hb.float(), emb(prev).float()], -1)                         # [A, B, Hd + H]

    def upper_packed(hk, anchors, pred):
        """Target layers K..L-1 over [real layer-K states of the whole sequence (no grad) ; A predicted blocks].
        Block a holds positions s_a .. s_a+B-1, attends to real positions < s_a and causally to itself.
        Returns final pre-norm states of the blocks [A, B, H]."""
        T = hk.shape[0]
        A = len(anchors)
        h = torch.cat([hk, pred.reshape(A * B, H).to(hk.dtype)], 0)[None]             # [1, T + A*B, H]
        bpos = torch.cat([torch.arange(s, s + B, device=dev) for s in anchors])
        pos = torch.cat([torch.arange(T, device=dev), bpos])[None]
        Q = T + A * B
        allow = torch.zeros(Q, Q, dtype=torch.bool, device=dev)
        allow[:T, :T] = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        tri = torch.ones(B, B, dtype=torch.bool, device=dev).tril()
        for a, s in enumerate(anchors):
            r0 = T + a * B
            allow[r0 : r0 + B, :s] = True
            allow[r0 : r0 + B, r0 : r0 + B] = tri
        mask = torch.zeros(1, 1, Q, Q, dtype=hk.dtype, device=dev).masked_fill(~allow[None, None], neg)
        pe = target.model.rotary_emb(h, pos)
        for layer in layers[K:]:
            o = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe, use_cache=False,
                      cache_position=pos[0])
            h = o[0] if isinstance(o, tuple) else o
        return h[0, T:].view(A, B, H)

    def upper_logits(hk, anchors, pred):
        return target.lm_head(target.model.norm(upper_packed(hk, anchors, pred))).float()

    def anchors_in(ids, lo, n, rng):
        hi = len(ids) - bs - 1
        if hi <= lo + 1:
            return []
        return sorted(rng.sample(range(lo + 1, hi), min(n, hi - lo - 1)))

    # ---------------- sanity: real layer-K states through the packed upper layers give the real tokens ----------------
    with torch.no_grad():
        ids, lo = val[0]
        x = ids.to(dev)
        _, hk = target_pass(x)
        an = anchors_in(ids, lo, 4, random.Random(5))
        real = torch.stack([hk[s : s + B] for s in an]).float()
        pr = upper_logits(hk, an, real).argmax(-1)
        truth = torch.stack([x[s + 1 : s + 1 + B] for s in an])
        print(f"[sanity] packed upper layers on REAL layer-{K} states: token agreement "
              f"{float((pr == truth).float().mean()):.4f} (should be ~1)", flush=True)

    # ---------------- ridge initialisation of P (as exp74) ----------------
    d0 = load_draft(args.draft, dev)
    d0.eval()
    rng = random.Random(1)
    XtX = torch.zeros(Hd + H + 1, Hd + H + 1, device=dev, dtype=torch.float64)
    XtY = torch.zeros(Hd + H + 1, H, device=dev, dtype=torch.float64)
    n_fit = 0
    with torch.no_grad():
        while n_fit < args.fit_anchors * B:
            ids, lo = train[rng.randrange(len(train))]
            x = ids.to(dev)
            an = anchors_in(ids, lo, 8, rng)
            if not an:
                continue
            feats, hk = target_pass(x)
            hb, ol = drafter_pass(d0, feats, x, an)
            X = proj_input(hb, ol, x, an).reshape(-1, Hd + H)
            X = torch.cat([X, torch.ones(X.shape[0], 1, device=dev)], 1).double()
            Y = torch.stack([hk[s : s + B] for s in an]).reshape(-1, H).double()
            XtX += X.T @ X
            XtY += X.T @ Y
            n_fit += X.shape[0]
        XtX += args.lam * float(XtX.diagonal().mean()) * torch.eye(XtX.shape[0], device=dev, dtype=torch.float64)
        Wr = torch.linalg.solve(XtX, XtY).float()                                    # [Hd + H + 1, H]
    del XtX, XtY
    print(f"[ridge] P initialised from {n_fit} positions, {time.time()-t0:.0f}s", flush=True)

    # ---------------- evaluation ----------------
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 1, len(ids) - bs - 1)]
    random.Random(0).shuffle(cand)
    ev = {}
    for i, s in sorted(cand[: args.anchors]):
        ev.setdefault(i, []).append(s)

    @torch.no_grad()
    def evaluate(draft, P):
        draft.eval()
        res = {key: {"tau": 0.0, "acc": 0.0, "acc4": 0.0} for key in (["own", "upper"] if P is not None else ["own"])}
        n = 0
        for i, an in ev.items():
            x = val[i][0].to(dev)
            feats, hk = target_pass(x)
            hb, ol = drafter_pass(draft, feats, x, an)
            preds = {"own": ol.argmax(-1)}
            if P is not None:
                preds["upper"] = upper_logits(hk, an, P(proj_input(hb, ol, x, an))).argmax(-1)
            for a, s in enumerate(an):
                truth = x[s + 1 : s + 1 + B]
                for key, pr in preds.items():
                    r = (pr[a] == truth).float()
                    res[key]["tau"] += matched(pr[a].tolist(), truth.tolist()) + 1
                    res[key]["acc"] += float(r.mean())
                    res[key]["acc4"] += float(r[:4].mean())
                n += 1
        return {key: {m: v / n for m, v in d.items()} for key, d in res.items()}

    rows = []
    P0 = nn.Linear(Hd + H, H).to(dev)
    with torch.no_grad():
        P0.weight.copy_(Wr[:-1].T)
        P0.bias.copy_(Wr[-1])
    r = evaluate(d0, P0)
    rows.append({"condition": "dflash_orig (own head)", **r["own"]})
    rows.append({"condition": "dflash_orig + ridge P + upper (= exp74)", **r["upper"]})
    print(f"[eval] before training: {r}", flush=True)
    del d0
    torch.cuda.empty_cache()

    def finetune(mode):
        torch.manual_seed(0)
        rng = random.Random(2)
        draft = load_draft(args.draft, dev).float()     # fp32 master weights: bf16 + lr 3e-5 rounds updates away
        draft.train()
        groups = [{"params": [p for p in draft.parameters() if p.requires_grad], "lr": args.lr}]
        P = None
        if mode == "upper":
            P = nn.Linear(Hd + H, H).to(dev)
            with torch.no_grad():
                P.weight.copy_(Wr[:-1].T)
                P.bias.copy_(Wr[-1])
            groups.append({"params": list(P.parameters()), "lr": args.lr_p})
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))
        log = []
        for step in range(args.steps):
            ids, lo = train[rng.randrange(len(train))]
            x = ids.to(dev)
            an = anchors_in(ids, lo, args.blocks, rng)
            if not an:
                continue
            feats, hk = target_pass(x)
            labels = torch.stack([x[s + 1 : s + 1 + B] for s in an])
            hb, ol = drafter_pass(draft, feats, x, an)
            ce_own = F.cross_entropy(ol.reshape(-1, ol.shape[-1]), labels.reshape(-1), reduction="none").view(-1, B)
            l_own = (ce_own * w).sum(1).mean() / w.sum()
            if mode == "upper":
                ul = upper_logits(hk, an, P(proj_input(hb, ol, x, an)))
                ce_up = F.cross_entropy(ul.reshape(-1, ul.shape[-1]), labels.reshape(-1), reduction="none").view(-1, B)
                l_up = (ce_up * w).sum(1).mean() / w.sum()
                loss = l_up + args.own_w * l_own
                acc = float((ul.argmax(-1) == labels).float().mean())
            else:
                loss = l_own
                acc = float((ol.argmax(-1) == labels).float().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            log.append((float(loss.detach()), acc))
            if step % 100 == 0 or step == args.steps - 1:
                tail = log[-100:]
                print(f"[train {mode}] step {step}: loss {sum(a for a, _ in tail)/len(tail):.4f}  "
                      f"acc {sum(b for _, b in tail)/len(tail):.3f}  {time.time()-t0:.0f}s", flush=True)
        return draft, P

    if not args.skip_control:
        draft, _ = finetune("control")
        r = evaluate(draft, None)
        rows.append({"condition": "control: DFlash fine-tuned, own head", **r["own"]})
        print(f"[eval] control: {r}", flush=True)
        del draft
        torch.cuda.empty_cache()

    draft, P = finetune("upper")
    r = evaluate(draft, P)
    rows.append({"condition": f"trained for completion: own head", **r["own"]})
    rows.append({"condition": f"trained for completion: P + target layers {K}..{L-1}", **r["upper"]})
    print(f"[eval] upper: {r}", flush=True)
    torch.save({"P": P.state_dict()}, f"{args.save}_P.pt")

    print_table(rows, list(rows[0]), f"upper-layer completion trained end to end (k = {K}: {L - K} target layers, "
                                     f"~{(L - K) / L:.0%} of a verify); tau out of {B + 1}. Break-even vs DFlash at "
                                     f"batch 1 ~ {9.4 if K >= 32 else 10.2}")
    save_json(args.save, {"args": vars(args), "rows": rows})
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
