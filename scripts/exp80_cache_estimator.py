"""Experiment 80: cache-reading target-state estimator (minimal), DFlash and target frozen.

Problem. After a verification at anchor s the verifier's states of every verified position are known. DFlash drafts
block 1; assume its first n tokens are accepted. To draft block 2 at anchor s+n without a verification in between,
DFlash needs features for the in-flight positions s .. s+n-1, whose tokens are known but whose verifier states are not.
Estimate L25 / L33 there (the shallow taps take their mean: no loss, exp79 E0) so that the unmodified DFlash drafts
block 2 as with real features. One verification per round; estimates live only inside the in-flight span.

Estimator (one model):
  memory    the verified context's real states at the 5 taps DFlash already stores (L1, L9, L17, L25, L33), last W
            positions, standardised per dimension, one memory token per (position, tap)
  queries   in-flight positions p: [embedding of x_p ; embedding of x_{p+1}] (x_{p+1} is the target's own greedy
            output if the prefix is accepted; for the last position it is block 2's anchor)
  body      blocks of [self-attention over the span (bidirectional: all tokens known) -> cross-attention to the
            memory -> MLP]
  output    standardised L25 and L33 (L33 = head + learned skip from L25)
Training: target text (greedy), random anchor s and span n in 1..span, teacher forcing (true tokens = a correct draft,
no cut involved). Loss: Huber on standardised L25 / L33 + KL of the frozen DFlash's block 2 (anchor s+n) with estimated
vs real features.
Eval (--stops): all rounds, oracle cut n from block 1 vs the true continuation; block 2 from real / tokens-only /
estimated features; tokens per verification chained (n + m2 + 1) vs normal DFlash (n + 1).

  python scripts/exp80_cache_estimator.py --cache seqs.pt --data .../data --out est80.pt
  python scripts/exp80_cache_estimator.py --cache seqs.pt --load est80.pt --steps 0        # eval only
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

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.feathead import token_emb  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


class XBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa_qkv, self.sa_o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ca_q, self.ca_kv, self.ca_o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mem, mmask):
        B, S, D = x.shape
        M, h = mem.shape[1], self.h
        q, k, v = self.sa_qkv(self.ln1(x)).view(B, S, 3, h, D // h).permute(2, 0, 3, 1, 4)
        x = x + self.sa_o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, D))
        q = self.ca_q(self.ln2(x)).view(B, S, h, D // h).transpose(1, 2)
        k, v = self.ca_kv(mem).view(B, M, 2, h, D // h).permute(2, 0, 3, 1, 4)
        x = x + self.ca_o(F.scaled_dot_product_attention(q, k, v, attn_mask=mmask).transpose(1, 2).reshape(B, S, D))
        return x + self.mlp(self.ln3(x))


class CacheEstimator(nn.Module):
    def __init__(self, H, ntaps, d=1024, layers=4, heads=16, window=256, span=16):
        super().__init__()
        self.cfg = dict(H=H, ntaps=ntaps, d=d, layers=layers, heads=heads, window=window, span=span)
        self.window, self.span = window, span
        self.tap_in = nn.ModuleList(nn.Linear(H, d) for _ in range(ntaps))
        self.tap_emb, self.mpos = nn.Embedding(ntaps, d), nn.Embedding(window, d)
        self.mem_ln = nn.LayerNorm(d)
        self.tok_in, self.qpos = nn.Linear(2 * H, d), nn.Embedding(span, d)
        self.blocks = nn.ModuleList(XBlock(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out25, self.out33 = nn.Linear(d, H), nn.Linear(d, H)
        self.skip = nn.Parameter(torch.zeros(H))
        for lin in (self.out25, self.out33):                 # start at the dataset mean (standardised 0)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("mu", torch.zeros(ntaps, H))
        self.register_buffer("sd", torch.ones(ntaps, H))

    def memory(self, z):
        """z [T, ntaps, H] standardised states of one sequence -> [T, ntaps, d]"""
        return torch.stack([lin(z[:, i]) for i, lin in enumerate(self.tap_in)], 1) + self.tap_emb.weight[None]

    def forward(self, mem, mvalid, e):
        """mem [A, W, ntaps, d], mvalid [A, W] bool, e [A, n, 2H] -> standardised (y25, y33) [A, n, H]"""
        A, W, K, d = mem.shape
        mem = self.mem_ln((mem + self.mpos.weight[:W][None, :, None]).reshape(A, W * K, d))
        mmask = mvalid[:, :, None].expand(A, W, K).reshape(A, 1, 1, W * K)
        x = self.tok_in(e) + self.qpos.weight[: e.shape[1]]
        for b in self.blocks:
            x = b(x, mem, mmask)
        h = self.norm(x)
        y25 = self.out25(h)
        return y25, self.out33(h) + self.skip * y25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences; first 20%% after shuffle = eval")
    ap.add_argument("--data", default="", help="gen_traindata shards for training")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--span", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4, help="sequences per step")
    ap.add_argument("--anchors-train", type=int, default=32, help="regression anchors per sequence (tiled)")
    ap.add_argument("--kl-anchors", type=int, default=4, help="anchors per sequence with the drafter KL term")
    ap.add_argument("--kl-weight", type=float, default=3.0)
    ap.add_argument("--zclip", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--stats-seqs", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-stops", type=int, default=150, help="anchors for the periodic stop eval")
    ap.add_argument("--stops", type=int, default=600, help="anchors for the final stop eval")
    ap.add_argument("--load", default="")
    ap.add_argument("--project", nargs="*", type=int, default=[],
                    help="also evaluate the estimates projected onto the top-r PCA directions of the real L25 / L33 "
                         "(fit on held-in sequences), rest = mean")
    ap.add_argument("--out", default="")
    ap.add_argument("--save", default="exp80_cache_estimator")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    rng = random.Random(0)
    torch.manual_seed(0)

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    val = seqs[: max(int(len(seqs) * 0.2), 1)]
    train = []
    if args.data:
        for f in sorted(Path(args.data).glob("*.pt")):
            train += [(r["ids"].long(), int(r["n_prompt"])) for r in torch.load(f)]
    else:
        train = seqs[len(val) :]
    train = [(ids, lo) for ids, lo in train if len(ids) <= args.max_len and len(ids) - lo > 3 * 16 + 8]
    rng.shuffle(train)
    print(f"[data] {len(train)} train / {len(val)} eval sequences", flush=True)

    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H, K = target.config.hidden_size, len(lids)
    i25, i33 = lids.index(25), lids.index(33)
    W, S = args.window, args.span

    @torch.no_grad()
    def feats_of(x):
        out = target(x[None], output_hidden_states=True, use_cache=False, logits_to_keep=1)
        return extract_context_feature(out.hidden_states, lids)[0]                     # [T, K*H] bf16

    if args.load:
        d = torch.load(args.load, map_location="cpu")
        est = CacheEstimator(**d["cfg"]).to(dev)
        est.load_state_dict(d["state"])
        W, S = est.window, est.span
    else:
        est = CacheEstimator(H, K, args.dim, args.layers, args.heads, W, S).to(dev)
        acc_s = torch.zeros(K, H, dtype=torch.float64, device=dev)
        acc_q = torch.zeros(K, H, dtype=torch.float64, device=dev)
        n_pos = 0
        for ids, lo in train[: args.stats_seqs]:
            f = feats_of(ids.to(dev))[lo:].double().view(-1, K, H)
            acc_s += f.sum(0)
            acc_q += f.pow(2).sum(0)
            n_pos += f.shape[0]
        mu = acc_s / n_pos
        est.mu.copy_(mu.float())
        est.sd.copy_((acc_q / n_pos - mu * mu).clamp_min(1e-8).sqrt().float().clamp_min(1e-4))
    print(f"[estimator] {sum(p.numel() for p in est.parameters()) / 1e6:.1f}M parameters, window {W}, span {S}",
          flush=True)

    def standardise(f):
        z = (f.float().view(-1, K, H) - est.mu) / est.sd
        return z.clamp(-args.zclip, args.zclip) if args.zclip > 0 else z

    def estimate(f, x, anchors, n, z=None):
        """standardised (y25, y33) [A, n, H] for positions s .. s+n-1 of each anchor s"""
        z = standardise(f) if z is None else z
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem_all = est.memory(z)                                                    # [T, K, d]
            st = torch.tensor(anchors, device=dev)
            idx = st[:, None] - W + torch.arange(W, device=dev)[None]                  # [A, W]
            valid = idx >= 0
            mem = mem_all[idx.clamp_min(0)]
            toks = torch.stack([x[s : s + n] for s in anchors])
            nxt = torch.stack([x[s + 1 : s + n + 1] for s in anchors])
            e = torch.cat([token_emb(draft, target, toks), token_emb(draft, target, nxt)], -1).float()
            y25, y33 = est(mem, valid, e)
        return y25.float(), y33.float()

    def raw_part(y25, y33):
        """[n, K*H] features for the in-flight positions: shallow taps at their mean, L25 / L33 estimated"""
        n = y25.shape[0]
        out = est.mu[None].expand(n, K, H).clone()
        out[:, i25] = y25 * est.sd[i25] + est.mu[i25]
        out[:, i33] = y33 * est.sd[i33] + est.mu[i33]
        return out.view(n, K * H)

    # ---------------------------------------------------------------------------------------------- eval
    pca = {}
    if args.project:
        held = seqs[len(val) :]
        s1 = {t: torch.zeros(H, dtype=torch.float64, device=dev) for t in (i25, i33)}
        s2 = {t: torch.zeros(H, H, dtype=torch.float64, device=dev) for t in (i25, i33)}
        npos = 0
        for ids_h, lo_h in held[:200]:
            fh = feats_of(ids_h.to(dev))[lo_h:].double().view(-1, K, H)
            for t in (i25, i33):
                s1[t] += fh[:, t].sum(0)
                s2[t] += fh[:, t].T @ fh[:, t]
            npos += fh.shape[0]
        for t in (i25, i33):
            m = s1[t] / npos
            _, U = torch.linalg.eigh(s2[t] / npos - torch.outer(m, m))
            pca[t] = (m.float(), U.flip(1).float())
        print(f"[pca] fitted on {npos} held-in positions, ranks {args.project}", flush=True)

    def project(e, rank):
        n_ = e.shape[0]
        ev = e.float().view(n_, K, H).clone()
        for t, (m, U) in pca.items():
            Ur = U[:, :rank]
            ev[:, t] = m + ((ev[:, t] - m) @ Ur) @ Ur.T
        return ev.view(n_, K * H)

    def stop_eval(n_anchors, tag):
        est.eval()
        vr = random.Random(7)
        cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 8, len(ids) - 3 * bs)]
        vr.shuffle(cand)
        by = {}
        for i, s in cand[:n_anchors]:
            by.setdefault(i, []).append(s)
        recs = []
        with torch.inference_mode():
            for i, anchors in by.items():
                ids, lo = val[i]
                x = ids.to(dev)
                f = feats_of(x)
                z = standardise(f)
                for s in anchors:
                    y1 = blocks_logits(draft, target, f[None, :s], x, [(s, 0)], bs)[0].argmax(-1).tolist()
                    n = matched(y1, x[s + 1 :].tolist())
                    r = {"n": n, "real": 0, "tokens": 0, "est": 0, **{f"est@pca{rr}": 0 for rr in args.project}}
                    if n > 0:
                        a = s + n
                        truth = x[a + 1 :].tolist()
                        lg = blocks_logits(draft, target, f[None, :a], x, [(a, 0)], bs)[0]
                        r["real"] = matched(lg.argmax(-1).tolist(), truth)
                        lg = blocks_logits(draft, target, f[None, :s], x, [(a, n)], bs)[0]
                        r["tokens"] = matched(lg.argmax(-1).tolist(), truth)
                        if n <= S:
                            y25, y33 = estimate(f, x, [s], n, z)
                            fz = torch.cat([f[:s], raw_part(y25[0], y33[0]).to(f.dtype)], 0)[None]
                            lg = blocks_logits(draft, target, fz, x, [(a, 0)], bs)[0]
                            r["est"] = matched(lg.argmax(-1).tolist(), truth)
                            for rr in args.project:
                                fz = torch.cat([f[:s], project(raw_part(y25[0], y33[0]), rr).to(f.dtype)], 0)[None]
                                lg = blocks_logits(draft, target, fz, x, [(a, 0)], bs)[0]
                                r[f"est@pca{rr}"] = matched(lg.argmax(-1).tolist(), truth)
                    recs.append(r)
        est.train()
        k = len(recs)
        normal = sum(r["n"] + 1 for r in recs) / k
        rows = [{"block 2 from": "none (normal DFlash)", "first token right": "-", "mean m2": 0.0,
                 "tokens / verification": normal, "vs normal": 1.0}]
        for c in ("real", "tokens", "est") + tuple(f"est@pca{rr}" for rr in args.project):
            t = sum(r["n"] + r[c] + 1 for r in recs) / k
            rows.append({"block 2 from": c, "first token right": sum(r[c] >= 1 for r in recs if r["n"] > 0)
                         / max(sum(r["n"] > 0 for r in recs), 1), "mean m2": sum(r[c] for r in recs) / k,
                         "tokens / verification": t, "vs normal": t / normal})
        print_table(rows, list(rows[0]), f"[{tag}] chained round, all {k} rounds, oracle cut, drafter {args.draft}")
        return rows

    # ---------------------------------------------------------------------------------------------- train
    hist = []
    if args.steps > 0:
        opt = torch.optim.AdamW(est.parameters(), lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 200) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(s, args.steps)
                                                                                   / args.steps))))
        it, step, tl = 0, 0, time.time()
        run = {"reg": 0.0, "kl": 0.0, "z25": 0.0, "z33": 0.0, "n": 0}
        while step < args.steps:
            opt.zero_grad(set_to_none=True)
            for _ in range(args.batch):
                ids, lo = train[it % len(train)]
                it += 1
                x = ids.to(dev)
                T = len(ids)
                n = rng.randint(1, S)
                hi = T - n - bs - 2
                if hi <= lo + 1:
                    continue
                anc = list(range(lo + 1 + rng.randrange(S), hi, S))
                if len(anc) > args.anchors_train:
                    anc = sorted(rng.sample(anc, args.anchors_train))
                if not anc:
                    continue
                f = feats_of(x)
                z = standardise(f)
                y25, y33 = estimate(f, x, anc, n, z)
                r25 = torch.stack([z[s : s + n, i25] for s in anc])
                r33 = torch.stack([z[s : s + n, i33] for s in anc])
                reg = F.smooth_l1_loss(y25, r25) + F.smooth_l1_loss(y33, r33)
                kl_tot = torch.zeros((), device=dev)
                if args.kl_weight > 0:
                    kl_idx = rng.sample(range(len(anc)), min(args.kl_anchors, len(anc)))
                    for a_i in kl_idx:
                        s = anc[a_i]
                        a = s + n                                              # block 2's anchor (cut = n)
                        mix = torch.cat([f[:s], raw_part(y25[a_i], y33[a_i]).to(f.dtype)], 0)[None]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lg = blocks_logits(draft, target, mix, x, [(a, 0)], bs).float()
                        with torch.no_grad():
                            ref = blocks_logits(draft, target, f[None, :a], x, [(a, 0)], bs).float()
                        kl_tot = kl_tot + F.kl_div(torch.log_softmax(lg, -1), torch.log_softmax(ref, -1),
                                                   log_target=True, reduction="none").sum(-1).mean() / len(kl_idx)
                loss = reg + args.kl_weight * kl_tot
                (loss / args.batch).backward()
                run["reg"] += float(reg.detach())
                run["kl"] += float(kl_tot.detach())
                run["z25"] += float(F.cosine_similarity(y25.detach(), r25, dim=-1).mean())
                run["z33"] += float(F.cosine_similarity(y33.detach(), r33, dim=-1).mean())
                run["n"] += 1
            torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 and run["n"]:
                m = run["n"]
                print(f"[step {step}/{args.steps}] huber {run['reg'] / m:.4f} kl {run['kl'] / m:.4f} "
                      f"zcos25 {run['z25'] / m:.3f} zcos33 {run['z33'] / m:.3f}  {(time.time() - tl) / step:.2f}s/step",
                      flush=True)
                run = {k_: 0.0 for k_ in run}
            if step % args.eval_every == 0 or step == args.steps:
                if args.out:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state": est.state_dict(), "cfg": est.cfg}, args.out)
                if step < args.steps:
                    hist.append({"step": step, "rows": stop_eval(args.eval_stops, f"step {step}")})

    rows = stop_eval(args.stops, "final")
    save_json(args.save + f"_{Path(args.draft).name}", {"args": vars(args), "history": hist, "final": rows})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
