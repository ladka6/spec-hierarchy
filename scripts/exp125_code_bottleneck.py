"""Experiment 125: how much must a speculative feature carry once the drafter is trained to read it?

exp84 gave the frozen drafter the real deep features of the in-flight positions restricted to a rank-r linear
subspace: 256 of 4096 dimensions gave 0.68 of the ceiling, 1024 gave 0.92. That is the "imitate the target's state"
route. Here everything is trained for the drafter's own loss: the in-flight positions get a learned r-dimensional
CODE of their real features (encoder: 5 taps -> r, decoder: r -> the drafter's fused feature space), and the drafter
is fine-tuned together with the encoder and decoder on the block loss only. No term asks the code to resemble the
target's state. This is the upper bound of a speculative-feature model that would have to PREDICT r numbers per
in-flight token: if a small r is enough, the thing to predict is small.

Dual mode. A block at anchor s with lag g gets
  context   real fused features of positions < s - g, and the decoded code of positions s - g .. s - 1
  queries   the in-flight tokens x_{s-g} .. x_s and the MASKs, as in lag training (the token path stays)
Arms (--code):
  0      no code rows: plain lag training (the baseline, same data and steps)
  r > 0  code of r dimensions; starts as the rank-r PCA projection of the fused real feature (so step 0 is the
         exp84 point on this drafter), then encoder, decoder and drafter are trained on the block loss
  real   the real fused features at the in-flight positions (the ceiling of this layout)
  state:L        the REAL state after target layer L - 1 (hidden_states[L]) through one projection (closed-form
                 start, exp118): the ceiling of any producer that aims at that single state
  path:LAYERS    a SPECULATIVE state: only the listed target layers run over the in-flight tokens (frozen, each
                 reading the verify's real prefix KV at its own depth), the removed spans are bridged (affine
                 closed-form start + MLP, exp116), the output stands in for the state after the last listed layer
                 and goes through the same projection. Bridges, projection and drafter are trained on the block
                 loss only: nothing asks the state to match the real one. Cost = number of listed layers
                 (0.54 ms each on Qwen3-8B) + bridges. Examples: path:0,1,28   path:0,9,19,28   path:0,1,21-28
                 A part ending in "a" runs ATTENTION-ONLY: path:0,1,21-28a keeps the attention of layers 21..28
                 (which reads the real prefix KV) and replaces each MLP, about 3/4 of a layer's weights, by a
                 rank --sur-rank stand-in (linear part = truncated closed-form fit of the MLP, nonlinear part from
                 zero). An attention-only layer costs about a quarter of a layer.
--freeze-draft: the drafter is not trained (what joint training adds).
Start: the lag-trained drafter. (A first version started the code at zero from the raw taps: the drafter learned
to ignore the rows and all code arms stayed at the baseline, an optimisation failure, not a result.)

Reported on held-out sequences, per lag (4, 8, 16) and for standard blocks (lag 0): loss, first drafted token
right, tau = accepted drafted tokens + 1; at the start and at the end of training.

  python scripts/exp125_code_bottleneck.py --data /scratch-shared/$USER/hspec_lagft/data \
      --draft ~/hspec_ckpt/drafter_lag16_fp32 --code 64 --out /scratch-shared/$USER/hspec_code/r64
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

from exp116_band_bridge import BF, BandStack, CtxBridge, rot  # noqa: E402

from hspec.lagtrain import pack, sample_blocks  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


class Sur(nn.Module):
    """rank-r stand-in for one MLP sublayer (input: the layer's own post-attention norm)"""

    def __init__(self, H, r):
        super().__init__()
        self.a = nn.Linear(H, r, bias=False)                 # linear part, set from the closed-form fit
        self.b = nn.Linear(r, H, bias=True)
        self.gate = nn.Linear(H, r, bias=False)
        self.up = nn.Linear(H, r, bias=False)
        self.down = nn.Linear(r, H, bias=False)
        for m in (self.a, self.b, self.down):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.b.bias)

    def forward(self, h):
        dt = h.dtype
        h = h.float()
        return (self.b(self.a(h)) + self.down(F.silu(self.gate(h)) * self.up(h))).to(dt)


class Path(BandStack):
    """BandStack whose layers in attn_only keep their attention and replace the MLP by a Sur"""

    def __init__(self, target, kept, stop, kind, rank, attn_only, sur_rank):
        super().__init__(target, kept, stop, kind, rank, 16)
        self.attn_only = set(attn_only)
        H, dev = target.config.hidden_size, target.lm_head.weight.device
        self.sur = nn.ModuleDict({str(g): Sur(H, sur_rank) for g in sorted(self.attn_only)}).to(dev)

    def cost(self):
        full = sum(p.numel() for p in self._orig[0].parameters())
        att = sum(p.numel() for p in self._orig[0].self_attn.parameters())
        own = sum(p.numel() for p in self.parameters())
        n_att = len(self.attn_only)
        return ((len(self.kept) - n_att) * full + n_att * att + own) / full

    def forward(self, x0, a, kvp, cos, sin, hs):
        m = x0.shape[1]
        mask = torch.ones(m, a + m, dtype=torch.bool, device=x0.device)
        mask[:, a:] = torch.tril(mask[:, a:])
        rep = self.nh // self.nkv
        x, outs = x0, {}
        for g in range(self.stop):
            if g not in self.kset:
                if str(g) in self.bridges:
                    br = self.bridges[str(g)]
                    with torch.autocast("cuda", enabled=False):
                        x = br(x, hs[g + 1][:, max(0, a - br.win) : a]) if isinstance(br, CtxBridge) else br(x)
                    outs[g] = x
                continue
            lay = self._orig[g]
            attn = lay.self_attn
            hn = lay.input_layernorm(x)
            q = attn.q_proj(hn).view(1, m, -1, self.hd)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            q = rot(q.transpose(1, 2).to(BF), cos, sin)
            kn, vn = self._kv(attn, hn, cos, sin)
            Kp, Vp = kvp[g]
            K = torch.cat([Kp[:, :, :a], kn], 2).repeat_interleave(rep, 1)
            V = torch.cat([Vp[:, :, :a], vn], 2).repeat_interleave(rep, 1)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            x = x + attn.o_proj(o.transpose(1, 2).reshape(1, m, -1))
            h2 = lay.post_attention_layernorm(x)
            if g in self.attn_only:
                with torch.autocast("cuda", enabled=False):
                    x = x + self.sur[str(g)](h2)
            else:
                mlp = lay.mlp
                x = x + mlp.down_proj(mlp.act_fn(mlp.gate_proj(h2)) * mlp.up_proj(h2))
        return x, outs


class Code(nn.Module):
    """fused real feature of one position -> r numbers -> a vector in the drafter's fused feature space.
    Linear path initialised to the rank-r PCA projection (the exp84 point: useful from step 0, so the drafter does
    not learn to ignore the rows), plus zero-initialised nonlinear residuals on both sides."""

    def __init__(self, dim, r, mu, U, hid=1024):
        super().__init__()
        self.enc = nn.Linear(dim, r)
        self.dec = nn.Linear(r, dim)
        self.enc_res = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, r))
        self.dec_res = nn.Sequential(nn.Linear(r, hid), nn.GELU(), nn.Linear(hid, dim))
        with torch.no_grad():
            self.enc.weight.copy_(U.T)                       # c = U^T (x - mu)
            self.enc.bias.copy_(-(mu @ U))
            self.dec.weight.copy_(U)                         # x' = U c + mu
            self.dec.bias.copy_(mu)
            for m in (self.enc_res[-1], self.dec_res[-1]):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.float()
        c = self.enc(x) + self.enc_res(x)
        return self.dec(c) + self.dec_res(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", required=True, help="the lag-trained drafter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--code", default="64", help="0 (lag baseline), r > 0, or real")
    ap.add_argument("--max-lag", type=int, default=16)
    ap.add_argument("--p-zero", type=float, default=0.25)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-code", type=float, default=1e-4)
    ap.add_argument("--pca-seqs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-seqs", type=int, default=48, help="as train_lag.py: the same held-out sequences")
    ap.add_argument("--val-blocks", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-draft", action="store_true")
    ap.add_argument("--bridge", default="mlp", choices=["lin", "mlp", "attn"])
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--sur-rank", type=int, default=256)
    ap.add_argument("--save", default="exp125_code")
    args = ap.parse_args()
    dev, t0 = "cuda", time.time()
    kept, attn_only, sl = [], [], 0
    if args.code == "real":
        mode = "real"
    elif args.code.startswith("path:"):
        mode = "path"
        for part in args.code[5:].split(","):
            ao = part.endswith("a")
            lo, _, hi = part.rstrip("a").partition("-")
            layers = list(range(int(lo.rstrip("a")), int(hi or lo.rstrip("a")) + 1))
            kept += layers
            attn_only += layers if ao else []
        sl = max(kept) + 1
    elif args.code.startswith("state:"):
        mode, sl = "state", int(args.code[6:])
    else:
        mode = "none" if int(args.code) == 0 else "code"

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = []
    for f in sorted(Path(args.data).glob("shard*.pt")):
        data += torch.load(f)
    data = [d for d in data if len(d["ids"]) <= args.max_len]
    rng.shuffle(data)
    val, train = data[: args.val_seqs], data[args.val_seqs :]
    if args.max_train:
        train = train[: args.max_train]
    print(f"{len(train)} train / {len(val)} val sequences, arm: {args.code}", flush=True)

    target = load_target(args.target, dev)
    for p in target.parameters():
        p.requires_grad_(False)
    draft = load_draft(args.draft, dev).float()
    if args.freeze_draft:
        for p in draft.parameters():
            p.requires_grad_(False)
    else:
        draft.train()
    lids, bs = list(draft.target_layer_ids), draft.block_size
    fc = draft.fc                                            # trained with the drafter, applied here
    draft.fc = nn.Identity()                                 # the drafter is fed fused features
    width, Hd = fc.in_features, fc.out_features
    H = target.config.hidden_size
    stack = proj = None
    if mode == "path":
        stack = Path(target, kept, sl, args.bridge, args.rank, attn_only, args.sur_rank)
        print(f"[path] layers {kept}, attention-only {attn_only}, removed spans {stack.spans}, "
              f"stands in for hidden_states[{sl}], "
              f"cost {stack.cost():.2f} layers", flush=True)

    def feats_of(ids):
        """the sequence's real features; for state / path arms also the hidden states and the verify's prefix KV"""
        with torch.no_grad():
            hs = target(ids[None], output_hidden_states=True, logits_to_keep=1).hidden_states
            seq = {"feats": extract_context_feature(hs, lids)}
            if mode == "state":
                seq["state"] = hs[sl]
            if mode == "path":
                pos = torch.arange(len(ids), device=dev)[None]
                cos, sin = target.model.rotary_emb(hs[0], pos)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    kvp = stack.prefix_kv(hs, cos, sin)
                seq.update(hs=hs, cos=cos, sin=sin, kvp=kvp)
            return seq

    if mode in ("state", "path"):                            # closed-form starts (exp118 projection, exp116 bridges)
        with torch.no_grad():
            xtx = torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev)
            xty = torch.zeros(H + 1, Hd, dtype=torch.float64, device=dev)
            spans = stack.spans if stack is not None and stack.bridges else []
            acc = {b: [torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev),
                       torch.zeros(H + 1, H, dtype=torch.float64, device=dev)] for _, b in spans}

            def add(x, y, A, B):
                w = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                xa = torch.cat([x, torch.ones_like(x[:, :1])], -1) * w
                A += xa.T @ xa
                B += xa.T @ (y * w)

            mlp_io, hooks = {}, []                           # input / output of the MLPs to be replaced
            macc = {g: [torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev),
                        torch.zeros(H + 1, H, dtype=torch.float64, device=dev)] for g in attn_only}
            for g in attn_only:
                hooks.append(target.model.layers[g].mlp.register_forward_hook(
                    lambda mod, inp, out, g=g: mlp_io.__setitem__(g, (inp[0], out))))
            for d in train[: args.pca_seqs]:
                ids = d["ids"].long().to(dev)
                hs = target(ids[None], output_hidden_states=True, logits_to_keep=1).hidden_states
                lo = d["n_prompt"]
                y = fc(extract_context_feature(hs, lids).float())[0, lo:].double()
                add(hs[sl][0, lo:].double(), y, xtx, xty)
                for a, b in spans:
                    x = hs[a][0, lo:].double()
                    add(x, hs[b + 1][0, lo:].double() - x, *acc[b])
                for g in attn_only:
                    add(mlp_io[g][0][0, lo:].double(), mlp_io[g][1][0, lo:].double(), *macc[g])
                del hs
                mlp_io.clear()
            for h in hooks:
                h.remove()

            def solve(A, B):
                eye = torch.eye(H + 1, dtype=torch.float64, device=dev)
                return torch.linalg.solve(A + args.ridge * A.diagonal().mean() * eye, B).float()

            sol = solve(xtx, xty)
            proj = nn.Linear(H, Hd, bias=True).to(dev)
            proj.weight.copy_(sol[:H].T)
            proj.bias.copy_(sol[H])
            for _, b in spans:
                sol = solve(*acc[b])
                stack.bridges[str(b)].lin.weight.copy_(sol[:H].T)
                stack.bridges[str(b)].lin.bias.copy_(sol[H])
            for g in attn_only:                              # rank-r part of the closed-form linear fit of the MLP
                sol = solve(*macc[g])
                U, S, Vh = torch.linalg.svd(sol[:H].T, full_matrices=False)
                r = args.sur_rank
                sur = stack.sur[str(g)]
                sur.a.weight.copy_(Vh[:r])
                sur.b.weight.copy_(U[:, :r] * S[:r])
                sur.b.bias.copy_(sol[H])
                print(f"[fit] MLP of layer {g}: rank {r} keeps {float(S[:r].pow(2).sum() / S.pow(2).sum()):.3f} "
                      f"of the linear fit", flush=True)
            del xtx, xty, acc, macc
        print(f"[fit] projection from hidden_states[{sl}]" + (f", {len(spans)} bridges" if spans else "")
              + f", {time.time() - t0:.0f}s", flush=True)

    code = None
    if mode == "code":                                       # PCA of the fused real features (response positions)
        r = int(args.code)
        with torch.no_grad():
            sm = torch.zeros(Hd, dtype=torch.float64, device=dev)
            xx = torch.zeros(Hd, Hd, dtype=torch.float64, device=dev)
            n = 0
            for d in train[: args.pca_seqs]:
                ids = d["ids"].long().to(dev)
                x = fc(feats_of(ids)["feats"].float())[0, d["n_prompt"] :].double()
                sm += x.sum(0)
                xx += x.T @ x
                n += x.shape[0]
            mu = sm / n
            cov = xx / n - torch.outer(mu, mu)
            ev, evec = torch.linalg.eigh(cov)
            U = evec[:, -r:].flip(1).float()
            print(f"[pca] {n} positions, top-{r} directions hold {float(ev[-r:].sum() / ev.sum()):.3f} of the "
                  f"variance, {time.time() - t0:.0f}s", flush=True)
        code = Code(Hd, r, mu.float(), U).to(dev)
    dparams = [] if args.freeze_draft else [p for p in list(draft.parameters()) + list(fc.parameters())
                                             if p.requires_grad]
    extra_mods = [m for m in (code, proj, stack) if m is not None]
    eparams = [p for m in extra_mods for p in m.parameters()]
    groups = ([{"params": dparams, "lr": args.lr}] if dparams else []) \
        + ([{"params": eparams, "lr": args.lr_code}] if eparams else [])
    assert groups, "nothing to train"
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    total = max(1, int(len(train) * args.epochs / args.batch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    print(f"trainable: drafter {sum(p.numel() for p in dparams) / 1e6:.0f} M"
          + (f", feature side {sum(p.numel() for p in eparams) / 1e6:.1f} M" if eparams else "")
          + f", {total} steps, {time.time() - t0:.0f}s", flush=True)

    def forward(seq, ids, blocks):
        """hidden states of the packed blocks; the in-flight positions of each block read the arm's features"""
        T = ids.shape[0]
        real = fc(seq["feats"].float())                      # [1, T, Hd]
        noise, pos, mask, starts = pack(draft, target, real, ids, blocks, bs)
        if mode == "none":
            ctx = real
        elif mode == "path":                                 # every block has its own speculative rows
            rows, rpos, spans_b = [], [], []
            c = T
            for blk in blocks:
                s, g = blk[0], blk[1]
                if g:
                    a = s - g
                    st = stack(seq["hs"][0][:, a:s], a, seq["kvp"], seq["cos"][:, a:s], seq["sin"][:, a:s],
                               seq["hs"])[0]
                    rows.append(proj(st.float()).to(real.dtype))
                    rpos.append(torch.arange(a, s, device=ids.device))
                spans_b.append((c, c + g))
                c += g
            ctx = torch.cat([real] + rows, 1)
            q = noise.shape[1]
            m = torch.zeros((q, c + q), dtype=torch.bool, device=ids.device)
            m[:, :T] = mask[0, 0, :, :T]
            m[:, c:] = mask[0, 0, :, T:]
            a = 0
            for blk, (ra, rb) in zip(blocks, spans_b):
                m[a : a + blk[1] + bs, ra:rb] = True
                a += blk[1] + bs
            mask = m[None, None]
            pos = torch.cat([pos[:, :T]] + [r[None] for r in rpos] + [pos[:, T:]], 1)
        else:
            extra = (real if mode == "real" else proj(seq["state"].float()) if mode == "state"
                     else code(real.detach()))
            ctx = torch.cat([real, extra.to(real.dtype)], 1)  # [1, 2T, Hd]: real rows, then the in-flight rows
            q = noise.shape[1]
            m = torch.zeros((q, 2 * T + q), dtype=torch.bool, device=ids.device)
            m[:, :T] = mask[0, 0, :, :T]
            m[:, 2 * T :] = mask[0, 0, :, T:]
            a = 0
            for blk in blocks:
                s, g = blk[0], blk[1]
                m[a : a + g + bs, T + s - g : T + s] = True
                a += g + bs
            mask = m[None, None]
            pos = torch.cat([torch.arange(T, device=ids.device)[None], pos], 1)
        hid = draft(target_hidden=ctx, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                    past_key_values=None, use_cache=False)[0]
        return hid, starts

    def loss_fn(seq, ids, blocks, stats=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid, starts = forward(seq, ids, blocks)
            T = ids.shape[0]
            rows, labels, weights, first = [], [], [], []
            for blk, st in zip(blocks, starts):
                s = blk[0]
                for k in range(1, bs):
                    if s + k >= T:
                        break
                    rows.append(st + k - 1)
                    labels.append(int(ids[s + k]))
                    weights.append(args.gamma ** (k - 1))
                    if k == 1:
                        first.append(len(rows) - 1)
            rows_t = torch.tensor(rows, device=ids.device)
            labels_t = torch.tensor(labels, device=ids.device)
            w = torch.tensor(weights, device=ids.device, dtype=torch.float32)
            logits = draft.compute_logits(hid[rows_t], _output_head(target)).float()
        ce = F.cross_entropy(logits, labels_t, reduction="none")
        loss = (ce * w).sum() / w.sum()
        ok = logits.argmax(-1) == labels_t
        acc1 = ok[first].float().mean()
        if not stats:
            return loss, acc1
        taus, i = [], 0                                      # accepted drafted tokens + 1 per block
        for blk in blocks:
            n = min(bs - 1, T - 1 - blk[0])
            run = ok[i : i + n].tolist()
            taus.append(1 + (run.index(False) if False in run else n))
            i += n
        return loss, acc1, taus

    def evaluate():
        draft.eval()
        vrng = random.Random(123)
        rows = []
        with torch.no_grad():
            for lag in (0, 4, 8, 16):
                ls, a1, taus = [], [], []
                for d in val:
                    ids = d["ids"].long().to(dev)
                    blocks = [(s, min(lag, s - 1)) for s, _ in
                              sample_blocks(d["n_prompt"], len(ids), args.val_blocks, bs, 0, 1.0, vrng)]
                    if not blocks:
                        continue
                    l, a, t = loss_fn(feats_of(ids), ids, blocks, stats=True)
                    ls.append(float(l))
                    a1.append(float(a))
                    taus += t
                rows.append({"lag": lag, "loss": sum(ls) / len(ls), "first token": sum(a1) / len(a1),
                             "tau": sum(taus) / len(taus), "blocks": len(taus)})
        if not args.freeze_draft:
            draft.train()
        return rows

    def show(tag, rows):
        print(f"[{tag}] " + "  ".join(f"lag{r['lag']}: loss {r['loss']:.3f} first {r['first token']:.3f} "
                                       f"tau {r['tau']:.3f}" for r in rows), flush=True)

    start = evaluate()
    show("step 0", start)
    step, order = 0, list(range(len(train)))
    while step < total:
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if step >= total:
                break
            opt.zero_grad(set_to_none=True)
            tot_l, tot_a, n = 0.0, 0.0, 0
            for j in order[i : i + args.batch]:
                d = train[j]
                ids = d["ids"].long().to(dev)
                blocks = sample_blocks(d["n_prompt"], len(ids), args.blocks, bs, args.max_lag, args.p_zero, rng)
                if not blocks:
                    continue
                loss, acc = loss_fn(feats_of(ids), ids, blocks)
                (loss / args.batch).backward()
                tot_l += float(loss.detach())
                tot_a += float(acc)
                n += 1
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[step {step}/{total}] loss {tot_l / max(n, 1):.3f} acc1 {tot_a / max(n, 1):.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            if step % args.eval_every == 0 and step < total:
                show(f"step {step}", evaluate())
    end = evaluate()
    show("final", end)

    table = []
    for a, b in zip(start, end):
        table.append({"arm": args.code, "lag": a["lag"], "tau start": a["tau"], "tau end": b["tau"],
                      "first start": a["first token"], "first end": b["first token"],
                      "loss start": a["loss"], "loss end": b["loss"], "blocks": b["blocks"]})
    print_table(table, list(table[0]), f"In-flight positions read: {args.code} (drafter trained with it, "
                                       f"{total} steps; tau = accepted drafted tokens + 1)")
    save_json(args.save, {"args": vars(args), "start": start, "end": end, "table": table})
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"draft": {k: v.detach().to(torch.bfloat16) for k, v in draft.state_dict().items()},
                    "fc": {k: v.detach().to(torch.bfloat16) for k, v in fc.state_dict().items()},
                    "code": code.state_dict() if code is not None else None,
                    "proj": proj.state_dict() if proj is not None else None,
                    "bridges": stack.bridges.state_dict() if stack is not None else None,
                    "sur": stack.sur.state_dict() if stack is not None else None,
                    "args": vars(args)}, out / "state.pt")
        print(f"saved {out / 'state.pt'}", flush=True)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
