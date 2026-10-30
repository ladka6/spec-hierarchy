"""Experiment 75: upper bound for a "future summary". Does DFlash draft better if it gets a compact, TRUE summary of
the block it is drafting? (offline, Colab-ready)

If even the true summary does not help, there is nothing worth having the target predict. If it helps, the next step
is whether the target can predict that summary in its verify pass (one placeholder + gated LoRA, exp72 machinery):
one vector is a much easier target than 16 per-position states, and it needs no per-position token.

Summary of the block at anchor s: the target's real states at its deepest feature layer for positions s+1 .. s+B
(they have read the future tokens), averaged over --chunks equal chunks (1 = one mean; 4 = NCM-like coarse codes),
then squeezed through a learned bottleneck of d dimensions (down H -> d per chunk, up d*chunks -> drafter width).
d limits how much it can leak the tokens; the result is tau as a function of d.
The summary is ADDED to the drafter's input embeddings of the whole block (anchor + masks): no change to attention.

Fair comparison (DFlash has never seen such an input): every condition, including d = 0 (no summary), is the same
short fine-tune of DFlash from its checkpoint, same data, steps, lr and seed. "dflash_orig" = the untouched checkpoint.

  python scripts/exp75_futuresummary.py --cache /content/seqs.pt                      # d in 0 16 64 256
  python scripts/exp75_futuresummary.py --cache /content/seqs.pt --chunks 4 --ds 0 16 64
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


class Bottleneck(nn.Module):
    def __init__(self, H, d, chunks, out_dim, scale=0.3):
        super().__init__()
        self.scale = scale
        self.norm = nn.LayerNorm(H)             # deep-layer states have huge, outlier-heavy norms: normalise first
        self.down = nn.Linear(H, d)
        self.up = nn.Linear(d * chunks, out_dim)
        # bounded injection: the summary is rescaled to `scale` x the norm of the block's input embedding.
        # scale > 0: fixed, so the projections get full gradients from step 0. (An unbounded version grew to ~100x the
        # input and drowned the block positions; a learned gate tanh(g), g = 0 at init, never opened because the
        # projections behind a closed gate get no gradient and stay random.) scale <= 0: the learned gate.
        self.gate = nn.Parameter(torch.zeros(())) if scale <= 0 else None

    def forward(self, chunk_means, ref_norm):   # [A, chunks, H], scalar -> [A, out_dim]
        z = self.up(self.down(self.norm(chunk_means)).flatten(1))
        g = torch.tanh(self.gate) if self.gate is not None else self.scale
        return g * ref_norm * F.normalize(z, dim=-1)


class KVSummary(nn.Module):
    """The summary as `chunks` extra key/value entries of the drafter's context (the channel DFlash reads target
    features from), instead of one vector added to the block's inputs. Each chunk mean -> d dims -> the drafter's
    feature width (the 5 concatenated target layers), plus a learned chunk-index embedding; rescaled to the norm of a
    real feature row. The drafter's own fc + RMSNorm then treats the entries like target features."""

    def __init__(self, H, d, chunks, feat_dim):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        self.down = nn.Linear(H, d)
        self.up = nn.Linear(d, feat_dim)
        self.chunk_emb = nn.Parameter(torch.randn(chunks, feat_dim) * 0.02)

    def forward(self, chunk_means, ref_norm):   # [A, chunks, H], scalar -> [A, chunks, feat_dim]
        z = self.up(self.down(self.norm(chunk_means))) + self.chunk_emb
        return ref_norm * F.normalize(z, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True, help="(ids, n_prompt) greedy sequences, e.g. from exp69")
    ap.add_argument("--ds", nargs="+", type=int, default=[0, 16, 64, 256], help="bottleneck sizes (0 = no summary)")
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--inject", choices=["add", "kv"], default="add",
                    help="add: one vector added to the block's input embeddings; kv: one extra key/value context "
                         "entry per chunk, visible only to its own block")
    ap.add_argument("--source", choices=["deep", "tokens"], default="deep",
                    help="deep: the block's real deep-layer target states; tokens: the block's true token embeddings "
                         "(ORACLE channel check: a bag of the future tokens must help, else the channel is broken)")
    ap.add_argument("--scale", type=float, default=0.3,
                    help="summary norm as a fraction of the input-embedding norm (fixed); <= 0 = learned tanh gate")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--blocks", type=int, default=16, help="anchors per sequence per step")
    ap.add_argument("--lr", type=float, default=1e-4, help="drafter")
    ap.add_argument("--lr-b", type=float, default=1e-3, help="bottleneck")
    ap.add_argument("--gamma", type=float, default=0.9, help="position weight decay in the loss (as DFlash)")
    ap.add_argument("--anchors", type=int, default=500, help="eval anchors")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    args.save = args.save or f"exp75_futuresummary_{args.inject}_{args.source}_c{args.chunks}"
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
    H = target.config.hidden_size

    probe = load_draft(args.draft, dev)
    lids, bs = list(probe.target_layer_ids), probe.block_size
    B = bs - 1
    deep = lids[-1] + 1                          # hidden_states index of the deepest feature layer
    del probe
    torch.cuda.empty_cache()

    @torch.no_grad()
    def target_pass(x):
        out = target(x[None], output_hidden_states=True, use_cache=False)
        return extract_context_feature(out.hidden_states, lids), out.hidden_states[deep][0].float()

    emb = target.get_input_embeddings()

    @torch.no_grad()
    def source_of(x, hdeep):
        """Per-position source vectors [T, H] the summary is built from, standardised per dimension."""
        src = hdeep if args.source == "deep" else emb(x).float()
        return (src - src_mu) / src_sd

    # per-dimension standardisation stats: deep states have a few huge outlier dimensions shared by every position,
    # which drown the position-specific content (LayerNorm alone keeps them dominant)
    with torch.no_grad():
        acc_s, acc_q, n_s = 0.0, 0.0, 0
        for ids, _lo in train[:40]:
            x_ = ids.to(dev)
            _, hd_ = target_pass(x_)
            v = hd_ if args.source == "deep" else emb(x_).float()
            acc_s = acc_s + v.sum(0)
            acc_q = acc_q + (v * v).sum(0)
            n_s += v.shape[0]
        src_mu = acc_s / n_s
        src_sd = (acc_q / n_s - src_mu ** 2).clamp_min(1e-6).sqrt()

    def summary_input(src, anchors):
        """[A, chunks, H]: chunk means of the (standardised) source vectors at positions s+1 .. s+B."""
        blk = torch.stack([src[s + 1 : s + 1 + B] for s in anchors])                 # [A, B, H]
        return torch.stack([c.mean(1) for c in blk.chunk(args.chunks, dim=1)], 1)

    stats = {"ratio": 0.0}

    def run(draft, bott, feats, x, anchors, hdeep, train_mode, sum_anchors=None):
        """Packed DFlash pass over all anchors, with the summary added to each block's input embeddings."""
        noise, pos, mask, starts = pack(draft, target, feats, x, [(s, 0) for s in anchors], bs)
        if bott is not None and args.inject == "kv":
            T, Q, c = feats.shape[1], noise.shape[1], args.chunks
            ref = feats.detach()[0].float().norm(dim=-1).mean()
            extra = bott(summary_input(source_of(x, hdeep), sum_anchors or anchors), ref)  # [A, c, F]
            stats["ratio"] = float(extra.detach().norm(dim=-1).mean() / ref.clamp_min(1e-9))
            feats = torch.cat([feats, extra.reshape(1, -1, extra.shape[-1]).to(feats.dtype)], 1)
            spos = torch.tensor([s for s in anchors for _ in range(c)], device=dev)    # at the anchor's position
            pos = torch.cat([pos[0, :T], spos, pos[0, T:]])[None]
            m = mask[0, 0]
            m2 = torch.zeros(Q, T + len(spos) + Q, dtype=m.dtype, device=dev)
            m2[:, :T] = m[:, :T]
            m2[:, T + len(spos):] = m[:, T:]
            for a, st in enumerate(starts):                                             # each block sees its own
                m2[st - 1 : st + B, T + a * c : T + (a + 1) * c] = True
            mask = m2[None, None]
        elif bott is not None:
            ref = noise.detach()[0].float().norm(dim=-1).mean()
            add = bott(summary_input(source_of(x, hdeep), sum_anchors or anchors), ref).to(noise.dtype)
            # how large the summary is next to the block's own input embeddings (logged during training)
            stats["ratio"] = float(add.detach().norm(dim=-1).mean() / noise.detach()[0].norm(dim=-1).mean().clamp_min(1e-9))
            rows = torch.zeros(noise.shape[1], add.shape[1], device=dev, dtype=noise.dtype)
            for a, st in enumerate(starts):
                rows = rows.index_add(0, torch.arange(st - 1, st + B, device=dev),
                                      add[a][None].expand(B + 1, -1))
            noise = noise + rows[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = draft(target_hidden=feats, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                        past_key_values=None, use_cache=False)[0]
            hb = torch.stack([hid[st : st + B] for st in starts])                      # [A, B, W]
            return draft.compute_logits(hb, head).float()                              # [A, B, V]

    def anchors_in(ids, lo, n, rng):
        hi = len(ids) - bs - 1
        if hi <= lo + 1:
            return []
        return sorted(rng.sample(range(lo + 1, hi), min(n, hi - lo - 1)))

    # fixed eval anchors on held-out sequences
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 1, len(ids) - bs - 1)]
    random.Random(0).shuffle(cand)
    ev = {}
    for i, s in sorted(cand[: args.anchors]):
        ev.setdefault(i, []).append(s)

    @torch.no_grad()
    def evaluate(draft, bott, shuffle=False):
        """shuffle: each block gets the summary of ANOTHER anchor of the same sequence (rolled by one). If tau does not
        drop, the trained drafter ignores the summary."""
        draft.eval()
        tau, acc, acc4, n = 0.0, 0.0, 0.0, 0
        for i, anchors in ev.items():
            if shuffle and len(anchors) < 2:
                continue
            x = val[i][0].to(dev)
            feats, hdeep = target_pass(x)
            sa = anchors[1:] + anchors[:1] if shuffle else None
            lg = run(draft, bott, feats, x, anchors, hdeep, False, sum_anchors=sa)
            pred = lg.argmax(-1)
            for a, s in enumerate(anchors):
                truth = x[s + 1 : s + 1 + B]
                r = (pred[a] == truth).float()
                tau += matched(pred[a].tolist(), truth.tolist()) + 1
                acc += float(r.mean())
                acc4 += float(r[:4].mean())
                n += 1
        return {"tau": tau / n, "acc": acc / n, "acc@1-4": acc4 / n}

    rows_out = []
    d0 = load_draft(args.draft, dev)
    r = evaluate(d0, None)
    rows_out.append({"condition": "dflash_orig", "d": "-", **r})
    print(f"[eval] dflash_orig: {r}", flush=True)
    del d0
    torch.cuda.empty_cache()

    w = torch.tensor([args.gamma ** k for k in range(B)], device=dev)
    for d in args.ds:
        torch.manual_seed(0)
        rng = random.Random(1)
        draft = load_draft(args.draft, dev).float()     # fp32 master weights: bf16 + lr 3e-5 rounds updates away
        draft.train()
        width = draft.config.hidden_size
        if d == 0:
            bott = None
        elif args.inject == "kv":
            bott = KVSummary(H, d, args.chunks, len(lids) * H).to(dev)
        else:
            bott = Bottleneck(H, d, args.chunks, width, args.scale).to(dev)
        groups = [{"params": [p for p in draft.parameters() if p.requires_grad], "lr": args.lr}]
        if bott is not None:
            groups.append({"params": list(bott.parameters()), "lr": args.lr_b})
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))
        log = []
        for step in range(args.steps):
            ids, lo = train[rng.randrange(len(train))]
            x = ids.to(dev)
            anchors = anchors_in(ids, lo, args.blocks, rng)
            if not anchors:
                continue
            feats, hdeep = target_pass(x)
            lg = run(draft, bott, feats, x, anchors, hdeep, True)                     # [A, B, V]
            labels = torch.stack([x[s + 1 : s + 1 + B] for s in anchors])
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels.reshape(-1), reduction="none").view(-1, B)
            loss = (ce * w).sum(1).mean() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # clip each group on its own: a joint clip lets the drafter's gradients swamp the bottleneck's, so the
            # zero-initialised summary never grew (first run: d = 16 tracked d = 0 exactly)
            for g in groups:
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            opt.step()
            sched.step()
            log.append((float(loss.detach()), float((lg[:, 0].argmax(-1) == labels[:, 0]).float().mean())))
            if step % 100 == 0 or step == args.steps - 1:
                tail = log[-100:]
                print(f"[train d={d}] step {step}: loss {sum(a for a, _ in tail)/len(tail):.4f}  "
                      f"acc@1 {sum(b for _, b in tail)/len(tail):.3f}  |summary|/|input| {stats['ratio']:.3f}  "
                      f"gate {float(torch.tanh(bott.gate.detach())) if getattr(bott, 'gate', None) is not None else (args.scale if bott is not None and args.inject == 'add' else 0.0):.3f}  "
                      f"{time.time()-t0:.0f}s", flush=True)
        r = evaluate(draft, bott)
        rows_out.append({"condition": "finetune" if d == 0 else f"finetune + true summary ({args.source})", "d": d, **r})
        print(f"[eval] d={d}: {r}", flush=True)
        if bott is not None:
            rs = evaluate(draft, bott, shuffle=True)
            rows_out.append({"condition": "  same, summary of another anchor (shuffled)", "d": d, **rs})
            print(f"[eval] d={d} shuffled: {rs}", flush=True)
        del draft, bott, opt
        torch.cuda.empty_cache()

    print_table(rows_out, list(rows_out[0]), f"upper bound: DFlash fine-tuned with a TRUE summary of the block "
                                             f"({args.chunks} chunk mean(s) of the real L{deep - 1} states through a "
                                             f"d-dim bottleneck) vs the same fine-tune without (d = 0); tau out of {B + 1}")
    save_json(args.save, {"args": vars(args), "rows": rows_out})
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
