"""Experiment 104: mixture block drafter (plan-set drafting), a learning experiment.

Idea. A parallel block drafter predicts every position independently given the context, so its block is ONE product
distribution: it can represent one plan. Where the target's continuation forks, it mixes or commits to one branch and
the tail is lost (exp103: the tail does not heal). A mixture of K products can represent K plans:

    p(block | ctx) = sum_z p(z | ctx) * prod_i p(x_i | ctx, z)

Each plan z gives a whole coherent block; the K blocks are merged into a trie and verified in one target pass
(lossless). A DDTree-style tree over one product's marginals cannot give coherent alternative TAILS: below a branch its
tokens are still the original plan's. The decisive test is therefore: trie of K plans vs the tree of the K = 1 drafter
(same recipe) at the same node count.

Model (MixDraft). DFlash's drafter body (target hidden states from 5 layers fused and injected as K/V into every
layer, target embedding and LM head shared and frozen), plus:
  plan embedding  e_z (K x --cond), added to the block inputs through in_proj
  plan modulation every layer: x = norm(h) * (1 + s) + b; h = h + (1 + g) * sublayer(x), (s, b, g) = Linear(e_z) per
                  sublayer (adaLN). Projections start near zero, so at step 0 every plan is the plain drafter.
  plan prior      MLP([fused target feature at s-1, embedding of the anchor token]) -> log p(z | ctx). Ranks plans.
All K plans of all anchors run in ONE packed pass (blocks attend to the context before their anchor and to their own
block only).

Training data (scripts/gen_forkdata_vllm.py): target greedy sequences plus, at forks (top-2 log-prob gap < 1 nat), the
target's greedy continuation after its 2nd choice. For an anchor s, the continuation set is the main block plus every
alternative block whose fork lies inside the block.

Loss (per anchor, per continuation y, position weights w_i = gamma^(i-1) as DFlash):
  ll_z(y) = sum_i w_i log p(y_i | ctx, z)
  soft (default): -log sum_z p(z | ctx) exp(ll_z(y))          (exact mixture likelihood; EM-like soft credit)
  hard:           winner z* = argmax_z log p(z|ctx) + ll_z(y); -(1 - eps)(log p(z*|ctx) + ll_z*) - eps * mean others
weighted over continuations (main 1, alternatives --alt-w x exp(-gap)) and normalised by sum(w).
K = 1 is the same model and loss with one plan (the control; train it with --alt-w 0 for the standard objective).

Init: --init scratch (random), target (copy evenly spaced target layers), dflash (released checkpoint; extra layers
start as identity).

Subcommands:
  train  train one drafter, quick evals during training, save a checkpoint
  eval   evaluate checkpoints (and the released DFlash as a reference) on the SAME held-out anchors as exp77:
         plan-0 tau, trie tau of the top-k plans and its node count, DDTree-style tree tau over plan-0 marginals at
         fixed budgets AND at each mixture model's per-anchor trie size (matched nodes), where plans diverge, prior
         calibration, fork coverage, drafter latency for K plans vs 1 at batch 1.

  python scripts/exp104_mixdraft.py train --data /scratch-shared/$USER/forkdata --cache /scratch-shared/$USER/seqs.pt --K 4 \
      --ckpt /scratch-shared/$USER/mix/K4.pt
  python scripts/exp104_mixdraft.py eval --cache /scratch-shared/$USER/seqs.pt --ckpts /scratch-shared/$USER/mix/K1.pt \
      /scratch-shared/$USER/mix/K4.pt --ref
"""

from __future__ import annotations

import argparse
import heapq
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

from dflash.model import (  # noqa: E402
    DFlashDraftModel,
    Qwen3DFlashDecoderLayer,
    _draft_value,
    _output_head,
    _raw_input_embeddings,
    build_target_layer_ids,
    extract_context_feature,
)

from hspec.models import load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def matched(chain, truth):
    n = 0
    while n < len(chain) and n < len(truth) and chain[n] == truth[n]:
        n += 1
    return n


def trie_size(chains):
    nodes = set()
    for c in chains:
        for k in range(1, len(c) + 1):
            nodes.add(tuple(c[:k]))
    return len(nodes)


def tree_tau_top(tv, ti, truth, budget):
    """exp77's DDTree-style best-first tree from stored top-w log-probs per position (tv, ti: [B][w] lists).
    Accepted = longest truth prefix in the tree + 1."""
    B, width = len(truth), len(tv[0])
    heap = [(-tv[0][j], (ti[0][j],)) for j in range(width)]
    heapq.heapify(heap)
    best = 0
    for _ in range(budget):
        if not heap:
            break
        negs, pre = heapq.heappop(heap)
        d = len(pre)
        if list(pre) == truth[:d]:
            best = max(best, d)
        if d < B:
            for j in range(width):
                heapq.heappush(heap, (negs - tv[d][j], pre + (ti[d][j],)))
    return best + 1


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------------------

def make_config(draft_id, layers):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(draft_id)
    tl = list(_draft_value(cfg, "target_layer_ids",
                           build_target_layer_ids(cfg.num_target_layers, cfg.num_hidden_layers)))
    cfg.target_layer_ids = tl                         # keep the same 5 target layers whatever the depth
    if isinstance(getattr(cfg, "dflash_config", None), dict):
        cfg.dflash_config["target_layer_ids"] = tl
    cfg.num_hidden_layers = layers
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = ["full_attention"] * layers
    cfg._attn_implementation = "sdpa"
    return cfg


def build_base(args, target=None):
    """DFlash-style drafter body with args.layers layers, fp32."""
    if args.init == "dflash":
        base = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.float32)
        n0 = len(base.layers)
        if args.layers > n0:
            cfg = base.config
            cfg.num_hidden_layers = args.layers
            if getattr(cfg, "layer_types", None):
                cfg.layer_types = list(cfg.layer_types) + ["full_attention"] * (args.layers - len(cfg.layer_types))
            for i in range(n0, args.layers):
                layer = Qwen3DFlashDecoderLayer(cfg, i)
                layer.apply(base._init_weights)
                with torch.no_grad():                 # identity at init: both sublayers write zero to the residual
                    layer.self_attn.o_proj.weight.zero_()
                    layer.mlp.down_proj.weight.zero_()
                base.layers.append(layer)
        elif args.layers < n0:
            base.layers = base.layers[: args.layers]
        return base
    base = DFlashDraftModel(make_config(args.draft, args.layers)).float()
    if args.init == "target":
        L = len(target.model.layers)
        idx = [round(i * (L - 1) / max(args.layers - 1, 1)) for i in range(args.layers)]
        for d, j in zip(base.layers, idx):
            sd = {k: v.float() for k, v in target.model.layers[j].state_dict().items()}
            missing, unexpected = d.load_state_dict(sd, strict=False)
            if missing:
                print(f"[init] layer from target {j}: missing {missing}", flush=True)
        print(f"[init] drafter layers copied from target layers {idx}", flush=True)
    return base


class MixDraft(nn.Module):
    def __init__(self, base, K, cond=256, prior_hidden=1024, init_std=1e-4):
        super().__init__()
        self.base, self.K = base, K
        H = base.config.hidden_size
        self.plan_emb = nn.Parameter(torch.randn(K, cond))
        self.in_proj = nn.Linear(cond, H, bias=False)
        nn.init.normal_(self.in_proj.weight, std=init_std)
        self.mods = nn.ModuleList([nn.Linear(cond, 6 * H) for _ in base.layers])
        for m in self.mods:
            nn.init.normal_(m.weight, std=init_std)
            nn.init.zeros_(m.bias)
        self.prior = nn.Sequential(nn.Linear(2 * H, prior_hidden), nn.SiLU(), nn.Linear(prior_hidden, K))
        nn.init.zeros_(self.prior[2].weight)
        nn.init.zeros_(self.prior[2].bias)

    def new_parameters(self):
        return [self.plan_emb] + list(self.in_proj.parameters()) + list(self.mods.parameters()) \
            + list(self.prior.parameters())

    def forward(self, feats, noise, pos, mask, plan):
        """feats [1, T, 5H] target features, noise [1, Q, H] block inputs, pos [1, T + Q], mask [1, 1, Q, T + Q],
        plan [Q] plan index of every query row. Returns (hidden [Q, H], fused context features [T, H])."""
        b = self.base
        c = self.plan_emb[plan]
        h = noise + self.in_proj(c).to(noise.dtype)[None]
        th = b.hidden_norm(b.fc(feats))
        pe = b.rotary_emb(h, pos)
        for layer, mod in zip(b.layers, self.mods):
            s1, b1, g1, s2, b2, g2 = mod(c).to(h.dtype)[None].chunk(6, -1)
            x = layer.input_layernorm(h) * (1 + s1) + b1
            a = layer.self_attn(hidden_states=x, target_hidden=th, position_embeddings=pe, attention_mask=mask)[0]
            h = h + (1 + g1) * a
            x = layer.post_attention_layernorm(h) * (1 + s2) + b2
            h = h + (1 + g2) * layer.mlp(x)
        return b.norm(h)[0], th[0]

    def log_prior(self, th, emb_anchor, anchors):
        """[A, K] log p(z | ctx) from the fused feature before the anchor and the anchor token's embedding."""
        if self.K == 1:
            return torch.zeros(len(anchors), 1, device=th.device)
        a = torch.as_tensor(anchors, device=th.device)
        z = torch.cat([th[a - 1].float(), emb_anchor.float()], -1)
        return self.prior.float()(z).log_softmax(-1)


def pack_blocks(draft_cfg_src, target, x, anchors, K, bs):
    """K copies of every anchor's block [x_s, MASK x (bs - 1)], plan-major (copy k of anchor a is block k * A + a).
    Returns noise [1, Q, H], pos [1, T + Q], mask [1, 1, Q, T + Q] (bool, True = attend), plan [Q]."""
    dev = x.device
    T, A = x.shape[0], len(anchors)
    s_t = torch.as_tensor(anchors, device=dev)
    toks = torch.full((K * A, bs), draft_cfg_src.mask_token_id, dtype=torch.long, device=dev)
    toks[:, 0] = x[s_t].repeat(K)
    scale = float(_draft_value(draft_cfg_src.config, "input_embedding_scale", 1.0))
    noise = _raw_input_embeddings(target, toks.view(1, -1), scale)
    blkpos = (s_t[:, None] + torch.arange(bs, device=dev)[None]).repeat(K, 1).view(-1)
    pos = torch.cat([torch.arange(T, device=dev), blkpos])[None]
    Q = K * A * bs
    blk = torch.arange(Q, device=dev) // bs
    anchor_of = s_t.repeat(K)[blk]
    mask = torch.zeros(Q, T + Q, dtype=torch.bool, device=dev)
    mask[:, :T] = torch.arange(T, device=dev)[None, :] < anchor_of[:, None]
    mask[:, T:] = blk[:, None] == blk[None, :]
    return noise, pos, mask[None, None], blk // A


def run_blocks(model, target, head, feats, x, anchors, bs):
    """All K plans of all anchors in one pass. Returns logits [K, A, B, V] (bf16) and log prior [A, K]."""
    K = model.K
    with torch.autocast("cuda", dtype=torch.bfloat16):
        noise, pos, mask, plan = pack_blocks(model.base, target, x, anchors, K, bs)
        hid, th = model(feats, noise, pos, mask, plan)
        hb = hid.view(K, len(anchors), bs, -1)[:, :, 1:]
        logits = model.base.compute_logits(hb, head)
    emb_anchor = _raw_input_embeddings(target, x[torch.as_tensor(anchors, device=x.device)][None],
                                       float(_draft_value(model.base.config, "input_embedding_scale", 1.0)))[0]
    return logits, model.log_prior(th, emb_anchor, anchors)


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------

def load_train(data_dirs, min_resp=24):
    recs = []
    for d in data_dirs:
        for f in sorted(Path(d).glob("*.pt")):
            for r in torch.load(f):
                ids, lo = r["ids"].long(), int(r["n_prompt"])
                if len(ids) - lo < min_resp:
                    continue
                forks = [(int(p), int(a), float(g), c.long().tolist()) for p, a, g, c in r.get("forks", [])]
                recs.append((ids, lo, forks))
    return recs


def continuations(xl, forks, s, B, alt_w, max_conts):
    """Main block + alternative blocks whose fork lies inside the block: [(tokens [B], weight)]."""
    out = [(xl[s + 1 : s + 1 + B], 1.0)]
    if alt_w <= 0:
        return out
    for pos, alt, gap, cont in forks:
        if not (s + 1 <= pos <= s + B):
            continue
        before = pos - (s + 1)
        y = xl[s + 1 : pos] + [alt] + cont[: B - before - 1]
        if len(y) == B:
            out.append((y, alt_w * math.exp(-gap)))
        if len(out) >= max_conts:
            break
    return out


def eval_anchors(cache, val_frac, n_anchors, bs):
    """exp77's held-out anchors exactly (same shuffles), so numbers compare with exp77's dflash_orig rows."""
    seqs = torch.load(cache)
    random.Random(0).shuffle(seqs)
    nv = int(len(seqs) * val_frac)
    val = seqs[: max(nv, 1)]
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 1, len(ids) - bs - 1)]
    random.Random(0).shuffle(cand)
    ev = {}
    for i, s in sorted(cand[:n_anchors]):
        ev.setdefault(i, []).append(s)
    return val, ev


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def collect(model, target, head, feats_of, val, ev, bs, width=8, chunk=16):
    """Per anchor: truth, matched length of every plan (plans ordered by prior, best first), trie sizes of the
    top-k plans, plan-0 top-`width` log-probs per position (for the tree baseline), prior of plan 0."""
    model.eval()
    B = bs - 1
    out = []
    for i, an_all in ev.items():
        x = val[i][0].to("cuda").long()
        feats = feats_of(x)
        xl = x.tolist()
        for c0 in range(0, len(an_all), chunk):
            an = an_all[c0 : c0 + chunk]
            logits, lp = run_blocks(model, target, head, feats, x, an, bs)
            pred = logits.argmax(-1)                                                # [K, A, B]
            order = lp.argsort(-1, descending=True)                                 # [A, K]
            for a, s in enumerate(an):
                truth = xl[s + 1 : s + 1 + B]
                o = order[a].tolist()
                chains = [pred[k, a].tolist() for k in o]
                l0 = logits[o[0], a].float().log_softmax(-1).topk(width, -1)
                out.append({"truth": truth, "chains": chains,
                            "m": [matched(c, truth) for c in chains],
                            "nodes": [trie_size(chains[: k + 1]) for k in range(len(chains))],
                            "tv": l0.values.tolist(), "ti": l0.indices.tolist(),
                            "p0": float(lp[a, o[0]].exp())})
    return out


def summarize(name, res, budgets):
    K = len(res[0]["chains"])
    row = {"model": name, "K": K, "tau plan0": mean(r["m"][0] + 1 for r in res)}
    for k in range(1, K + 1):
        if k > 1:
            row[f"trie top{k}"] = mean(max(r["m"][:k]) + 1 for r in res)
            row[f"nodes top{k}"] = mean(r["nodes"][k - 1] for r in res)
    for b in budgets:
        row[f"tree {b}"] = mean(tree_tau_top(r["tv"], r["ti"], r["truth"], b) for r in res)
    if K > 1:
        B = len(res[0]["truth"])
        first, tail = [], []
        for r in res:
            c0 = r["chains"][0]
            for c in r["chains"][1:]:
                d = next((j for j in range(B) if c[j] != c0[j]), B)
                first.append(d)
                if d < B - 1:
                    tail.append(mean(c[j] != c0[j] for j in range(d + 1, B)))
        row["first diff pos"] = mean(first)
        row["tail differs after split"] = mean(tail)
        row["set helps (best > plan0)"] = mean(max(r["m"]) > r["m"][0] for r in res)
        fails = [r for r in res if r["m"][0] < B]
        row["fork coverage (plan0 fails)"] = mean(max(r["m"]) > r["m"][0] for r in fails)
        row["plan0 is a best plan"] = mean(r["m"][0] == max(r["m"]) for r in res)
        row["prior p(plan0)"] = mean(r["p0"] for r in res)
    return row


@torch.no_grad()
def time_draft(model, target, head, bs, K_list, ctx=16, reps=30):
    """ms per drafter call at batch 1: ctx new target features (the cached context is not re-projected in real
    inference), K plans of one block in one pass."""
    dev = "cuda"
    W = len(model.base.target_layer_ids) * model.base.config.hidden_size
    feats = torch.randn(1, ctx + 1, W, device=dev, dtype=torch.bfloat16)      # one feature row per context token
    x = torch.randint(0, 1000, (ctx + 1,), device=dev)                         # anchor = last token, index ctx
    out = {}
    K0 = model.K
    for K in K_list:
        model.K = K
        with torch.autocast("cuda", dtype=torch.bfloat16):
            noise, pos, mask, plan = pack_blocks(model.base, target, x, [ctx], K, bs)
            plan = plan.clamp(max=model.plan_emb.shape[0] - 1)
            for _ in range(5):
                model(feats, noise, pos, mask, plan)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                hid, _ = model(feats, noise, pos, mask, plan)
                model.base.compute_logits(hid[1:], head)
            torch.cuda.synchronize()
        out[K] = (time.perf_counter() - t) / reps * 1000
    model.K = K0
    return out


def load_ckpt(path, target):
    ck = torch.load(path, map_location="cpu")
    a = argparse.Namespace(**ck["args"])
    a.init = "scratch"                      # architecture only; the weights come from the checkpoint
    base = build_base(a, target)
    model = MixDraft(base, a.K, a.cond, a.prior_hidden)
    model.load_state_dict(ck["state"])
    return model, a


# ---------------------------------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------------------------------

def cmd_train(args):
    dev = "cuda"
    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    target = load_target(args.target, dev)
    for p_ in target.parameters():
        p_.requires_grad_(False)
    head = _output_head(target)
    base = build_base(args, target)
    model = MixDraft(base, args.K, args.cond, args.prior_hidden).to(dev)
    lids, bs = list(base.target_layer_ids), base.block_size
    B = bs - 1
    w = torch.tensor([args.gamma ** i for i in range(B)], device=dev)
    n_par = sum(p.numel() for p in base.parameters()) / 1e6
    n_new = sum(p.numel() for p in model.new_parameters()) / 1e6
    print(f"[model] {args.layers} layers, init {args.init}, K {args.K}: body {n_par:.0f}M + plan parts {n_new:.0f}M "
          f"params, target layers {lids}, block {bs}", flush=True)

    train = load_train(args.data)
    if args.limit_seqs:
        train = train[: args.limit_seqs]
    nf = sum(len(f) for _, _, f in train)
    print(f"[data] {len(train)} training sequences, {nf} forks", flush=True)
    val, ev = eval_anchors(args.cache, args.val_frac, args.quick_anchors, bs)

    @torch.no_grad()
    def feats_of(x):
        return extract_context_feature(target(x[None], output_hidden_states=True, use_cache=False).hidden_states,
                                       lids)

    new = {id(p) for p in model.new_parameters()}
    body = [p for p in model.parameters() if id(p) not in new]
    groups = [{"params": model.new_parameters(), "lr": args.lr_new or args.lr}]
    if not args.freeze_base:
        groups.append({"params": body, "lr": args.lr})
    else:
        for p_ in body:
            p_.requires_grad_(False)
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=args.wd)
    steps = args.steps or math.ceil(args.epochs * len(train) / args.seqs_per_step)
    warm = max(1, int(args.warmup * steps))

    def lr_at(i):
        if i < warm:
            return (i + 1) / warm
        q = (i - warm) / max(1, steps - warm)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q))

    order = list(range(len(train)))
    rng.shuffle(order)
    cursor = 0
    log, hist = [], []
    print(f"[train] {steps} steps x {args.seqs_per_step} sequences, {args.anchors_per_seq} anchors each", flush=True)
    for step in range(steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_at(step)
        opt.zero_grad(set_to_none=True)
        tot, wins, ent = 0.0, torch.zeros(args.K), []
        for _ in range(args.seqs_per_step):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            ids, lo, forks = train[order[cursor]]
            cursor += 1
            x = ids.to(dev)
            hi = len(ids) - bs
            an = sorted(rng.sample(range(max(lo, 1), hi), min(args.anchors_per_seq, hi - max(lo, 1))))
            feats = feats_of(x)
            xl = ids.tolist()
            conts = [continuations(xl, forks, s, B, args.alt_w, args.max_conts) for s in an]
            M = max(len(c) for c in conts)
            lab = torch.zeros(len(an), B, M, dtype=torch.long)
            cw = torch.zeros(len(an), M)
            for a, cs in enumerate(conts):
                for m, (y, wt) in enumerate(cs):
                    lab[a, :, m] = torch.tensor(y)
                    cw[a, m] = wt
            lab, cw = lab.to(dev), cw.to(dev)
            logits, lp = run_blocks(model, target, head, feats, x, an, bs)              # [K,A,B,V], [A,K]
            logp = logits.float().log_softmax(-1)
            ll = (logp.gather(-1, lab[None].expand(args.K, -1, -1, -1)) * w[None, None, :, None]).sum(2)  # [K,A,M]
            score = lp.T[:, :, None] + ll
            if args.K == 1:
                term = ll[0]
            elif args.assign == "soft":
                term = score.logsumexp(0)
            else:
                ks = score.detach().argmax(0)
                best = score.gather(0, ks[None])[0]
                others = (ll.sum(0) - ll.gather(0, ks[None])[0]) / (args.K - 1)
                term = (1 - args.eps) * best + args.eps * others
            loss = -(term * cw).sum() / cw.sum() / w.sum() / args.seqs_per_step
            loss.backward()
            tot += float(loss.detach())
            wins += torch.bincount(score[:, :, 0].detach().argmax(0).cpu(), minlength=args.K).float()
            ent.append(float(-(lp.exp() * lp).sum(-1).mean()))
        for g in opt.param_groups:
            torch.nn.utils.clip_grad_norm_([p for p in g["params"] if p.grad is not None], 1.0)
        opt.step()
        log.append((tot, wins / wins.sum(), mean(ent)))
        if step % args.log_every == 0 or step == steps - 1:
            tail = log[-args.log_every:]
            sh = torch.stack([b for _, b, _ in tail]).mean(0).tolist()
            print(f"[train K={args.K}] step {step}/{steps}: loss {mean(a for a, _, _ in tail):.4f}  "
                  f"plan wins (main block) {[round(v, 2) for v in sh]}  prior entropy {mean(e for *_, e in tail):.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        if (args.eval_every and step > 0 and step % args.eval_every == 0) or step == steps - 1:
            res = collect(model, target, head, feats_of, val, ev, bs)
            row = summarize(f"step {step}", res, [15, 45])
            row["step"] = step
            hist.append(row)
            print(f"[quick eval] {row}", flush=True)
            save(args, model, hist)
    save(args, model, hist)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


def save(args, model, hist):
    if not args.ckpt:
        return
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"args": vars(args), "state": state, "hist": hist}, args.ckpt)
    print(f"[save] {args.ckpt}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------------------------------------------

def cmd_eval(args):
    dev = "cuda"
    target = load_target(args.target, dev)
    head = _output_head(target)
    models = []
    if args.ref:
        ra = argparse.Namespace(init="dflash", draft=args.draft, layers=5)
        models.append(("released DFlash", MixDraft(build_base(ra), 1, init_std=0.0), None))   # exactly DFlash
    for p in args.ckpts:
        m, a = load_ckpt(p, target)
        models.append((Path(p).stem, m, a))
    bs = models[0][1].base.block_size
    lids = list(models[0][1].base.target_layer_ids)
    val, ev = eval_anchors(args.cache, args.val_frac, args.anchors, bs)
    print(f"[eval] {sum(len(v) for v in ev.values())} anchors in {len(ev)} held-out sequences", flush=True)

    @torch.no_grad()
    def feats_of(x):
        return extract_context_feature(target(x[None], output_hidden_states=True, use_cache=False).hidden_states,
                                       lids)

    rows, allres, lat = [], {}, []
    for name, m, a in models:
        m = m.to(dev).to(torch.bfloat16)
        res = collect(m, target, head, feats_of, val, ev, bs)
        allres[name] = res
        row = summarize(name, res, args.tree_budgets)
        if a is not None:
            row.update({"layers": a.layers, "init": a.init, "alt_w": a.alt_w})
        rows.append(row)
        print(f"[eval] {row}", flush=True)
        t = time_draft(m, target, head, bs, sorted({1, m.K}))
        lat.append({"model": name, **{f"draft ms, {k} plan(s)": v for k, v in t.items()}})
        m.to("cpu")
        torch.cuda.empty_cache()

    # matched node count: every single-plan model's tree at each mixture model's per-anchor trie size
    match_rows = []
    trees = [n for n, m, _ in models if m.K == 1]
    for n, m, _ in models:
        if m.K == 1:
            continue
        rm = allres[n]
        row = {"set model": n, "trie tau": mean(max(r["m"]) + 1 for r in rm),
               "nodes": mean(r["nodes"][-1] for r in rm)}
        for tname in trees:
            rt = allres[tname]
            row[f"tree of {tname}"] = mean(tree_tau_top(q["tv"], q["ti"], q["truth"], r["nodes"][-1])
                                           for q, r in zip(rt, rm))
        match_rows.append(row)

    cols = list(dict.fromkeys(k for r in rows for k in r))
    print_table(rows, cols, "1. mixture drafter vs single-plan drafters (tau out of 16, same held-out anchors as exp77)")
    if match_rows:
        print_table(match_rows, list(match_rows[0]), "2. trie of all plans vs tree at the SAME per-anchor node count")
    print_table(lat, list(dict.fromkeys(k for r in lat for k in r)), "3. drafter latency at batch 1 (bf16)")
    save_json(args.out, {"args": vars(args), "models": rows, "matched": match_rows, "latency": lat})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--target", default="Qwen/Qwen3-8B")
        p.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16", help="config source (and weights for --init dflash)")
        p.add_argument("--cache", required=True, help="held-out (ids, n_prompt) sequences, as exp77")
        p.add_argument("--val-frac", type=float, default=0.2)
    t = sub.choices["train"]
    t.add_argument("--data", nargs="+", required=True, help="directories of gen_forkdata_vllm shards")
    t.add_argument("--K", type=int, default=4)
    t.add_argument("--layers", type=int, default=5)
    t.add_argument("--init", choices=["scratch", "target", "dflash"], default="scratch")
    t.add_argument("--cond", type=int, default=256)
    t.add_argument("--prior-hidden", type=int, default=1024)
    t.add_argument("--assign", choices=["soft", "hard"], default="soft")
    t.add_argument("--eps", type=float, default=0.05, help="hard assignment: weight shared by the other plans")
    t.add_argument("--alt-w", type=float, default=1.0, help="weight of fork alternatives (x exp(-gap)); 0 = main only")
    t.add_argument("--max-conts", type=int, default=4)
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--steps", type=int, default=0, help="overrides --epochs")
    t.add_argument("--seqs-per-step", type=int, default=4)
    t.add_argument("--anchors-per-seq", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4, help="drafter body")
    t.add_argument("--lr-new", type=float, default=0.0, help="plan parts (0 = same as --lr)")
    t.add_argument("--freeze-base", action="store_true")
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--warmup", type=float, default=0.02)
    t.add_argument("--gamma", type=float, default=0.9)
    t.add_argument("--quick-anchors", type=int, default=150)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--ckpt", default="")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--limit-seqs", type=int, default=0, help="smoke test: only the first N training sequences")
    e = sub.choices["eval"]
    e.add_argument("--ckpts", nargs="*", default=[])
    e.add_argument("--ref", action="store_true", help="also evaluate the released DFlash drafter")
    e.add_argument("--anchors", type=int, default=500)
    e.add_argument("--tree-budgets", type=int, nargs="+", default=[15, 30, 45, 60, 120])
    e.add_argument("--out", default="exp104_mixdraft")
    args = ap.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_eval(args)


if __name__ == "__main__":
    main()
