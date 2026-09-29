"""One-GPU three-level decoding with split verification on an early-exit (LayerSkip) target.

  drafter      DFlash block drafter proposes a block (chain) or a DDTree-style tree
  exit check   the target's first k layers + its own final norm / LM head verify the tree
               (greedy walk, gated: a position is trusted only if the exit's top-1
               probability >= thr). The accepted path's layer-1..k KV and layer-k hidden
               states are kept; they are exactly what the full target computes there.
  upper pass   every m rounds (or earlier, see below) the target's layers k+1..L run on the
               layer-k hidden states of all exit-accepted positions, reusing the lower KV.
               Its predictions are the final word: the first mismatch rolls back.

The upper pass is forced when the exit is unsure (no trusted next token), when a stop token
is pending, when `max_pending` positions wait, or at the length limit.

Drafter context: positions verified by an upper pass have all target features; positions
only exit-verified have the shallow ones (target layers below k) and zeros for the deep ones
(the depth-lag format of hspec.lagtrain, see train_lag.py --depth-exit).

Greedy losslessness: every final token is the full target's argmax given the final prefix.

Time is reported two ways: real wall clock (eager HF, same for the baselines) and a
simulated clock with costs from hspec.async3.Costs scaled by the measured split ratios
(bench_split.py: lower k layers + exit head ~ r_exit, upper layers ~ r_up of a full pass).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dflash.model import _draft_value, _make_cache, _output_head, _raw_input_embeddings

from hspec.async3 import Costs
from hspec.lagtrain import deep_columns
from hspec.pipeline import GenResult, _first_stop, _sync_time, crop
from hspec.tree import build_ddtree, chain, greedy_walk, tree_inputs, tree_mask


@dataclass
class SplitConfig:
    exit_layer: int = 16
    thr: float = 0.7            # trust the exit where its top-1 probability >= thr
    budget: int = 32            # drafter tree nodes checked by the exit (0 = chain of bs-1)
    m: int = 4                  # exit rounds per upper pass
    max_pending: int = 96       # force an upper pass once this many positions wait
    r_exit: float = 0.53        # simulated cost of lower layers + exit head, fraction of a pass
    r_up: float = 0.53          # simulated cost of the upper layers
    top_k: int = 16
    thr_joint: float = 0.0      # also trust the exit at >= this when it agrees with the drafter's top-1
    merge: bool = False         # fuse each upper pass with a full check of a new drafter tree
    r_lower: float = 0.47       # simulated cost of the lower layers without the exit head
    feat_exit: int = 0          # exit layer the drafter's depth-lag training assumed (0 = exit_layer)


@dataclass
class SplitStats:
    rounds: int = 0
    upper_passes: int = 0
    rollbacks: int = 0
    unsure_flushes: int = 0
    rollback_tokens: int = 0
    merged: int = 0
    exit_accepted: list = field(default_factory=list)
    merged_accepted: list = field(default_factory=list)
    upper_q: list = field(default_factory=list)


def _crop_layers(cache, layers, length):
    for i in layers:
        lay = cache.layers[i]
        if getattr(lay, "is_initialized", True) and lay.keys is not None and lay.keys.numel():
            lay.keys = lay.keys[..., :length, :]
            lay.values = lay.values[..., :length, :]


def _compact_layers(cache, layers, keep, rows):
    for i in layers:
        lay = cache.layers[i]
        idx = torch.tensor(list(range(keep)) + [keep + r for r in rows], device=lay.keys.device)
        lay.keys = lay.keys.index_select(-2, idx)
        lay.values = lay.values.index_select(-2, idx)


class SplitTarget:
    """Runs contiguous layer ranges of a HF decoder-only model with a shared DynamicCache in
    which lower and upper layers may hold different lengths."""

    def __init__(self, model, feat_layer_ids):
        self.model, self.base = model, model.model
        self.L = model.config.num_hidden_layers
        self.feat_ids = list(feat_layer_ids)
        self.cache = _make_cache(model.config)

    def run(self, lo, hi, h, pos, mask):
        """Layers [lo, hi) on hidden h [1, q, H]; returns (h_out, {layer_id: output})."""
        pe = self.base.rotary_emb(h, pos)
        feats = {}
        for i in range(lo, hi):
            h = self.base.layers[i](h, attention_mask=mask, position_ids=pos, past_key_values=self.cache,
                                    use_cache=True, position_embeddings=pe)
            if isinstance(h, tuple):
                h = h[0]
            if i in self.feat_ids:
                feats[i] = h
        return h, feats

    def head(self, h):
        return self.model.lm_head(self.base.norm(h))

    def embed(self, ids):
        return self.base.embed_tokens(ids)


def _causal(past, q, device):
    return torch.ones((q, past + q), dtype=torch.bool, device=device).tril(past)[None, None]


@torch.inference_mode()
def split_generate(draft, target, input_ids, max_new_tokens, stop_token_ids, cfg: SplitConfig,
                   costs: Costs | None = None, exit_head=None) -> GenResult:
    """exit_head: optional trained hspec.exit.ExitHead (kind 'lin') for the exit check."""
    costs = costs or Costs()
    device = input_ids.device
    bs = draft.block_size
    k = cfg.exit_layer
    tg = SplitTarget(target, draft.target_layer_ids)
    L = tg.L
    lower, upper = range(0, k), range(k, L)
    n = input_ids.shape[1]
    max_len = n + max_new_tokens
    size = max_len + bs + 2
    stop = torch.tensor(stop_token_ids, device=device) if stop_token_ids else None
    H = target.config.hidden_size
    W = H * len(tg.feat_ids)
    ids = torch.full((1, size), draft.mask_token_id, dtype=torch.long, device=device)
    ids[:, :n] = input_ids
    feats = torch.zeros((1, size, W), dtype=target.dtype, device=device)
    hk = torch.zeros((1, size, H), dtype=target.dtype, device=device)
    deep = deep_columns(draft, cfg.feat_exit or k, W, device)
    col = {lid: j for j, lid in enumerate(tg.feat_ids)}
    dcache = _make_cache(draft.config)
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    head = _output_head(target)

    def put_feats(fd, p0, rows=None):
        for lid, h in fd.items():
            j = col[lid]
            x = h[0] if rows is None else h[0, rows]
            feats[0, p0 : p0 + x.shape[0], j * H : (j + 1) * H] = x

    res = GenResult(ids, n, 0.0)
    st = SplitStats()
    sim_ms = 0.0
    t0 = _sync_time()

    def exit_logits(h):
        return exit_head.logits(h) if exit_head is not None else tg.head(h)

    # prefill: all layers
    pos = torch.arange(n, device=device)[None]
    h, fd = tg.run(0, L, tg.embed(input_ids), pos, _causal(0, n, device))
    put_feats(fd, 0)
    ids[0, n] = tg.head(h[:, -1:])[0, -1].argmax()
    res.target_calls = 1
    F = E = n                      # upper KV = [0, F), lower KV = [0, E); tokens <= F are final
    anchor = int(ids[0, n])        # token at E (in no KV yet)
    tentative = False              # anchor is the exit's unsure guess: verify before building on it
    dlen = 0                       # drafter cache holds [0, dlen) (all final features)
    since = 0
    final = None
    if _first_stop(ids[0, n : n + 1], stop) is not None:
        final = n + 1

    def draft_tree():
        """Drafter tree at anchor E; context features [dlen, E), shallow-only beyond F."""
        nonlocal dlen, sim_ms
        vs = min(bs, max_len - E)
        block = torch.full((1, vs), draft.mask_token_id, dtype=torch.long, device=device)
        block[0, 0] = anchor
        noise = _raw_input_embeddings(target, block, scale)
        ctx = feats[:, dlen:E].clone()
        ctx[:, F - dlen :, deep] = 0
        dpos = torch.arange(dlen, E + vs, device=device)[None]
        dh = draft(target_hidden=ctx, noise_embedding=noise, position_ids=dpos, past_key_values=dcache,
                   use_cache=True)[:, 1 - vs :, :]
        crop(dcache, F)
        dlen = F
        logits = draft.compute_logits(dh, head)[0]
        res.draft_calls += 1
        sim_ms += costs.draft_ms
        tree = (build_ddtree(logits, cfg.budget, top_k=cfg.top_k) if cfg.budget > 0
                else chain(logits.argmax(-1).tolist()))
        return tree, logits.argmax(-1).tolist()

    def finish_check(old_F):
        nonlocal final
        hit = _first_stop(ids[0, old_F + 1 : F + 1], stop)
        if hit is not None:
            final = old_F + 1 + hit + 1
        elif F + 1 >= max_len:
            final = max_len

    def settle_pending(pred, fd_up, P):
        """Compare pending tokens [F+1, E) and the anchor at E with the full model's predictions
        pred[0 .. P-1] (for positions F+1 .. E). Returns True if everything, anchor included,
        is confirmed; otherwise rolls back / replaces the anchor and returns False."""
        nonlocal F, E, anchor, tentative
        known = ids[0, F + 1 : E]
        a = int((known == pred[: P - 1]).long().cumprod(0).sum()) if P > 1 else 0
        if a < P - 1:                               # mismatch inside the pending tokens at p
            p = F + 1 + a
            ids[0, p] = pred[a]
            put_feats(fd_up, F, rows=list(range(p - F)))
            st.rollbacks += 1
            st.rollback_tokens += E - p
            _crop_layers(tg.cache, upper, p)
            _crop_layers(tg.cache, lower, p)
            F = E = p
            anchor, tentative = int(pred[a]), False
            return False
        t = int(pred[P - 1])
        if t != anchor:                             # pending tokens fine, anchor wrong
            if not tentative:
                st.rollbacks += 1
            put_feats(fd_up, F, rows=list(range(P)))
            _crop_layers(tg.cache, upper, E)
            _crop_layers(tg.cache, lower, E)
            ids[0, E] = t
            F = E
            anchor, tentative = t, False
            return False
        return True

    def upper_pass():
        """Verify positions [F, E) and the anchor at E with layers k..L (nothing new drafted)."""
        nonlocal F, E, anchor, sim_ms, since, tentative
        P = E - F
        since = 0
        if P == 0:
            tentative = False
            return
        old_F = F
        pos = torch.arange(F, E, device=device)[None]
        h, fd = tg.run(k, L, hk[:, F:E], pos, _causal(F, P, device))
        pred = tg.head(h)[0].argmax(-1)
        st.upper_passes += 1
        st.upper_q.append(P)
        res.target_calls += 1
        sim_ms += cfg.r_up * costs.t(P)
        if settle_pending(pred, fd, P):
            put_feats(fd, F, rows=list(range(P)))
            F = E
            tentative = False
        finish_check(old_F)

    def merged_pass():
        """Upper pass fused with a full verification of a new drafter tree at the anchor:
        lower layers on the tree, then layers k..L on [pending positions + tree] at once."""
        nonlocal F, E, anchor, sim_ms, since, tentative
        since = 0
        old_F = F
        P = E - F
        tree, _ = draft_tree()
        T = len(tree)
        tids, tpos = tree_inputs(anchor, tree, E, device)
        ht, fd_low = tg.run(0, k, tg.embed(tids), tpos, tree_mask(tree, E, device))
        q = P + 1 + T
        m = torch.zeros((q, F + q), dtype=torch.bool, device=device)
        m[:, :F] = True
        if P:
            m[:P, F : F + P] = torch.ones((P, P), dtype=torch.bool, device=device).tril()
            m[P:, F : F + P] = True
        m[P:, F + P :] = tree_mask(tree, 0, device)[0, 0]
        hin = torch.cat([hk[:, F:E], ht], dim=1)
        pos = torch.cat([torch.arange(F, E, device=device)[None], tpos], dim=1)
        h, fd_up = tg.run(k, L, hin, pos, m[None, None])
        lg = tg.head(h)
        pred = lg[0].argmax(-1)
        st.upper_passes += 1
        st.merged += 1
        st.upper_q.append(q)
        res.target_calls += 1
        sim_ms += cfg.r_lower * costs.t(1 + T) + cfg.r_up * costs.t(q)
        if P and not settle_pending(pred, fd_up, P):
            _crop_layers(tg.cache, lower, E)       # drop the tree rows (E already reset)
            finish_check(old_F)
            return
        path, bonus = greedy_walk(tree, lg[:, P:])
        rows = [0] + [nd + 1 for nd in path]
        _compact_layers(tg.cache, lower, E, rows)
        _compact_layers(tg.cache, upper, F, list(range(P)) + [P + r for r in rows])
        put_feats(fd_up, F, rows=list(range(P)) + [P + r for r in rows])
        put_feats(fd_low, E, rows=rows)
        ids[0, E] = anchor
        if path:
            ids[0, E + 1 : E + 1 + len(path)] = torch.tensor([tree.tokens[nd] for nd in path], device=device)
        E = F = E + len(rows)
        anchor, tentative = int(bonus), False
        if E < size:
            ids[0, E] = anchor
        st.merged_accepted.append(len(path))
        finish_check(old_F)

    while final is None:
        if F == E and not tentative and F + 1 >= max_len:
            final = max_len
            break
        pending_stop = _first_stop(ids[0, F + 1 : E + 1], stop) is not None
        near_end = E + 2 >= max_len or min(bs, max_len - E) < 2
        if tentative or since >= cfg.m or E - F >= cfg.max_pending or pending_stop or (E > F and near_end):
            if tentative:
                st.unsure_flushes += 1
            if cfg.merge and not pending_stop and not near_end:
                merged_pass()
            else:
                upper_pass()
            continue

        # ---- one cheap round: draft, then the exit check (lower layers + exit head) on the tree
        tree, dtop = draft_tree()
        tids, tpos = tree_inputs(anchor, tree, E, device)
        h, fd = tg.run(0, k, tg.embed(tids), tpos, tree_mask(tree, E, device))
        pr = torch.softmax(exit_logits(h)[0].float(), -1)
        p1, top = pr.max(-1)
        p1, top = p1.tolist(), top.tolist()
        st.rounds += 1
        since += 1
        sim_ms += cfg.r_exit * costs.t(1 + len(tree))
        res.mid_calls += 1
        ch = tree.children()
        path, cur, row, depth = [], -1, 0, 0
        while True:
            t, conf = top[row], p1[row]
            joint = cfg.thr_joint > 0 and depth < len(dtop) and t == dtop[depth] and conf >= cfg.thr_joint
            nd = ch.get(cur, {}).get(t)
            if conf < cfg.thr and not joint:
                nxt, unsure = t, True               # unsure: keep its guess, verify it next
                break
            if nd is None:
                nxt, unsure = t, False              # confident correction
                break
            path.append(nd)
            cur, row, depth = nd, nd + 1, depth + 1
        rows = [0] + [nd + 1 for nd in path]
        _compact_layers(tg.cache, lower, E, rows)
        hk[0, E : E + len(rows)] = h[0, rows]
        put_feats(fd, E, rows=rows)
        feats[0, E : E + len(rows)][:, deep] = 0
        ids[0, E] = anchor
        if path:
            ids[0, E + 1 : E + 1 + len(path)] = torch.tensor([tree.tokens[nd] for nd in path], device=device)
        E = E + len(rows)
        anchor, tentative = nxt, unsure
        st.exit_accepted.append(len(path))
        if E < size:
            ids[0, E] = anchor

    res.decode_time = _sync_time() - t0
    res.output_ids = ids[:, : min(final, max_len)]
    res.sim_ms = sim_ms
    res.split_stats = st
    return res


__all__ = ["SplitConfig", "split_generate"]
