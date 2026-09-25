"""Three-stage decoding with an asynchronous target, on a simulated clock.

The lower stages (drafter + middle model, "device A") and the target ("device B", possibly
behind a network link) run concurrently in a real deployment. Here the code runs them in
sequence, but every event carries a virtual timestamp built from measured per-call costs,
and a target result only takes effect once its virtual arrival time has passed. Call
counts and outputs are exact; speed comes from the virtual clock.

Target check timing: a check of q tokens submitted at time s starts at
max(s + L/2, target free), takes c_T(q), and its result arrives L/2 later.

blocking=True reproduces the synchronous three-stage decoder (the lower stages wait for
every check); blocking=False lets them keep drafting on the assumption that pending tokens
will be accepted, with several checks in flight. When a check rejects at position p, all
later checks are dropped (not-yet-started ones free the target), the caches are cropped
to p and decoding continues from the target's token at p.

Losslessness (greedy): a token is final only once the target has confirmed it, and every
confirmed token is the target's argmax given the confirmed prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dflash.model import _make_cache, extract_context_feature

from hspec.pipeline import GenResult, _first_stop, crop, draft_logits, propose, _greedy_accept
from hspec.tree import Tree, build_ddtree, compact, greedy_walk, verify_tree


def _interp(table: dict[int, float], q: float) -> float:
    ks = sorted(table)
    if q <= ks[0]:
        return table[ks[0]]
    if q >= ks[-1]:
        return table[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= q <= b:
            w = (q - a) / (b - a)
            return table[a] * (1 - w) + table[b] * w
    return table[ks[-1]]


@dataclass
class Costs:
    """Per-call costs in ms (tables are query-length -> ms, interpolated)."""
    target: dict[int, float] = field(default_factory=lambda: {1: 12.99, 17: 13.19, 33: 13.77, 65: 15.14,
                                                              129: 17.68, 257: 26.31, 513: 46.37})
    mid: dict[int, float] = field(default_factory=lambda: {1: 5.81, 17: 5.81, 33: 5.81, 65: 7.13,
                                                           129: 11.69, 257: 21.70, 513: 43.14})
    draft_ms: float = 3.55
    latency_ms: float = 0.0

    def t(self, q):
        return _interp(self.target, q)

    def m(self, q):
        return _interp(self.mid, q)


@dataclass
class _Check:
    c0: int
    c1: int
    preds: torch.Tensor      # target argmax for positions c0+1 .. c1+1
    feats: torch.Tensor      # [1, c1-c0+1, W] target features for positions c0 .. c1
    start_t: float
    end_t: float
    done_t: float


@torch.inference_mode()
def async_three_stage_generate(
    draft,
    target,
    mid,
    input_ids,
    max_new_tokens,
    stop_token_ids,
    costs: Costs,
    window: int = 16,
    blocking: bool = False,
    mid_tree: int = 0,
    max_inflight: int = 64,
    max_ahead: int = 256,
    block_size: int | None = None,
) -> GenResult:
    device = input_ids.device
    bs = draft.block_size if block_size is None else block_size
    n = input_ids.shape[1]
    max_len = n + max_new_tokens
    size = max_len + bs + 2
    output_ids = torch.full((1, size), draft.mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(size, device=device).unsqueeze(0)
    stop = torch.tensor(stop_token_ids, device=device) if stop_token_ids else None
    L = costs.latency_ms

    tcache = _make_cache(target.config)
    mcache = _make_cache(mid.config)
    dcache = _make_cache(draft.config)

    out = target(input_ids, position_ids=position_ids[:, :n], past_key_values=tcache,
                 use_cache=True, logits_to_keep=1)
    output_ids[:, :n] = input_ids
    output_ids[0, n] = out.logits[0, -1].argmax()
    mout = mid(input_ids, position_ids=position_ids[:, :n], past_key_values=mcache,
               use_cache=True, logits_to_keep=1, output_hidden_states=True)
    f0 = extract_context_feature(mout.hidden_states, draft.target_layer_ids)
    feat_buf = torch.zeros((1, size, f0.shape[-1]), dtype=f0.dtype, device=device)
    feat_buf[:, :n] = f0      # drafter context features per position (best available)

    res = GenResult(output_ids, n, 0.0)
    res.target_calls = 1
    st = {"tf": n, "tsub": n, "start": n, "mlen": n, "dlen": 0, "now": 0.0, "busy": 0.0,
          "done": False, "final": None}
    inflight: list[_Check] = []
    stats = {"rollbacks": 0, "wasted_target_calls": 0, "cancelled": 0, "lower_wait_ms": 0.0,
             "waste_tokens": 0}

    if _first_stop(output_ids[0, n : n + 1], stop) is not None:
        st["done"], st["final"] = True, n + 1

    def submit():
        c0, c1 = st["tsub"], st["start"]
        tout = target(output_ids[:, c0 : c1 + 1], position_ids=position_ids[:, c0 : c1 + 1],
                      past_key_values=tcache, use_cache=True, output_hidden_states=True)
        q = c1 - c0 + 1
        start_t = max(st["now"] + L / 2, st["busy"])
        end_t = start_t + costs.t(q)
        st["busy"] = end_t
        inflight.append(_Check(c0, c1, tout.logits[0].argmax(-1),
                               extract_context_feature(tout.hidden_states, draft.target_layer_ids),
                               start_t, end_t, end_t + L / 2))
        st["tsub"] = c1 + 1
        res.target_calls += 1
        res.tq.append(q)

    def rollback(p, now):
        """Token at p was just set by the target; everything after p is discarded."""
        stats["rollbacks"] += 1
        stats["waste_tokens"] += st["start"] - p
        started = [c.end_t for c in inflight if c.start_t <= now]
        stats["wasted_target_calls"] += len(inflight)
        stats["cancelled"] += sum(c.start_t > now for c in inflight)
        inflight.clear()
        st["busy"] = max(started + [min(st["busy"], now)])   # cancel checks that have not started
        st["start"] = p
        st["tsub"] = p
        crop(tcache, p)
        st["mlen"] = min(st["mlen"], p)
        crop(mcache, st["mlen"])
        st["dlen"] = max(min(st["dlen"], p - 1), 0)
        crop(dcache, st["dlen"])

    def process(ck: _Check):
        c0, c1, start = ck.c0, ck.c1, st["start"]
        assert st["tf"] == c0, (st["tf"], c0)
        hi = min(c1 + 1, start)
        cmp = output_ids[0, c0 + 1 : hi + 1] == ck.preds[: hi - c0]
        a = int(cmp.long().cumprod(0).sum())
        rows = min(a + 1, c1 - c0 + 1)
        feat_buf[:, c0 : c0 + rows] = ck.feats[:, :rows]
        old_tf = st["tf"]
        if a < hi - c0:                       # target disagrees at p
            p = c0 + 1 + a
            output_ids[0, p] = ck.preds[a]
            rollback(p, ck.done_t)
            new_tf = p
        elif hi == c1 + 1:                    # all verified, including the token after c1
            new_tf = c1 + 1
        elif c1 + 1 < max_len:                # nothing drafted past c1 yet: take the target's token
            output_ids[0, c1 + 1] = ck.preds[-1]
            st["start"] = c1 + 1
            new_tf = c1 + 1
        else:
            new_tf = c1
        st["tf"] = new_tf
        res.checks.append({"pending": c1 - c0, "accepted": min(a, c1 - c0)})
        hit = _first_stop(output_ids[0, old_tf + 1 : new_tf + 1], stop)
        if hit is not None:
            st["done"], st["final"] = True, old_tf + 1 + hit + 1
        elif new_tf + 1 >= max_len:
            st["done"], st["final"] = True, max_len

    def lower_round():
        start, mlen, dlen = st["start"], st["mlen"], st["dlen"]
        vs = min(bs, max_len - start)
        offset = start - mlen
        ctx = feat_buf[:, dlen:start]
        cost = 0.0
        if mid_tree > 0 and vs > 1:
            logits = draft_logits(draft, target, ctx, output_ids[:, start : start + vs],
                                  position_ids, start, dcache)
            st["dlen"] = start
            res.draft_calls += 1
            cost += costs.draft_ms
            dtree = build_ddtree(logits, mid_tree)
            tree = Tree()
            for i, tok in enumerate(output_ids[0, mlen + 1 : start + 1].tolist()):
                tree.add(tok, i - 1)
            anchor_node, base = offset - 1, len(tree)
            for tok, par in zip(dtree.tokens, dtree.parents):
                tree.add(tok, anchor_node if par < 0 else base + par)
            mo = verify_tree(mid, mcache, output_ids[0, mlen], tree, mlen, hidden=True)
            q = 1 + len(tree)
            path, bonus = greedy_walk(tree, mo.logits, start=anchor_node)
            a = len(path)
            if a:
                output_ids[0, start + 1 : start + 1 + a] = torch.tensor(
                    [tree.tokens[nd] for nd in path], device=device)
            output_ids[0, start + a + 1] = bonus
            produced = min(a + 1, max_len - start - 1)
            rows = ([offset] + [nd + 1 for nd in path])[:produced]
            compact(mcache, mlen, list(range(offset)) + rows)
            feats = extract_context_feature(mo.hidden_states, draft.target_layer_ids)[:, rows]
        else:
            block = output_ids[:, start : start + vs].clone()
            if vs > 1:
                block = propose(draft, target, ctx, block, position_ids, start, dcache)
                st["dlen"] = start
                res.draft_calls += 1
                cost += costs.draft_ms
            mo = mid(torch.cat([output_ids[:, mlen:start], block], dim=1),
                     position_ids=position_ids[:, mlen : start + vs], past_key_values=mcache,
                     use_cache=True, output_hidden_states=True)
            q = offset + vs
            a, bonus = _greedy_accept(block, mo.logits[:, offset:])
            output_ids[:, start : start + a + 1] = block[:, : a + 1]
            output_ids[0, start + a + 1] = bonus
            produced = min(a + 1, max_len - start - 1)
            feats = extract_context_feature(mo.hidden_states, draft.target_layer_ids)
            feats = feats[:, offset : offset + produced]
        res.mid_calls += 1
        res.mq.append(q)
        cost += costs.m(q)
        # keep target features for positions the target already confirmed
        keep_from = max(start, st["tf"] + 1)
        if keep_from < start + produced:
            feat_buf[:, keep_from : start + produced] = feats[:, keep_from - start :]
        st["mlen"] = start + produced
        crop(mcache, st["mlen"])
        st["start"] = start + produced
        res.rounds.append(produced)
        st["now"] += cost

    while not st["done"]:
        while inflight and inflight[0].done_t <= st["now"] and not st["done"]:
            process(inflight.pop(0))
        if st["done"]:
            break
        start, tf = st["start"], st["tf"]
        stop_pending = _first_stop(output_ids[0, tf + 1 : start + 1], stop) is not None
        blocked = start + 1 >= max_len or stop_pending or start - tf >= max_ahead
        unsub = start + 1 - st["tsub"]
        # tokens this check would newly verify (c0 itself is covered by the previous check)
        nver = start - max(st["tsub"] - 1, tf)
        if unsub > 0 and len(inflight) < max_inflight and (nver >= window or blocked):
            submit()
            if blocking:
                blocked = True
        if blocked or (blocking and inflight):
            if not inflight:
                raise RuntimeError("blocked with nothing in flight")
            wait = max(inflight[0].done_t - st["now"], 0.0)
            stats["lower_wait_ms"] += wait
            st["now"] += wait
            continue
        lower_round()

    res.decode_time = st["now"] / 1000.0
    res.output_ids = output_ids[:, : min(st["final"], max_len)]
    res.async_stats = stats
    return res
