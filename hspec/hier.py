"""Hierarchical speculative decoding on a simulated clock: one decoder, several algorithms.

Device A runs the drafter and the middle model (mid); device B runs the target. Every
event carries a virtual timestamp from measured per-call costs (see hspec.async3 for the
clock model: a check of q tokens starts at max(submit + L/2, target free), takes c_T(q),
arrives L/2 later). Outputs and call counts are exact; speed comes from the clock.

Algorithms (combinable):

  window check   submit pending tokens to the target once `window` are unsubmitted.
  conf check (C) also submit early once the product of mid's confidence over the
                 unsubmitted tokens drops below `tau` (and at least `min_window` wait).
  blocking       the lower stages wait for every check (synchronous three-stage).
  hedging (A)    at positions where mid's top-1 probability is below `fork_thr`, fork an
                 alternative branch that starts with mid's second choice. Live branches
                 are extended together with the main branch in the same batched drafter /
                 mid forward (memory-bound, so nearly free; the clock charges the batched
                 call). If the target rejects the main token at a fork position and picks
                 that branch's token, the branch becomes the main branch and its progress is
                 kept instead of being redone after a rollback.
  pearl          no mid: the drafter's own tokens are sent to the target unverified, and the
                 drafter only has target features for confirmed positions (feature lag),
                 i.e. PEARL-style asynchronous two-stage decoding with a DFlash drafter.

Greedy losslessness: every confirmed token is the target's argmax given the confirmed
prefix; branches only change which unconfirmed tokens get proposed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from dflash.model import (
    _draft_value,
    _make_cache,
    _output_head,
    _raw_input_embeddings,
    extract_context_feature,
)

from hspec.async3 import Costs, _Check
from hspec.pipeline import GenResult, _first_stop, _greedy_accept, crop, draft_logits, propose
from hspec.tree import Tree, build_ddtree, compact, greedy_walk, verify_tree


@dataclass
class HierConfig:
    window: int = 16            # max unsubmitted tokens before a check
    check_rule: str = "window"  # "window" | "conf"
    tau: float = 0.5            # conf rule: submit when prod(conf) < tau ...
    min_window: int = 4         # ... and at least this many tokens are unsubmitted
    blocking: bool = False
    mid_tree: int = 0
    max_branches: int = 0       # hedging: live alternative branches (0 = off)
    fork_thr: float = 0.6       # fork where mid's top-1 probability is below this
    alt_tree: int = 0           # mid tree budget on alternative branches (0 = plain chain)
    pearl: bool = False         # two-stage async, drafter tokens sent unverified
    pearl_len: int = 8
    draft_batch_beta: float = 0.15   # drafter cost at batch n: c_D * (1 + beta (n - 1))
    max_inflight: int = 64
    max_ahead: int = 256


@dataclass
class _Branch:
    ids: torch.Tensor
    feat: torch.Tensor
    conf: torch.Tensor
    alt: torch.Tensor
    mcache: object
    dcache: object
    start: int
    mlen: int
    dlen: int
    fork: int = -1


def _clone_cache(cache, length):
    if cache is None:
        return None
    c = copy.deepcopy(cache)
    crop(c, length)
    return c


@torch.inference_mode()
def _draft_lagged(draft, head_src, feat, ctx_end, anchor, anchor_tok, bs):
    """Drafter logits [bs-1, V] for a block at `anchor`, features only for [0, ctx_end)."""
    device = feat.device
    block = torch.full((1, bs), draft.mask_token_id, dtype=torch.long, device=device)
    block[0, 0] = anchor_tok
    noise = _raw_input_embeddings(head_src, block,
                                  float(_draft_value(draft.config, "input_embedding_scale", 1.0)))
    pos = torch.cat([torch.arange(ctx_end, device=device),
                     torch.arange(anchor, anchor + bs, device=device)]).unsqueeze(0)
    hidden = draft(target_hidden=feat[:, :ctx_end], noise_embedding=noise, position_ids=pos,
                   past_key_values=None, use_cache=False)[:, 1 - bs :, :]
    return draft.compute_logits(hidden, _output_head(head_src))[0]


@torch.inference_mode()
def hier_generate(draft, target, mid, input_ids, max_new_tokens, stop_token_ids,
                  costs: Costs, cfg: HierConfig, block_size: int | None = None) -> GenResult:
    device = input_ids.device
    bs = draft.block_size if block_size is None else block_size
    n = input_ids.shape[1]
    max_len = n + max_new_tokens
    size = max_len + bs + 2
    stop = torch.tensor(stop_token_ids, device=device) if stop_token_ids else None
    position_ids = torch.arange(size, device=device).unsqueeze(0)
    L = costs.latency_ms
    use_mid = not cfg.pearl

    ids = torch.full((1, size), draft.mask_token_id, dtype=torch.long, device=device)
    ids[:, :n] = input_ids
    tcache = _make_cache(target.config)
    out = target(input_ids, position_ids=position_ids[:, :n], past_key_values=tcache, use_cache=True,
                 logits_to_keep=1, output_hidden_states=not use_mid)
    ids[0, n] = out.logits[0, -1].argmax()
    if use_mid:
        mcache = _make_cache(mid.config)
        mo = mid(input_ids, position_ids=position_ids[:, :n], past_key_values=mcache, use_cache=True,
                 logits_to_keep=1, output_hidden_states=True)
        f0 = extract_context_feature(mo.hidden_states, draft.target_layer_ids)
    else:
        mcache = None
        f0 = extract_context_feature(out.hidden_states, draft.target_layer_ids)
    feat = torch.zeros((1, size, f0.shape[-1]), dtype=f0.dtype, device=device)
    feat[:, :n] = f0
    main = _Branch(ids, feat, torch.ones(size, device=device), torch.zeros(size, dtype=torch.long, device=device),
                   mcache, _make_cache(draft.config), start=n, mlen=n, dlen=0)
    alts: list[_Branch] = []

    res = GenResult(ids, n, 0.0)
    res.target_calls = 1
    st = {"tf": n, "tsub": n, "now": 0.0, "busy": 0.0, "done": False, "final": None}
    inflight: list[_Check] = []
    stats = {"rollbacks": 0, "caught": 0, "forks": 0, "branch_rounds": 0, "waste_tokens": 0,
             "wasted_target_calls": 0, "lower_wait_ms": 0.0, "lower_steps": 0}
    if _first_stop(ids[0, n : n + 1], stop) is not None:
        st["done"], st["final"] = True, n + 1

    # ------------------------------------------------------------------ target side
    def submit():
        c0, c1 = st["tsub"], main.start
        tout = target(main.ids[:, c0 : c1 + 1], position_ids=position_ids[:, c0 : c1 + 1],
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

    def drop_inflight(p, now):
        started = [c.end_t for c in inflight if c.start_t <= now]
        stats["wasted_target_calls"] += len(inflight)
        inflight.clear()
        st["busy"] = max(started + [min(st["busy"], now)])
        st["tsub"] = p
        crop(tcache, p)

    def rollback(p, now):
        stats["rollbacks"] += 1
        stats["waste_tokens"] += main.start - p
        drop_inflight(p, now)
        main.start = p
        if use_mid:
            main.mlen = min(main.mlen, p)
            crop(main.mcache, main.mlen)
        main.dlen = max(min(main.dlen, p - 1), 0)
        crop(main.dcache, main.dlen)
        alts.clear()

    def process(ck: _Check):
        nonlocal main
        c0, c1 = ck.c0, ck.c1
        assert st["tf"] == c0, (st["tf"], c0)
        hi = min(c1 + 1, main.start)
        cmp = main.ids[0, c0 + 1 : hi + 1] == ck.preds[: hi - c0]
        a = int(cmp.long().cumprod(0).sum())
        rows = min(a + 1, c1 - c0 + 1)
        main.feat[:, c0 : c0 + rows] = ck.feats[:, :rows]
        old_tf = st["tf"]
        if a < hi - c0:
            p, tok = c0 + 1 + a, int(ck.preds[a])
            hit = next((b for b in alts if b.fork == p and int(b.ids[0, p]) == tok), None)
            if hit is not None:
                stats["caught"] += 1
                stats["waste_tokens"] += main.start - p
                hit.feat[:, c0 : c0 + rows] = ck.feats[:, :rows]
                drop_inflight(p, ck.done_t)
                main = hit
                alts.clear()
            else:
                main.ids[0, p] = tok
                rollback(p, ck.done_t)
            new_tf = p
        elif hi == c1 + 1:
            new_tf = c1 + 1
        elif c1 + 1 < max_len:
            main.ids[0, c1 + 1] = ck.preds[-1]
            main.start = c1 + 1
            new_tf = c1 + 1
        else:
            new_tf = c1
        st["tf"] = new_tf
        alts[:] = [b for b in alts if b.fork > new_tf]
        res.checks.append({"pending": c1 - c0, "accepted": min(a, c1 - c0)})
        stop_hit = _first_stop(main.ids[0, old_tf + 1 : new_tf + 1], stop)
        if stop_hit is not None:
            st["done"], st["final"] = True, old_tf + 1 + stop_hit + 1
        elif new_tf + 1 >= max_len:
            st["done"], st["final"] = True, max_len

    # ------------------------------------------------------------------ lower side
    def mid_round(b: _Branch, budget: int | None = None):
        """One draft + mid round on branch b. Returns (drafted, mid query tokens, produced)."""
        budget = cfg.mid_tree if budget is None else budget
        start, mlen, dlen = b.start, b.mlen, b.dlen
        vs = min(bs, max_len - start)
        offset = start - mlen
        ctx = b.feat[:, dlen:start]
        drafted = vs > 1
        if budget > 0 and vs > 1:
            logits = draft_logits(draft, target, ctx, b.ids[:, start : start + vs], position_ids, start, b.dcache)
            b.dlen = start
            dtree = build_ddtree(logits, budget)
            tree = Tree()
            for i, tok in enumerate(b.ids[0, mlen + 1 : start + 1].tolist()):
                tree.add(tok, i - 1)
            anchor_node, base = offset - 1, len(tree)
            for tok, par in zip(dtree.tokens, dtree.parents):
                tree.add(tok, anchor_node if par < 0 else base + par)
            mo = verify_tree(mid, b.mcache, b.ids[0, mlen], tree, mlen, hidden=True)
            q = 1 + len(tree)
            path, bonus = greedy_walk(tree, mo.logits, start=anchor_node)
            a = len(path)
            if a:
                b.ids[0, start + 1 : start + 1 + a] = torch.tensor([tree.tokens[nd] for nd in path], device=device)
            b.ids[0, start + a + 1] = bonus
            produced = min(a + 1, max_len - start - 1)
            rows = ([offset] + [nd + 1 for nd in path])[:produced]
            compact(b.mcache, mlen, list(range(offset)) + rows)
            lrows = mo.logits[0, rows]
            feats = extract_context_feature(mo.hidden_states, draft.target_layer_ids)[:, rows]
        else:
            block = b.ids[:, start : start + vs].clone()
            if vs > 1:
                block = propose(draft, target, ctx, block, position_ids, start, b.dcache)
                b.dlen = start
            mo = mid(torch.cat([b.ids[:, mlen:start], block], dim=1), position_ids=position_ids[:, mlen : start + vs],
                     past_key_values=b.mcache, use_cache=True, output_hidden_states=True)
            q = offset + vs
            a, bonus = _greedy_accept(block, mo.logits[:, offset:])
            b.ids[:, start : start + a + 1] = block[:, : a + 1]
            b.ids[0, start + a + 1] = bonus
            produced = min(a + 1, max_len - start - 1)
            lrows = mo.logits[0, offset : offset + produced]
            feats = extract_context_feature(mo.hidden_states, draft.target_layer_ids)[:, offset : offset + produced]
        top = torch.softmax(lrows.float(), dim=-1).topk(2, dim=-1)
        b.conf[start + 1 : start + 1 + produced] = top.values[:, 0]
        b.alt[start + 1 : start + 1 + produced] = top.indices[:, 1]
        keep_from = max(start, st["tf"] + 1)
        if keep_from < start + produced:
            b.feat[:, keep_from : start + produced] = feats[:, keep_from - start :]
        b.mlen = start + produced
        crop(b.mcache, b.mlen)
        b.start = start + produced
        res.mq.append(q)
        return drafted, q, produced

    def pearl_round(b: _Branch):
        start = b.start
        k = min(cfg.pearl_len, bs - 1, max_len - start - 1)
        logits = _draft_lagged(draft, target, b.feat, st["tf"], start, b.ids[0, start], bs).float()
        b.ids[0, start + 1 : start + 1 + k] = logits[:k].argmax(-1)
        b.conf[start + 1 : start + 1 + k] = torch.softmax(logits[:k], -1).max(-1).values
        b.start = start + k
        return True, 0, k

    def fork_from(p):
        tok = int(main.alt[p])
        b = _Branch(main.ids.clone(), main.feat.clone(), main.conf.clone(), main.alt.clone(),
                    _clone_cache(main.mcache, min(main.mlen, p)), _clone_cache(main.dcache, max(min(main.dlen, p - 1), 0)),
                    start=p, mlen=min(main.mlen, p), dlen=max(min(main.dlen, p - 1), 0), fork=p)
        b.ids[0, p] = tok
        b.ids[0, p + 1 :] = draft.mask_token_id
        b.conf[p] = 1.0
        alts.append(b)
        stats["forks"] += 1

    def lower_step():
        old_start = main.start
        drafted, q, produced = (pearl_round if cfg.pearl else mid_round)(main)
        n_draft, q_total = int(drafted), q
        res.rounds.append(produced)
        for b in list(alts):
            if b.start >= main.start or b.start - b.fork >= cfg.max_ahead or b.start + 1 >= max_len:
                continue
            if _first_stop(b.ids[0, b.fork : b.start + 1], stop) is not None:
                continue
            d, qb, _ = mid_round(b, cfg.alt_tree)
            n_draft += int(d)
            q_total += qb
            stats["branch_rounds"] += 1
        cost = 0.0
        if n_draft:
            cost += costs.draft_ms * (1 + cfg.draft_batch_beta * (n_draft - 1))
            res.draft_calls += 1
        if use_mid:
            cost += costs.m(q_total)
            res.mid_calls += 1
        st["now"] += cost
        stats["lower_steps"] += 1
        # hedging: fork at every new position below the threshold, least confident first
        if cfg.max_branches > 0 and use_mid and len(alts) < cfg.max_branches:
            lo = max(old_start + 1, st["tf"] + 1)
            hi = main.start
            if hi >= lo:
                c = main.conf[lo : hi + 1].tolist()
                for j in sorted(range(len(c)), key=c.__getitem__):
                    p = lo + j
                    if c[j] >= cfg.fork_thr or len(alts) >= cfg.max_branches:
                        break
                    if all(b.fork != p for b in alts) and p + 1 < max_len:
                        fork_from(p)

    # ------------------------------------------------------------------ main loop
    while not st["done"]:
        while inflight and inflight[0].done_t <= st["now"] and not st["done"]:
            process(inflight.pop(0))
        if st["done"]:
            break
        start, tf = main.start, st["tf"]
        stop_pending = _first_stop(main.ids[0, tf + 1 : start + 1], stop) is not None
        blocked = start + 1 >= max_len or stop_pending or start - tf >= cfg.max_ahead
        unsub = start + 1 - st["tsub"]
        seg0 = max(st["tsub"] - 1, tf) + 1
        nver = start + 1 - seg0
        want = nver >= cfg.window or blocked
        if cfg.check_rule == "conf" and nver >= cfg.min_window:
            want = want or float(main.conf[seg0 : start + 1].prod()) < cfg.tau
        if unsub > 0 and len(inflight) < cfg.max_inflight and want:
            submit()
            if cfg.blocking:
                blocked = True
        if blocked or (cfg.blocking and inflight):
            if not inflight:
                raise RuntimeError("blocked with nothing in flight")
            wait = max(inflight[0].done_t - st["now"], 0.0)
            stats["lower_wait_ms"] += wait
            st["now"] += wait
            continue
        lower_step()

    res.decode_time = st["now"] / 1000.0
    res.output_ids = main.ids[:, : min(st["final"], max_len)]
    res.async_stats = stats
    return res
