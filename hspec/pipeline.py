"""Greedy generation loops.

Three decoders, all greedy (temperature 0), all built on the DFlash block-diffusion drafter:

* ``ar_generate``          plain autoregressive decoding with the target (speed baseline).
* ``two_stage_generate``   DFlash speculative decoding with a chosen verifier ``V`` and a
                           chosen feature model ``F`` (whose hidden states condition the
                           drafter). F = V = target reproduces vanilla DFlash.
* ``three_stage_generate`` drafter -> middle verifier (standard DFlash rounds, drafter
                           conditioned on the middle model's hidden states) -> the target
                           checks all pending tokens in one pass once a window is full.

Greedy three-stage decoding is lossless by construction: every emitted token is the
target's argmax given the emitted prefix.

Cache convention used throughout: a model whose cache holds ``L`` positions has processed
tokens ``[0, L)``. The drafter cache holds the injected context features of positions
``[0, dlen)``; ``ctx`` always holds features for ``[dlen, anchor)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from dflash.model import (
    DFlash2DraftModel,
    _draft_value,
    _make_cache,
    _output_head,
    _raw_input_embeddings,
    extract_context_feature,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cache_len(cache) -> int:
    return cache.get_seq_length()


def crop(cache, length: int) -> None:
    """Keep positions [0, length). No-op if the cache is already that short."""
    remove = _cache_len(cache) - length
    if remove > 0:
        cache.crop(-remove)   # negative = drop that many tokens (positive is deprecated in transformers 5.18)


def _sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _first_stop(tokens: torch.Tensor, stop: torch.Tensor | None) -> int | None:
    """Index of the first stop token in a 1-D tensor, or None."""
    if stop is None or tokens.numel() == 0:
        return None
    hit = torch.isin(tokens, stop).nonzero(as_tuple=True)[0]
    return int(hit[0]) if hit.numel() else None


def _greedy_accept(block: torch.Tensor, logits: torch.Tensor) -> tuple[int, torch.Tensor]:
    """block: [1, n] (block[0] is the anchor). logits: [1, n, V] for the same positions.

    Returns (number of accepted drafted tokens, bonus token)."""
    post = logits.argmax(dim=-1)
    a = int((block[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0])
    return a, post[0, a]


def _draft_hidden(draft, head_src, ctx, block, position_ids, start, dcache) -> torch.Tensor:
    """Drafter hidden states for block positions 1.. (anchor at ``start``).

    ctx covers positions [start - ctx_len, start). Positions after the anchor are always
    fed as mask tokens: in the three-stage loop the buffer can hold stale tokens from a path
    the target rejected, which would corrupt the drafter input. Leaves the drafter cache
    holding [0, start)."""
    vs = block.shape[1]
    block = block.clone()
    block[:, 1:] = draft.mask_token_id
    noise = _raw_input_embeddings(
        head_src, block, float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    )
    hidden = draft(
        target_hidden=ctx,
        noise_embedding=noise,
        position_ids=position_ids[:, start - ctx.shape[1] : start + vs],
        past_key_values=dcache,
        use_cache=True,
    )[:, 1 - vs :, :]
    crop(dcache, start)
    return hidden


@torch.inference_mode()
def draft_logits(draft, head_src, ctx, block, position_ids, start, dcache) -> torch.Tensor:
    """Per-position drafter logits [vs - 1, V] for the positions after the anchor."""
    hidden = _draft_hidden(draft, head_src, ctx, block, position_ids, start, dcache)
    return draft.compute_logits(hidden, _output_head(head_src))[0]


@torch.inference_mode()
def propose(draft, head_src, ctx, block, position_ids, start, dcache) -> torch.Tensor:
    """Run the drafter once on a block anchored at ``start``; returns the block with
    positions 1.. filled by the drafter's greedy proposal."""
    hidden = _draft_hidden(draft, head_src, ctx, block, position_ids, start, dcache)
    out = block.clone()
    if isinstance(draft, DFlash2DraftModel):
        tokens, _, _ = draft.propose(hidden, block[:, 0], _output_head(head_src), 0.0)
    else:
        tokens = draft.compute_logits(hidden, _output_head(head_src)).argmax(dim=-1)
    out[:, 1:] = tokens
    return out


@dataclass
class GenResult:
    output_ids: torch.Tensor          # [1, prompt + generated]
    num_input_tokens: int
    decode_time: float
    rounds: list[int] = field(default_factory=list)          # tokens produced per drafter round
    target_calls: int = 0
    mid_calls: int = 0
    draft_calls: int = 0
    feature_calls: int = 0
    checks: list[dict] = field(default_factory=list)          # three-stage: per target check
    tq: list[int] = field(default_factory=list)               # query tokens per decode-time target forward

    @property
    def generated(self) -> torch.Tensor:
        return self.output_ids[0, self.num_input_tokens :]

    @property
    def num_output_tokens(self) -> int:
        return int(self.output_ids.shape[1] - self.num_input_tokens)

    @property
    def tok_per_s(self) -> float:
        return self.num_output_tokens / self.decode_time if self.decode_time > 0 else 0.0

    def summary(self) -> dict:
        n = max(self.num_output_tokens, 1)
        d = {
            "num_output_tokens": self.num_output_tokens,
            "decode_time": self.decode_time,
            "tok_per_s": self.tok_per_s,
            "mean_round_len": sum(self.rounds) / len(self.rounds) if self.rounds else 0.0,
            "target_calls_per_token": self.target_calls / n,
            "mid_calls_per_token": self.mid_calls / n,
            "draft_calls_per_token": self.draft_calls / n,
            "target_calls": self.target_calls,
            "mid_calls": self.mid_calls,
            "mean_target_q": sum(self.tq) / len(self.tq) if self.tq else 1.0,
        }
        if self.checks:
            d["mean_pending_per_check"] = sum(c["pending"] for c in self.checks) / len(self.checks)
            d["mean_accepted_per_check"] = sum(c["accepted"] for c in self.checks) / len(self.checks)
            d["full_accept_rate"] = sum(c["accepted"] == c["pending"] for c in self.checks) / len(self.checks)
            if any("branches" in c for c in self.checks):
                d["mean_branches"] = sum(c.get("branches", 0) for c in self.checks) / len(self.checks)
                d["branch_hit_rate"] = sum(c.get("branch_hit", 0) for c in self.checks) / len(self.checks)
        return d


# ---------------------------------------------------------------------------
# autoregressive baseline
# ---------------------------------------------------------------------------

@torch.inference_mode()
def ar_generate(target, input_ids, max_new_tokens, stop_token_ids) -> GenResult:
    n = input_ids.shape[1]
    stop = torch.tensor(stop_token_ids, device=input_ids.device) if stop_token_ids else None
    cache = _make_cache(target.config)
    out = target(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    tokens = [out.logits[0, -1].argmax()]
    t0 = _sync_time()
    calls = 0
    while len(tokens) < max_new_tokens and not (stop is not None and torch.isin(tokens[-1], stop)):
        pos = torch.tensor([[n + len(tokens) - 1]], device=input_ids.device)
        out = target(tokens[-1].view(1, 1), position_ids=pos, past_key_values=cache, use_cache=True)
        tokens.append(out.logits[0, -1].argmax())
        calls += 1
    dt = _sync_time() - t0
    ids = torch.cat([input_ids, torch.stack(tokens).view(1, -1)], dim=1)
    res = GenResult(ids, n, dt, rounds=[1] * len(tokens), target_calls=calls + 1)
    res.tq = [1] * calls
    return res


# ---------------------------------------------------------------------------
# two-stage DFlash with configurable verifier / feature model
# ---------------------------------------------------------------------------

@torch.inference_mode()
def two_stage_generate(
    draft,
    head_src,
    verifier,
    feature_model,
    input_ids,
    max_new_tokens,
    stop_token_ids,
    block_size: int | None = None,
) -> GenResult:
    """DFlash greedy decoding. ``head_src`` provides embeddings + LM head for the drafter
    (always the bf16 target). With feature_model is verifier this is vanilla DFlash."""
    device = input_ids.device
    bs = draft.block_size if block_size is None else block_size
    shared = feature_model is verifier
    n = input_ids.shape[1]
    max_len = n + max_new_tokens
    output_ids = torch.full((1, max_len + bs + 1), draft.mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    stop = torch.tensor(stop_token_ids, device=device) if stop_token_ids else None
    vcache = _make_cache(verifier.config)
    dcache = _make_cache(draft.config)
    fcache = None if shared else _make_cache(feature_model.config)

    out = verifier(input_ids, position_ids=position_ids[:, :n], past_key_values=vcache,
                   use_cache=True, logits_to_keep=1, output_hidden_states=shared)
    output_ids[:, :n] = input_ids
    output_ids[0, n] = out.logits[0, -1].argmax()
    if shared:
        ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)
    else:
        fout = feature_model(input_ids, position_ids=position_ids[:, :n], past_key_values=fcache,
                             use_cache=True, logits_to_keep=1, output_hidden_states=True)
        ctx = extract_context_feature(fout.hidden_states, draft.target_layer_ids)

    res = GenResult(output_ids, n, 0.0)
    res.target_calls = 1
    t0 = _sync_time()
    start = n
    stopped = _first_stop(output_ids[0, n : n + 1], stop) is not None
    while start + 1 < max_len and not stopped:
        vs = min(bs, max_len - start)
        block = output_ids[:, start : start + vs].clone()
        if vs > 1:
            block = propose(draft, head_src, ctx, block, position_ids, start, dcache)
            res.draft_calls += 1
        out = verifier(block, position_ids=position_ids[:, start : start + vs], past_key_values=vcache,
                       use_cache=True, output_hidden_states=shared and vs > 1)
        res.target_calls += 1
        res.tq.append(vs)
        a, bonus = _greedy_accept(block, out.logits)
        output_ids[:, start : start + a + 1] = block[:, : a + 1]
        output_ids[0, start + a + 1] = bonus
        produced = min(a + 1, max_len - start - 1)
        hit = _first_stop(output_ids[0, start + 1 : start + produced + 1], stop)
        if hit is not None:
            produced, stopped = hit + 1, True
        new_ctx_ids = output_ids[:, start : start + produced]
        if shared:
            if vs > 1:
                ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)[:, :produced]
        else:
            fout = feature_model(new_ctx_ids, position_ids=position_ids[:, start : start + produced],
                                 past_key_values=fcache, use_cache=True, logits_to_keep=1,
                                 output_hidden_states=True)
            res.feature_calls += 1
            feats = extract_context_feature(fout.hidden_states, draft.target_layer_ids)
            ctx = feats if vs > 1 else torch.cat([ctx, feats], dim=1)
        start += produced
        crop(vcache, start)
        res.rounds.append(produced)

    res.decode_time = _sync_time() - t0
    res.output_ids = output_ids[:, : min(start + 1, max_len)]
    return res


# ---------------------------------------------------------------------------
# DDTree: DFlash drafter + best-first draft tree verified by the target
# ---------------------------------------------------------------------------

@torch.inference_mode()
def ddtree_generate(
    draft,
    target,
    input_ids,
    max_new_tokens,
    stop_token_ids,
    budget: int = 64,
    block_size: int | None = None,
    top_k: int = 16,
) -> GenResult:
    """Greedy DDTree. Each round the drafter gives per-position logits, a best-first tree
    of ``budget`` nodes is built from them and the target verifies the whole tree in one
    forward. The accepted path is re-run once on a cropped cache to get a clean cache and
    drafter features (implementation shortcut, not counted as a target call; it inflates
    measured wall time, so use the cost model for speed)."""
    if isinstance(draft, DFlash2DraftModel):
        raise ValueError("ddtree_generate needs a DFlash (v1) drafter")
    from hspec.tree import build_ddtree, greedy_walk, verify_tree

    device = input_ids.device
    bs = draft.block_size if block_size is None else block_size
    n = input_ids.shape[1]
    max_len = n + max_new_tokens
    output_ids = torch.full((1, max_len + bs + 1), draft.mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    stop = torch.tensor(stop_token_ids, device=device) if stop_token_ids else None
    tcache = _make_cache(target.config)
    dcache = _make_cache(draft.config)

    out = target(input_ids, position_ids=position_ids[:, :n], past_key_values=tcache,
                 use_cache=True, logits_to_keep=1, output_hidden_states=True)
    output_ids[:, :n] = input_ids
    output_ids[0, n] = out.logits[0, -1].argmax()
    ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)

    res = GenResult(output_ids, n, 0.0)
    res.target_calls = 1
    t0 = _sync_time()
    start = n
    stopped = _first_stop(output_ids[0, n : n + 1], stop) is not None
    while start + 1 < max_len and not stopped:
        vs = min(bs, max_len - start)
        block = output_ids[:, start : start + vs]
        logits = draft_logits(draft, target, ctx, block, position_ids, start, dcache)
        res.draft_calls += 1
        tree = build_ddtree(logits, budget, top_k=top_k)
        tout = verify_tree(target, tcache, output_ids[0, start], tree, start)
        res.target_calls += 1
        res.tq.append(1 + len(tree))
        path, bonus = greedy_walk(tree, tout.logits)
        a = len(path)
        if a:
            output_ids[0, start + 1 : start + 1 + a] = torch.tensor(
                [tree.tokens[i] for i in path], device=device)
        output_ids[0, start + a + 1] = bonus
        produced = min(a + 1, max_len - start - 1)
        hit = _first_stop(output_ids[0, start + 1 : start + produced + 1], stop)
        if hit is not None:
            produced, stopped = hit + 1, True
        # rebuild: clean cache + target features for [start, start + produced)
        crop(tcache, start)
        rout = target(output_ids[:, start : start + produced],
                      position_ids=position_ids[:, start : start + produced],
                      past_key_values=tcache, use_cache=True, logits_to_keep=1,
                      output_hidden_states=True)
        ctx = extract_context_feature(rout.hidden_states, draft.target_layer_ids)
        start += produced
        res.rounds.append(produced)

    res.decode_time = _sync_time() - t0
    res.output_ids = output_ids[:, : min(start + 1, max_len)]
    return res


# ---------------------------------------------------------------------------
# three-stage: drafter -> middle verifier -> deferred target check
# ---------------------------------------------------------------------------

@dataclass
class WindowPolicy:
    """When to call the target.

    fixed:    check once at least ``window`` tokens are pending.
    adaptive: window = P* = ln(1 + eps * c_T / c) / eps, with eps (per-token disagreement
              between middle model and target) and c_T / c (target-check cost over the
              per-token cost of the lower stages) estimated online.
    """

    mode: str = "fixed"
    window: int = 32
    min_window: int = 4
    max_window: int = 256
    prior_tokens: float = 50.0   # pseudo-count for eps estimate
    prior_eps: float = 0.03

    # online state
    checked: float = 0.0
    rejects: float = 0.0
    t_target: float = 0.0
    n_target: int = 0
    t_lower: float = 0.0
    tok_lower: int = 0

    def eps(self) -> float:
        return (self.rejects + self.prior_eps * self.prior_tokens) / (self.checked + self.prior_tokens)

    def current_window(self) -> int:
        if self.mode == "fixed":
            return self.window
        if self.n_target == 0 or self.tok_lower == 0:
            return self.window
        eps = self.eps()
        c_t = self.t_target / self.n_target
        c = self.t_lower / self.tok_lower
        p_star = math.log1p(eps * c_t / max(c, 1e-9)) / eps
        return int(min(max(p_star, self.min_window), self.max_window))


@torch.inference_mode()
def three_stage_generate(
    draft,
    target,
    mid,
    input_ids,
    max_new_tokens,
    stop_token_ids,
    policy: WindowPolicy,
    block_size: int | None = None,
    time_stages: bool = False,
    branch_k: int = 0,
    branch_len: int = 8,
    branch_margin: float = 0.5,
) -> GenResult:
    """Drafter proposes blocks, ``mid`` verifies them (standard greedy DFlash rounds, the
    drafter is conditioned on mid's hidden states), and the target checks all pending
    mid-accepted tokens in one forward pass when the window policy says so.

    ``time_stages`` synchronizes around each stage so the adaptive policy can measure
    costs (needed for mode="adaptive").

    Branching (branch_k > 0): at the check, up to ``branch_k`` pending positions where mid's
    top-1 / top-2 probability margin is below ``branch_margin`` get a sibling branch: mid's
    top-2 token followed by a copy of the next ``branch_len`` pending tokens. The target
    verifies the chain plus branches as one tree, so a disagreement at a low-margin
    position can still be recovered in the same check."""
    from hspec.tree import Tree, greedy_walk, verify_tree
    device = input_ids.device
    bs = draft.block_size if block_size is None else block_size
    n = input_ids.shape[1]
    max_len = n + max_new_tokens
    output_ids = torch.full((1, max_len + bs + 2), draft.mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    stop = torch.tensor(stop_token_ids, device=device) if stop_token_ids else None
    timing = time_stages or policy.mode == "adaptive"
    if branch_k > 0:   # mid's top-1/top-2 margin and top-2 token per sequence position
        margin_buf = torch.full((output_ids.shape[1],), float("inf"), device=device)
        alt_buf = torch.zeros(output_ids.shape[1], dtype=torch.long, device=device)

    tcache = _make_cache(target.config)
    mcache = _make_cache(mid.config)
    dcache = _make_cache(draft.config)

    # prefill: target picks the first token; mid provides drafter context features
    out = target(input_ids, position_ids=position_ids[:, :n], past_key_values=tcache,
                 use_cache=True, logits_to_keep=1)
    output_ids[:, :n] = input_ids
    output_ids[0, n] = out.logits[0, -1].argmax()
    mout = mid(input_ids, position_ids=position_ids[:, :n], past_key_values=mcache,
               use_cache=True, logits_to_keep=1, output_hidden_states=True)
    ctx = extract_context_feature(mout.hidden_states, draft.target_layer_ids)

    res = GenResult(output_ids, n, 0.0)
    res.target_calls = 1
    tf = n          # target cache holds [0, tf); token at tf is target-approved
    mlen = n        # mid cache length
    dlen = 0        # drafter cache length
    start = n       # current anchor (last token in the sequence)
    done = _first_stop(output_ids[0, n : n + 1], stop) is not None
    final_len = n + 1 if done else None

    t0 = _sync_time()
    while not done:
        # ---------------- stage 1+2: draft a block, mid verifies it ----------------
        tl0 = _sync_time() if timing else 0.0
        vs = min(bs, max_len - start)
        block = output_ids[:, start : start + vs].clone()
        if vs > 1:
            block = propose(draft, target, ctx, block, position_ids, start, dcache)
            dlen = start
            res.draft_calls += 1
        offset = start - mlen            # tokens mid has not processed yet before the anchor
        mid_in = torch.cat([output_ids[:, mlen:start], block], dim=1)
        mout = mid(mid_in, position_ids=position_ids[:, mlen : start + vs], past_key_values=mcache,
                   use_cache=True, output_hidden_states=True)
        res.mid_calls += 1
        a, bonus = _greedy_accept(block, mout.logits[:, offset:])
        output_ids[:, start : start + a + 1] = block[:, : a + 1]
        output_ids[0, start + a + 1] = bonus
        produced = min(a + 1, max_len - start - 1)
        if branch_k > 0:
            top = torch.softmax(mout.logits[0, offset : offset + produced].float(), dim=-1).topk(2, dim=-1)
            margin_buf[start + 1 : start + 1 + produced] = top.values[:, 0] - top.values[:, 1]
            alt_buf[start + 1 : start + 1 + produced] = top.indices[:, 1]
        feats = extract_context_feature(mout.hidden_states, draft.target_layer_ids)
        feats = feats[:, offset : offset + produced]
        ctx = feats if vs > 1 else torch.cat([ctx, feats], dim=1)
        mlen = start + produced
        crop(mcache, mlen)
        start += produced
        res.rounds.append(produced)
        if timing:
            policy.t_lower += _sync_time() - tl0
            policy.tok_lower += produced

        pending = start - tf
        eos_pending = _first_stop(output_ids[0, tf + 1 : start + 1], stop) is not None
        at_end = start + 1 >= max_len
        if pending < policy.current_window() and not eos_pending and not at_end:
            continue

        # ---------------- stage 3: target checks all pending tokens ----------------
        tt0 = _sync_time() if timing else 0.0
        t_in = output_ids[:, tf : start + 1]
        branch_pos: list[int] = []
        if branch_k > 0 and pending > 0:
            m = margin_buf[tf + 1 : start + 1]
            cand = (m < branch_margin).nonzero(as_tuple=True)[0]
            if cand.numel():
                order = m[cand].argsort()[:branch_k]
                branch_pos = sorted(int(tf + 1 + c) for c in cand[order].tolist())
        rec = {"pending": pending, "window": policy.current_window()}

        if not branch_pos:
            tout = target(t_in, position_ids=position_ids[:, tf : start + 1], past_key_values=tcache,
                          use_cache=True, output_hidden_states=True)
            res.tq.append(pending + 1)
            a_t, t_bonus = _greedy_accept(t_in, tout.logits)
            a_main, div = a_t, None
            tfeats = extract_context_feature(tout.hidden_states, draft.target_layer_ids)
        else:
            # main chain first (node i <-> position tf+1+i), then the branches
            main = output_ids[0, tf + 1 : start + 1].tolist()
            tree = Tree()
            for i, tok in enumerate(main):
                tree.add(tok, i - 1)
            for p in branch_pos:
                i = p - tf - 1
                alt = int(alt_buf[p])
                if alt == main[i]:
                    continue
                node = tree.add(alt, i - 1)
                for j in range(i + 1, min(i + 1 + branch_len, pending)):
                    node = tree.add(main[j], node)
            tout = verify_tree(target, tcache, output_ids[0, tf], tree, tf)
            res.tq.append(1 + len(tree))
            path, t_bonus = greedy_walk(tree, tout.logits)
            a_t = len(path)
            a_main = next((k for k, nd in enumerate(path) if nd >= pending), a_t)
            div = tf + 1 + a_main if a_main < a_t else None   # first position off the main chain
            if a_t:
                output_ids[0, tf + 1 : tf + 1 + a_t] = torch.tensor(
                    [tree.tokens[nd] for nd in path], device=device)
            rec.update(branches=len(tree) - pending, branch_hit=int(div is not None))
        res.target_calls += 1
        new_anchor = tf + a_t + 1
        output_ids[0, new_anchor] = t_bonus
        rec.update(accepted=a_t, main_accepted=a_main)
        res.checks.append(rec)
        policy.checked += min(a_main + 1, pending)
        policy.rejects += int(a_main < pending)

        if branch_pos:
            # clean target cache + features for [tf, new_anchor); not counted as a call
            crop(tcache, tf)
            rout = target(output_ids[:, tf:new_anchor], position_ids=position_ids[:, tf:new_anchor],
                          past_key_values=tcache, use_cache=True, logits_to_keep=1,
                          output_hidden_states=True)
            tfeats = extract_context_feature(rout.hidden_states, draft.target_layer_ids)
        keep = new_anchor if div is None else div   # prefix that is unchanged for mid / drafter

        # drafter context: target features for [d_keep, new_anchor)
        d_keep = min(dlen, new_anchor - 1, keep)
        crop(dcache, d_keep)
        dlen = d_keep
        ctx = tfeats[:, d_keep - tf : new_anchor - tf]

        mlen = min(mlen, keep)
        crop(mcache, mlen)
        crop(tcache, new_anchor)
        old_tf, tf, start = tf, new_anchor, new_anchor
        if timing:
            policy.t_target += _sync_time() - tt0
            policy.n_target += 1

        hit = _first_stop(output_ids[0, old_tf + 1 : new_anchor + 1], stop)
        if hit is not None:
            final_len, done = old_tf + 1 + hit + 1, True
        elif new_anchor + 1 >= max_len:
            final_len, done = max_len, True

    res.decode_time = _sync_time() - t0
    res.output_ids = output_ids[:, : min(final_len, max_len)]
    return res
