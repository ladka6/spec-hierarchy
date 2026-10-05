"""Feature-refreshed DFlash decoding (greedy, lossless), end to end.

One round, anchor token at position ``start`` (not yet processed by any model):

  1. drafter pass 1   block 1 from the real target features ctx: y_1 .. y_D (D = bs - 1)
  2. copy pass        a cheap copy of the target (e.g. 4-bit) processes the not-yet-seen accepted
                      tokens plus [anchor, y_1 .. y_D] with its own KV cache. It gives
                        - features for positions start .. start+D (stand-ins for the target's)
                        - its own next-token predictions: the first position j where it disagrees
                          with block 1, and its token there (the correction)
  3. drafter pass 2   restarts, each a fresh block anchored after a kept prefix, conditioned on
                      the real features before ``start`` (drafter cache) and the copy's features
                      for start .. anchor-1:
                        pc      prefix y_1..y_j, anchor = the copy's correction (always)
                        conf k  prefix y_1..y_i, anchor y_i for the k most likely stop positions i
  4. verify           the target checks block 1 and all restart chains as one tree; greedy walk;
                      KV compaction to the accepted path; the path's target features become the
                      next round's ctx (as in DDTree).

Every emitted token is the target's argmax given the emitted prefix, so the output equals plain
greedy decoding (up to bf16 numerics). Restarts run as separate drafter calls here (correctness
first); batching them is a speed optimisation.
"""

from __future__ import annotations

import torch

from dflash.model import _make_cache, extract_context_feature

from hspec.pipeline import GenResult, _first_stop, _sync_time, crop, draft_logits
from hspec.tree import Tree, compact, greedy_walk, verify_tree


def chains_to_tree(chains) -> Tree:
    """Prefix trie of token chains (lists), parents before children."""
    tree = Tree()
    kids: dict[int, dict[int, int]] = {-1: {}}
    for c in chains:
        cur = -1
        for t in c:
            nxt = kids[cur].get(int(t))
            if nxt is None:
                nxt = tree.add(int(t), cur)
                kids[cur][int(t)] = nxt
                kids[nxt] = {}
            cur = nxt
    return tree


def stop_posterior(conf: list[float]) -> dict[int, float]:
    """P(acceptance of block 1 stops right after keeping i tokens), i = 1 .. D (D = all kept)."""
    D = len(conf)
    post, run = {}, 1.0
    for i in range(1, D + 1):
        run *= conf[i - 1]
        post[i] = run * (1.0 - conf[i]) if i < D else run
    return post


@torch.inference_mode()
def refresh_generate(draft, target, copy, input_ids, max_new_tokens, stop_token_ids,
                     ks: int = 2, use_pc: bool = True, block_size: int | None = None,
                     profile: bool = False) -> GenResult:
    """Greedy feature-refreshed decoding. ks = number of confidence restarts besides the copy-corrected
    one (0 = pc only). use_pc=False gives the conf-only variant."""
    device = input_ids.device
    bs = draft.block_size if block_size is None else block_size
    D = bs - 1
    n = input_ids.shape[1]
    max_len = n + max_new_tokens
    output_ids = torch.full((1, max_len + 2 * bs + 2), draft.mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    stop = torch.tensor(stop_token_ids, device=device) if stop_token_ids else None
    tcache = _make_cache(target.config)
    dcache = _make_cache(draft.config)
    ccache = _make_cache(copy.config)

    out = target(input_ids, position_ids=position_ids[:, :n], past_key_values=tcache,
                 use_cache=True, logits_to_keep=1, output_hidden_states=True)
    output_ids[:, :n] = input_ids
    output_ids[0, n] = out.logits[0, -1].argmax()
    ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)
    copy(input_ids, position_ids=position_ids[:, :n], past_key_values=ccache, use_cache=True, logits_to_keep=1)
    c_len = n                                   # copy cache holds [0, c_len) (prompt prefilled, untimed)

    res = GenResult(output_ids, n, 0.0)
    res.target_calls = 1
    res.copy_calls = 0
    res.timing = {"draft1": 0.0, "copy": 0.0, "draft2": 0.0, "verify": 0.0}
    tick = _sync_time if profile else (lambda: 0.0)
    t0 = _sync_time()
    start = n
    stopped = _first_stop(output_ids[0, n : n + 1], stop) is not None
    while start + 1 < max_len and not stopped:
        # 1. block 1 from real features (drafter cache ends at [0, start) afterwards)
        ta = tick()
        block = output_ids[:, start : start + bs]
        lg = draft_logits(draft, target, ctx, block, position_ids, start, dcache)
        res.draft_calls += 1
        pr = torch.softmax(lg.float(), -1)
        conf_t, y_t = pr.max(-1)
        y, conf = y_t.tolist(), conf_t.tolist()
        tb = tick()

        # 2. copy pass over the unseen accepted tokens + anchor + block 1
        seq = torch.cat([output_ids[:, c_len : start + 1], y_t.view(1, -1)], 1)
        co = copy(seq, position_ids=position_ids[:, c_len : start + bs], past_key_values=ccache,
                  use_cache=True, logits_to_keep=bs, output_hidden_states=True)
        res.copy_calls += 1
        cfeat = extract_context_feature(co.hidden_states, draft.target_layer_ids)[:, -bs:].to(ctx.dtype)
        cpred = co.logits[0].argmax(-1).tolist()            # predictions for start+1 .. start+bs
        crop(ccache, start)                                 # keep only verified-or-anchor-free prefix
        c_len = start
        del co
        tc = tick()

        # 3. restarts (separate drafter calls; drafter cache is restored to [0, start) after each)
        chains = [y]
        if use_pc:
            j = next((k for k in range(D) if cpred[k] != y[k]), D)
            a = start + j + 1                              # restart anchor position
            blk = torch.full((1, bs), draft.mask_token_id, dtype=torch.long, device=device)
            blk[0, 0] = cpred[j]
            lr = draft_logits(draft, target, cfeat[:, : j + 1], blk, position_ids, a, dcache)
            crop(dcache, start)
            res.draft_calls += 1
            chains.append(y[:j] + [cpred[j]] + lr.argmax(-1).tolist())
        if ks:
            post = stop_posterior(conf)
            for i in sorted(post, key=lambda i: -post[i])[:ks]:
                a = start + i
                blk = torch.full((1, bs), draft.mask_token_id, dtype=torch.long, device=device)
                blk[0, 0] = y[i - 1]
                lr = draft_logits(draft, target, cfeat[:, :i], blk, position_ids, a, dcache)
                crop(dcache, start)
                res.draft_calls += 1
                chains.append(y[:i] + lr.argmax(-1).tolist())
        room = max_len - start                             # never draft past the length budget
        chains = [c[:room] for c in chains]
        tree = chains_to_tree(chains)
        td = tick()

        # 4. verify the tree with the target
        tout = verify_tree(target, tcache, output_ids[0, start], tree, start, hidden=True)
        res.target_calls += 1
        res.tq.append(1 + len(tree))
        path, bonus = greedy_walk(tree, tout.logits)
        acc = len(path)
        if acc:
            output_ids[0, start + 1 : start + 1 + acc] = torch.tensor([tree.tokens[i] for i in path], device=device)
        output_ids[0, start + acc + 1] = bonus
        produced = min(acc + 1, max_len - start - 1)
        hit = _first_stop(output_ids[0, start + 1 : start + produced + 1], stop)
        if hit is not None:
            produced, stopped = hit + 1, True
        te = tick()
        if profile:
            for k, v in (("draft1", tb - ta), ("copy", tc - tb), ("draft2", td - tc), ("verify", te - td)):
                res.timing[k] += v
        rows = ([0] + [nd + 1 for nd in path])[:produced]
        compact(tcache, start, rows)
        ctx = extract_context_feature(tout.hidden_states, draft.target_layer_ids)[:, rows]
        start += produced
        res.rounds.append(produced)

    res.decode_time = _sync_time() - t0
    res.output_ids = output_ids[:, : min(start + 1, max_len)]
    return res
