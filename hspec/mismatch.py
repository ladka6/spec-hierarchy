"""Untimed target-only replay for a failed exact-match check.

Reports evidence, not an automatic numerical-error verdict. A small logit margin alone
does not prove that a mismatch is harmless. Never turn a failed match into a passing run.
"""

from __future__ import annotations

import torch
from dflash.model import _make_cache
from hspec.pipeline import crop
from hspec.reserve import common_prefix


def _top(logits):
    values, indices = logits.float().topk(min(5, logits.shape[-1]))
    return dict(tokens=indices.tolist(), logits=values.tolist(),
                margin=float(values[0] - values[1]) if len(values) > 1 else None)


@torch.inference_mode()
def diagnose_mismatch(target, input_ids, reference_ids, actual_ids, history):
    """Replay AR and the recorded target query shapes up to the first divergence.

    All concurrent workers must be drained before calling. Query history is essential:
    replaying the entire prefix in one forward would introduce yet another shape change.
    """
    reference = tuple(reference_ids[0].tolist())
    actual = tuple(actual_ids[0].tolist())
    n = input_ids.shape[1]
    p = common_prefix(reference, actual)
    result = dict(first_difference=p, generated_index=p - n, prompt_tokens=n,
                  reference_length=len(reference), actual_length=len(actual),
                  expected_token=reference[p] if p < len(reference) else None,
                  actual_token=actual[p] if p < len(actual) else None,
                  input_ids=input_ids[0].tolist(), reference_ids=list(reference), actual_ids=list(actual),
                  target_history=history)
    if p >= min(len(reference), len(actual)):
        result["kind"] = "length_mismatch"
        return result
    if p < n:
        result["kind"] = "prompt_corruption"
        return result
    result["kind"] = "token_mismatch"
    device = next(target.parameters()).device
    ids = input_ids.to(device)
    cache = _make_cache(target.config)
    out = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    for position in range(n, p):
        out = target(torch.tensor([[reference[position]]], device=device),
                     position_ids=torch.tensor([[position]], device=device),
                     past_key_values=cache, use_cache=True)
    result["ar_replay"] = _top(out.logits[0, -1])

    cache = _make_cache(target.config)
    out = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    if p == n:
        result["query_replay"] = _top(out.logits[0, -1])
        result["query_size"] = n
        return result
    replay_differences = []
    for i, event in enumerate(history):
        anchor = event["anchor"]
        query = event["query_tokens"]
        emitted = event["emitted_tokens"]
        if cache.get_seq_length() != anchor:
            result["history_error"] = dict(check=i, anchor=anchor, cache_length=cache.get_seq_length())
            return result
        out = target(torch.tensor([query], device=device),
                     position_ids=torch.arange(anchor, anchor + len(query), device=device)[None],
                     past_key_values=cache, use_cache=True)
        predicted = out.logits[0, :len(emitted)].argmax(-1).tolist()
        if predicted != emitted:
            replay_differences.append(i)
        if anchor < p <= anchor + len(emitted):
            result.update(query_replay=_top(out.logits[0, p - anchor - 1]),
                          query_size=len(query), query_anchor=anchor,
                          replay_different_checks=replay_differences)
            return result
        crop(cache, anchor + len(emitted))
    result["history_error"] = "no target check accounts for the differing token"
    return result
