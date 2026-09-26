"""Unmodified DFlash + middle + target backend for the feature-ready reserve.

Supports CPU correctness runs and 1/2/3 CUDA-device placements. Same-device stages
are serialized by the scheduler, deliberately avoiding an unmeasured kernel-overlap
assumption. On separate devices the three workers execute independently. Cache
snapshots are copied for safety; copies and feature transfers are charged to real time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from dflash.model import _make_cache, _output_head, extract_context_feature
from hspec.pipeline import GenResult, _greedy_accept, crop, propose
from hspec.reserve import (DraftJob, ReadyState, ReserveConfig, TargetResult,
                           common_prefix, reserve_decode)


@dataclass(frozen=True)
class _Features:
    feats: torch.Tensor             # on the middle device, covers prefix[:-1]
    mcache: object                  # immutable, also covers prefix[:-1]
    dcache: object                  # immutable, on the draft device
    dlen: int                       # reusable draft context length (may require cropping)


class _HeadSource(nn.Module):
    """Only copy the target embedding/head when drafting on a different device."""

    def __init__(self, target, device):
        super().__init__()
        self.embedding = copy.deepcopy(target.get_input_embeddings()).to(device)
        self.lm_head = copy.deepcopy(_output_head(target)).to(device)

    def get_input_embeddings(self):
        return self.embedding


def _clone(cache, length):
    value = copy.deepcopy(cache)
    crop(value, length)
    return value


class TorchReserveBackend:
    def __init__(self, draft, target, mid, *, max_length, reserve_size=2,
                 fork_threshold=0.6, block_size=None):
        self.draft_model, self.target, self.mid = draft, target, mid
        self.devices = {name: next(model.parameters()).device for name, model in
                        (("draft", draft), ("middle", mid), ("target", target))}
        if any(d.type not in ("cpu", "cuda") for d in self.devices.values()):
            raise ValueError("reserve backend currently supports CPU and CUDA only")
        if any(m.training for m in (draft, target, mid)):
            raise ValueError("all three models must be in eval mode")
        if (mid.config.hidden_size != target.config.hidden_size
                or mid.config.num_hidden_layers != target.config.num_hidden_layers
                or mid.config.vocab_size != target.config.vocab_size):
            raise ValueError("middle model must share the target hidden space and vocabulary")
        if reserve_size < 0 or not 0 <= fork_threshold <= 1:
            raise ValueError("invalid reserve size or fork threshold")
        self.resources = {name: str(device) for name, device in self.devices.items()}
        self.max_length, self.reserve_size = max_length, reserve_size
        self.fork_threshold = fork_threshold
        self.bs = draft.block_size if block_size is None else block_size
        if self.bs < 2:
            raise ValueError("block_size must be >= 2")
        self.head = (target if self.devices["draft"] == self.devices["target"]
                     else _HeadSource(target, self.devices["draft"]).eval())
        self.tcache = None
        self.target_prefix = None
        self._sync("draft")

    def _sync(self, stage):
        device = self.devices[stage]
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()

    def _ids(self, prefix, stage):
        return torch.tensor([prefix], dtype=torch.long, device=self.devices[stage])

    @torch.inference_mode()
    def prefill(self, prompt: tuple[int, ...]) -> ReadyState:
        self.tcache = _make_cache(self.target.config)
        out = self.target(self._ids(prompt, "target"), past_key_values=self.tcache,
                          use_cache=True, logits_to_keep=1)
        prefix = prompt + (int(out.logits[0, -1].argmax()),)
        self.target_prefix = prefix
        mcache = _make_cache(self.mid.config)
        out = self.mid(self._ids(prompt, "middle"), past_key_values=mcache,
                       use_cache=True, output_hidden_states=True)
        feats = extract_context_feature(out.hidden_states, self.draft_model.target_layer_ids)
        self._sync("target")
        self._sync("middle")
        return ReadyState(prefix, _Features(feats, mcache, _make_cache(self.draft_model.config), 0))

    @torch.inference_mode()
    def draft(self, state):
        f = state.payload
        anchor = len(state.prefix) - 1
        size = min(self.bs, self.max_length - anchor)
        dcache = _clone(f.dcache, f.dlen)
        device = self.devices["draft"]
        ctx = f.feats[:, f.dlen:anchor].to(device)
        block = torch.full((1, size), self.draft_model.mask_token_id,
                           dtype=torch.long, device=device)
        block[0, 0] = state.prefix[-1]
        pos = torch.arange(anchor + size, device=device).unsqueeze(0)
        block = propose(self.draft_model, self.head, ctx, block, pos, anchor, dcache)
        tokens = tuple(block[0, 1:].tolist())
        self._sync("draft")
        return DraftJob(state, tokens, dcache)

    @torch.inference_mode()
    def middle(self, job):
        state, f = job.state, job.state.payload
        anchor = len(state.prefix) - 1
        mcache = _clone(f.mcache, anchor)
        block = self._ids((state.prefix[-1],) + job.tokens, "middle")
        pos = torch.arange(anchor, anchor + block.shape[1], device=self.devices["middle"])[None]
        out = self.mid(block, position_ids=pos, past_key_values=mcache,
                       use_cache=True, output_hidden_states=True)
        accepted, bonus = _greedy_accept(block, out.logits)
        prefix = (state.prefix + job.tokens[:accepted] + (int(bonus),))[:self.max_length]
        produced = len(prefix) - len(state.prefix)
        feats = torch.cat((f.feats, extract_context_feature(
            out.hidden_states, self.draft_model.target_layer_ids)[:, :produced]), dim=1)
        payload = _Features(feats, _clone(mcache, len(prefix) - 1), job.payload, anchor)
        states = [ReadyState(prefix, payload, state.score, state.fork)]
        if self.reserve_size and produced:
            top = out.logits[0, :produced].float().softmax(-1).topk(2, dim=-1)
            confidences = top.values[:, 0].tolist()
            for j in sorted(range(produced), key=confidences.__getitem__)[:self.reserve_size]:
                if confidences[j] >= self.fork_threshold:
                    continue
                p = anchor + 1 + j
                if p + 1 >= self.max_length:
                    continue
                alt = int(top.indices[j, 1])
                alt_prefix = prefix[:p] + (alt,)
                ratio = float(top.values[j, 1] / top.values[j, 0])
                alt_payload = _Features(feats[:, :p], _clone(mcache, p), job.payload, anchor)
                states.append(ReadyState(alt_prefix, alt_payload, state.score * ratio, p))
        self._sync("middle")
        return states

    @torch.inference_mode()
    def recover(self, state, prefix):
        """Rebuild only missing middle features after a target correction/advance."""
        f = state.payload
        anchor = len(prefix) - 1
        keep = min(common_prefix(state.prefix, prefix), len(state.prefix) - 1, anchor)
        cache = _clone(f.mcache, keep)
        feats = f.feats[:, :keep]
        if keep < anchor:
            pos = torch.arange(keep, anchor, device=self.devices["middle"])[None]
            out = self.mid(self._ids(prefix[keep:anchor], "middle"), position_ids=pos,
                           past_key_values=cache, use_cache=True, output_hidden_states=True)
            feats = torch.cat((feats, extract_context_feature(
                out.hidden_states, self.draft_model.target_layer_ids)), dim=1)
        self._sync("middle")
        return ReadyState(prefix, _Features(feats, cache, f.dcache, min(f.dlen, keep)), 1.0)

    @torch.inference_mode()
    def verify(self, committed, proposal):
        if committed != self.target_prefix or proposal[:len(committed)] != committed:
            raise ValueError("target received a stale or incompatible prefix")
        anchor = len(committed) - 1
        block = self._ids(proposal[anchor:], "target")
        pos = torch.arange(anchor, len(proposal), device=self.devices["target"])[None]
        out = self.target(block, position_ids=pos, past_key_values=self.tcache, use_cache=True)
        accepted, bonus = _greedy_accept(block, out.logits)
        pending = len(proposal) - len(committed)
        prefix = (committed + proposal[len(committed):len(committed) + accepted]
                  + (int(bonus),))[:self.max_length]
        crop(self.tcache, len(prefix) - 1)
        self.target_prefix = prefix
        self._sync("target")
        return TargetResult(prefix, accepted, pending, accepted < pending)


def reserve_generate(draft, target, mid, input_ids, max_new_tokens, stop_token_ids,
                     config: ReserveConfig | None = None, *, fork_threshold=0.6,
                     block_size=None, backend: TorchReserveBackend | None = None) -> GenResult:
    """Measured wall-clock decoder. Reuse ``backend`` to avoid repeated head copies.

    ``decode_time`` includes draining outstanding calls after the final token. Setup
    and prefill times are reported separately. This API is greedy and batch-size one.
    """
    cfg = config or ReserveConfig()
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("expected one nonempty prompt")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if not 0 <= fork_threshold <= 1 or (block_size is not None and block_size < 2):
        raise ValueError("invalid fork_threshold or block_size")
    n = input_ids.shape[1]
    if max_new_tokens == 0:
        r = GenResult(input_ids.clone(), n, 0.0)
        r.reserve_stats, r.reserve_trace = {}, []
        return r
    start = perf_counter()
    if backend is None:
        backend = TorchReserveBackend(draft, target, mid, max_length=n + max_new_tokens,
                                      reserve_size=cfg.reserve_size,
                                      fork_threshold=fork_threshold, block_size=block_size)
    else:
        if (backend.draft_model is not draft or backend.target is not target or backend.mid is not mid):
            raise ValueError("backend models do not match")
        backend.max_length = n + max_new_tokens
        backend.reserve_size = cfg.reserve_size
        backend.fork_threshold = fork_threshold
        if block_size is not None:
            backend.bs = block_size
    setup = perf_counter() - start
    start = perf_counter()
    initial = backend.prefill(tuple(input_ids[0].tolist()))
    prefill = perf_counter() - start
    result = reserve_decode(backend, initial, n + max_new_tokens, stop_token_ids, cfg)
    ids = torch.tensor([result.prefix], device=input_ids.device)
    r = GenResult(ids, n, result.decode_time)
    r.target_calls = 1 + result.metrics["stages"]["target"]["calls"]
    r.mid_calls = result.metrics["stages"]["middle"]["calls"]
    r.draft_calls = result.metrics["stages"]["draft"]["calls"]
    for e in result.trace:
        if e["stage"] == "target" and "accepted" in e:
            r.checks.append(dict(pending=e["pending"], accepted=e["accepted"]))
            r.tq.append(e["pending"] + 1)
    result.metrics.update(setup_seconds=setup, prefill_seconds=prefill,
                          placement=backend.resources,
                          timing="wall_clock_including_drain",
                          head_copy=backend.head is not target)
    r.reserve_stats, r.reserve_trace = result.metrics, result.trace
    return r
