"""Real two-GPU backend: the target runs in its own process on a second GPU.

The lower stages (drafter + middle model) run in the calling process; target checks are
sent to a server process and answered asynchronously, so both GPUs really work at the
same time. A network delay L is emulated: a check may only start L/2 after it was sent,
and its answer counts as arrived L/2 after the target finished. All timestamps use
time.perf_counter (CLOCK_MONOTONIC, shared between processes on one machine).

Rollback: the client bumps an epoch and tells the server to crop its KV cache; checks
from an older epoch that the server has not started yet are skipped, and answers from an
older epoch are ignored.

Use with hier_generate(..., backend=ProcTarget(...)); pearl mode is not supported (the
server does not ship hidden states back).
"""

from __future__ import annotations

import queue
import time

import torch
import torch.multiprocessing as mp

from hspec.async3 import _Check


def _ms():
    return time.perf_counter() * 1000.0


def _load(model_id):
    """HF id, or "factory:module:function" (used by the CPU test with a tiny model)."""
    if model_id.startswith("factory:"):
        import importlib
        _, mod, fn = model_id.split(":")
        return getattr(importlib.import_module(mod), fn)()
    from hspec.models import load_target
    return load_target(model_id)


def _server(model_id, device, req_q, res_q, latency_ms, sys_path):
    import sys
    sys.path[:0] = [p for p in sys_path if p not in sys.path]
    if device is not None:
        torch.cuda.set_device(device)
    from dflash.model import _make_cache

    from hspec.pipeline import crop

    torch.set_grad_enabled(False)
    target = _load(model_id)
    dev = next(target.parameters()).device
    cache, epoch = None, 0
    res_q.put(("ready",))
    while True:
        msgs = [req_q.get()]
        while True:
            try:
                msgs.append(req_q.get_nowait())
            except queue.Empty:
                break
        newest = max([m[1] for m in msgs if m[0] in ("rollback", "prefill")] + [epoch])
        for m in msgs:
            kind = m[0]
            if kind == "stop":
                return
            if kind == "latency":
                latency_ms = m[2]
                continue
            if kind == "prefill":
                _, epoch, ids = m
                cache = _make_cache(target.config)
                ids = ids.to(dev)
                out = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                res_q.put(("prefill", epoch, int(out.logits[0, -1].argmax())))
            elif kind == "rollback":
                _, epoch, p = m
                crop(cache, p)
            elif kind == "check":
                _, ep, cid, c0, c1, toks, deliver_at = m
                if ep < newest or ep < epoch:
                    continue
                delay = deliver_at - _ms()
                if delay > 0:
                    time.sleep(delay / 1000.0)
                t0 = _ms()
                pos = torch.arange(c0, c1 + 1, device=dev).unsqueeze(0)
                out = target(toks.to(dev), position_ids=pos, past_key_values=cache, use_cache=True)
                preds = out.logits[0].argmax(-1).cpu()
                t1 = _ms()
                res_q.put(("check", ep, cid, preds, t1 + latency_ms / 2, t1 - t0, c1 - c0 + 1))


class ProcTarget:
    """Client side of the target server (see module docstring)."""

    has_feats = False
    is_real = True

    def __init__(self, model_id, device=1, latency_ms=0.0, local_device="cuda:0"):
        """device=None runs the server on CPU (tests)."""
        import sys
        ctx = mp.get_context("spawn")
        self.req, self.res = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_server, args=(model_id, device, self.req, self.res, latency_ms,
                                                      list(sys.path)), daemon=True)
        self.proc.start()
        assert self.res.get(timeout=1800)[0] == "ready"
        self.L, self.local = latency_ms, local_device
        self.epoch, self.cid = 0, 0
        self.inflight: list[_Check] = []
        self.by_cid: dict[int, _Check] = {}
        self.t0 = _ms()
        self.target_ms: list[tuple[int, float]] = []   # (q, compute ms) per answered check

    def set_latency(self, latency_ms):
        """Takes effect for checks sent from now on (the server adds its half on return)."""
        self.L = latency_ms
        self.req.put(("latency", self.epoch, latency_ms))

    def close(self):
        self.req.put(("stop",))
        self.proc.join(timeout=60)

    # -- backend interface -------------------------------------------------------------
    def prefill(self, input_ids, position_ids, want_hidden):
        assert not want_hidden
        self.epoch += 1
        self.inflight, self.by_cid = [], {}
        self.req.put(("prefill", self.epoch, input_ids.cpu()))
        while True:
            m = self.res.get(timeout=600)
            if m[0] == "prefill" and m[1] == self.epoch:
                break
        self.t0 = _ms()
        return m[2], None

    def now(self):
        return _ms() - self.t0

    def spend(self, ms):
        pass

    def submit(self, toks, c0, c1, position_ids):
        self.cid += 1
        ck = _Check(c0, c1, None, None, float("nan"), float("nan"), float("inf"))
        self.inflight.append(ck)
        self.by_cid[self.cid] = ck
        self.req.put(("check", self.epoch, self.cid, c0, c1, toks.cpu(), _ms() + self.L / 2))

    def _take(self, m):
        if m[0] != "check":
            return
        _, ep, cid, preds, ready_at, compute_ms, q = m
        self.target_ms.append((q, compute_ms))        # every forward the target ran, stale or not
        if ep != self.epoch:
            return
        ck = self.by_cid.pop(cid, None)
        if ck is not None:
            ck.preds = preds.to(self.local)
            ck.done_t = ready_at - self.t0

    def _drain(self):
        while True:
            try:
                self._take(self.res.get_nowait())
            except queue.Empty:
                return

    def pop_ready(self):
        self._drain()
        if self.inflight and self.inflight[0].preds is not None and self.inflight[0].done_t <= self.now():
            return self.inflight.pop(0)
        return None

    def wait_first(self):
        t = self.now()
        ck = self.inflight[0]
        while ck.preds is None:
            try:
                self._take(self.res.get(timeout=300))
            except queue.Empty:
                raise RuntimeError(f"no answer from the target server for 300 s (check {ck.c0}..{ck.c1}, "
                                   f"epoch {self.epoch}, server alive: {self.proc.is_alive()})") from None
        delay = ck.done_t - self.now()
        if delay > 0:
            time.sleep(delay / 1000.0)
        return self.now() - t

    def drop(self, p, t_learn):
        n = len(self.inflight)
        self.epoch += 1
        self.inflight, self.by_cid = [], {}
        self.req.put(("rollback", self.epoch, p))
        return n
