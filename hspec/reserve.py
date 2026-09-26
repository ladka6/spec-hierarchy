"""Bounded feature-ready job reserve with independent, real-time stage workers.

This scheduler has no tensor dependencies. Backend states are immutable snapshots:
workers must never mutate a submitted state's caches. At most one call per stage is
in flight; stages sharing a resource are explicitly serialized and that wait is logged.
Only the target advances the committed prefix. Late incompatible results are dropped.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Protocol


def common_prefix(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compatible(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return common_prefix(a, b) == min(len(a), len(b))


@dataclass(frozen=True)
class ReadyState:
    prefix: tuple[int, ...]          # includes anchor; features cover prefix[:-1]
    payload: Any = None
    score: float = 1.0
    fork: int | None = None         # zero-based correction position, if alternative


@dataclass(frozen=True)
class DraftJob:
    state: ReadyState
    tokens: tuple[int, ...]         # tokens after the anchor
    payload: Any = None

    @property
    def prefix(self):
        return self.state.prefix + self.tokens


@dataclass(frozen=True)
class TargetResult:
    prefix: tuple[int, ...]
    accepted: int
    pending: int
    rejected: bool


class ReserveBackend(Protocol):
    resources: dict[str, str]

    def draft(self, state: ReadyState) -> DraftJob: ...
    def middle(self, job: DraftJob) -> list[ReadyState]: ...
    def recover(self, state: ReadyState, prefix: tuple[int, ...]) -> ReadyState: ...
    def verify(self, committed: tuple[int, ...], proposal: tuple[int, ...]) -> TargetResult: ...


@dataclass(frozen=True)
class ReserveConfig:
    reserve_size: int = 2           # additional live paths, excluding the leading path
    max_ahead: int = 64
    target_window: int = 32         # upper bound; target also checks shorter proposals
    target_fallback: bool = True    # target may decode one token when no proposal is ready
    synchronous: bool = False      # ablation: one stage at a time, reserve must be zero

    def __post_init__(self):
        if self.reserve_size < 0 or self.max_ahead < 1 or self.target_window < 1:
            raise ValueError("reserve_size must be >= 0; max_ahead and target_window must be > 0")
        if self.synchronous and self.reserve_size:
            raise ValueError("synchronous baseline requires reserve_size=0")


@dataclass
class ReserveResult:
    prefix: tuple[int, ...]
    decode_time: float
    metrics: dict
    trace: list[dict]


@dataclass
class _Branch:
    state: ReadyState
    status: str = "ready"
    proposal: DraftJob | None = None


def reserve_decode(backend: ReserveBackend, initial: ReadyState, max_length: int,
                   stop_ids=(), config: ReserveConfig | None = None) -> ReserveResult:
    """Decode from a target-selected first token. Time excludes prefill and includes drain.

    ``initial.prefix`` contains the prompt plus the first target token. All backend
    methods must finish device work before returning. Worker errors propagate after
    outstanding jobs are drained. The target backend owns its own sequential cache.
    """
    cfg = config or ReserveConfig()
    if not initial.prefix or max_length < len(initial.prefix):
        raise ValueError("max_length must include the initial prefix")
    stops = set(stop_ids)
    committed = initial.prefix
    branches = {0: _Branch(initial)}
    next_id = 1
    recovery_source = initial
    sync_needs_check = False
    active: dict[str, tuple[Future, int | None]] = {}
    locks = {r: Lock() for r in backend.resources.values()}
    trace: list[dict] = []
    t0 = perf_counter()
    last = 0.0
    final_at = None
    pending_recovery = None
    metrics = dict(rollbacks=0, branch_hits=0, reusable_tokens=0, stale_results=0,
                   ready_queue_seconds=0.0, ready_queue_empty_seconds=0.0,
                   max_live_branches=1, recovery_delays=[], confirmation_bursts=[])
    idle = {s: 0.0 for s in ("draft", "middle", "target")}
    queue_peak = 0

    def done():
        return len(committed) >= max_length or committed[-1] in stops

    def runnable(b):
        return (b.status == "ready" and len(b.state.prefix) < max_length
                and len(b.state.prefix) - len(committed) < cfg.max_ahead
                and not any(t in stops for t in b.state.prefix[len(committed):]))

    def account():
        nonlocal last, queue_peak
        now = perf_counter() - t0
        dt = now - last
        ready = sum(runnable(b) for b in branches.values())
        queue_peak = max(queue_peak, ready)
        metrics["ready_queue_seconds"] += ready * dt
        metrics["ready_queue_empty_seconds"] += (ready == 0) * dt
        for stage in idle:
            idle[stage] += (stage not in active) * dt
        last = now

    def rank(item):
        _, b = item
        return b.state.score, len(b.state.prefix)

    def invoke(stage, fn, args, bid, prefix, kind):
        submitted = perf_counter() - t0

        def run():
            entered = perf_counter() - t0
            with locks[backend.resources[stage]]:
                started = perf_counter() - t0
                value = fn(*args)
                ended = perf_counter() - t0
            return value, dict(stage=stage, kind=kind, branch=bid, submitted=submitted,
                               entered=entered, started=started, ended=ended,
                               input_length=len(prefix), prefix=prefix)

        active[stage] = (pool.submit(run), bid)

    def collect(stage):
        future, bid = active.pop(stage)
        value, event = future.result()
        event["received"] = perf_counter() - t0
        trace.append(event)
        return value, event, bid

    def admit(states):
        nonlocal next_id
        for state in states:
            if not compatible(state.prefix, committed):
                continue
            if any(b.state.prefix == state.prefix for b in branches.values()):
                continue
            branches[next_id] = _Branch(state)
            next_id += 1
        # Retain the strongest paths, including any running path if it ranks highly.
        keep = sorted(branches.items(), key=rank, reverse=True)[:cfg.reserve_size + 1]
        keep_ids = {i for i, _ in keep}
        for i in list(branches):
            if i not in keep_ids:
                del branches[i]
        metrics["max_live_branches"] = max(metrics["max_live_branches"], len(branches))

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="reserve") as pool:
        while not done():
            account()
            # Apply target decisions first so simultaneous stale completions cannot win.
            for stage in ("target", "middle", "draft"):
                if stage not in active or not active[stage][0].done():
                    continue
                value, event, bid = collect(stage)
                if stage == "target":
                    sync_needs_check = False
                    old = committed
                    assert value.prefix[:len(old)] == old and len(value.prefix) > len(old)
                    committed = value.prefix[:max_length]
                    for p in range(len(old), len(committed)):
                        if committed[p] in stops:
                            committed = committed[:p + 1]
                            break
                    event.update(accepted=value.accepted, pending=value.pending,
                                 rejected=value.rejected, output_prefix=committed)
                    metrics["confirmation_bursts"].append(
                        {"time": event["received"], "tokens": len(committed) - len(old)})
                    if value.rejected:
                        metrics["rollbacks"] += 1
                        pending_recovery = event["received"]
                        matches = [b for b in branches.values()
                                   if b.state.fork == len(committed) - 1
                                   and len(b.state.prefix) >= len(committed)
                                   and compatible(b.state.prefix, committed)]
                        if matches:
                            metrics["branch_hits"] += 1
                            metrics["reusable_tokens"] += max(
                                len(b.state.prefix) - len(committed) for b in matches)
                    previous = list(branches.values())
                    if previous:
                        recovery_source = max(previous, key=lambda b: common_prefix(
                            b.state.prefix, committed)).state
                    for i, b in list(branches.items()):
                        if not compatible(b.state.prefix, committed):
                            del branches[i]
                    if not branches and not done():
                        # Keep an immutable snapshot as a source for middle catch-up.
                        branches[next_id] = _Branch(recovery_source, "recover")
                        next_id += 1
                    # If an alternative won, restore its priority relative to new forks.
                    if branches:
                        scale = max(b.state.score for b in branches.values())
                        for b in branches.values():
                            s = b.state
                            b.state = ReadyState(s.prefix, s.payload, s.score / max(scale, 1e-30), s.fork)
                elif stage == "draft":
                    event["output_prefix"] = value.prefix
                    event["output_start"] = len(value.state.prefix)
                    if bid in branches and compatible(value.state.prefix, committed):
                        branches[bid].proposal = value
                        branches[bid].status = "drafted"
                    else:
                        metrics["stale_results"] += 1
                else:
                    states = value if isinstance(value, list) else [value]
                    if event["kind"] == "verify":
                        sync_needs_check = True
                    recovery_source = states[0]
                    event["output_prefix"] = states[0].prefix
                    event["output_start"] = event["input_length"]
                    if bid in branches:
                        del branches[bid]
                        admit(states)
                    else:
                        metrics["stale_results"] += 1
            if done():
                final_at = perf_counter() - t0
                break
            if not branches:
                branches[next_id] = _Branch(recovery_source, "recover")
                next_id += 1

            # Catch up features if the target advanced beyond a ready endpoint.
            if "middle" not in active and (not cfg.synchronous or (not active and not sync_needs_check)):
                candidates = [(i, b) for i, b in branches.items()
                              if b.status == "recover" or (b.status == "ready"
                                  and len(b.state.prefix) < len(committed))]
                if candidates:
                    i, b = max(candidates, key=rank)
                    b.status = "recovering"
                    invoke("middle", backend.recover, (b.state, committed), i, b.state.prefix, "recover")
                else:
                    candidates = [(i, b) for i, b in branches.items() if b.status == "drafted"]
                    if candidates:
                        i, b = max(candidates, key=rank)
                        b.status = "verifying"
                        invoke("middle", backend.middle, (b.proposal,), i, b.state.prefix, "verify")

            if "draft" not in active and (not cfg.synchronous or (not active and not sync_needs_check)):
                candidates = [(i, b) for i, b in branches.items()
                              if runnable(b) and len(b.state.prefix) >= len(committed)]
                if candidates:
                    i, b = max(candidates, key=rank)
                    b.status = "drafting"
                    invoke("draft", backend.draft, (b.state,), i, b.state.prefix, "draft")

            if "target" not in active and (not cfg.synchronous or not active):
                candidates = []
                for b in branches.values():
                    # Middle-approved paths outrank unverified small-model proposals.
                    for quality, prefix in ((1, b.state.prefix),
                                            (0, b.proposal.prefix if b.proposal else ())):
                        if len(prefix) > len(committed) and prefix[:len(committed)] == committed:
                            candidates.append((quality, b.state.score, len(prefix), prefix))
                if candidates or (cfg.target_fallback and not cfg.synchronous):
                    proposal = max(candidates)[-1] if candidates else committed
                    proposal = proposal[:min(max_length, len(committed) + cfg.target_window)]
                    if pending_recovery is not None and len(proposal) > len(committed):
                        metrics["recovery_delays"].append(perf_counter() - t0 - pending_recovery)
                        pending_recovery = None
                    invoke("target", backend.verify, (committed, proposal), None, committed,
                           "verify" if candidates else "fallback")

            if not active:
                raise RuntimeError("reserve scheduler stalled without runnable work")
            # Account the previous interval before sleeping with the new queue state.
            account()
            wait([f for f, _ in active.values()], return_when=FIRST_COMPLETED)

        if final_at is None:
            final_at = perf_counter() - t0
        # Include unavoidable in-flight work in resource cost and total runtime.
        for stage in list(active):
            value, event, _ = collect(stage)
            event["drained"] = True
            if stage == "draft":
                event.update(output_prefix=value.prefix, output_start=len(value.state.prefix))
            elif stage == "middle":
                event["output_prefix"] = (value[0] if isinstance(value, list) else value).prefix

    elapsed = perf_counter() - t0
    for event in trace:
        output = event.pop("output_prefix", ())
        event.pop("prefix", None)
        start = event.get("output_start", event["input_length"])
        event["compatible_output_tokens"] = max(0, common_prefix(output, committed) - start)
        event["proposed_tokens"] = max(0, len(output) - start)
    metrics.update(time_to_final_token=final_at, drain_seconds=elapsed - final_at,
                   ready_queue_peak=queue_peak,
                   mean_ready_queue=metrics["ready_queue_seconds"] / max(final_at, 1e-12),
                   ready_queue_empty_fraction=metrics["ready_queue_empty_seconds"] / max(final_at, 1e-12),
                   stage_idle_seconds=idle,
                   stage_idle_fraction={s: t / max(final_at, 1e-12) for s, t in idle.items()})
    metrics["stages"] = {}
    for stage in idle:
        events = [e for e in trace if e["stage"] == stage]
        proposed = sum(e["proposed_tokens"] for e in events)
        useful = sum(e["compatible_output_tokens"] for e in events)
        unique = set()
        for event in events:
            start = event.get("output_start", event["input_length"])
            unique.update(range(start, start + event["compatible_output_tokens"]))
        metrics["stages"][stage] = dict(
            calls=len(events), service_seconds=sum(e["ended"] - e["started"] for e in events),
            resource_wait_seconds=sum(e["started"] - e["entered"] for e in events),
            proposed_tokens=proposed, compatible_tokens=useful,
            unique_compatible_tokens=len(unique),
            unique_compatible_fraction=len(unique) / proposed if proposed else None,
            compatible_fraction=useful / proposed if proposed else None)
    return ReserveResult(committed, elapsed, metrics, trace)
