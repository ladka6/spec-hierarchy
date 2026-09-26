"""Scheduler tests use a deterministic backend; no torch/GPU/downloads required."""

import time
import unittest

from hspec.reserve import (DraftJob, ReadyState, ReserveConfig, TargetResult,
                           reserve_decode)


def oracle(prefix):
    return (sum(prefix) + len(prefix) * 3) % 23


class FakeBackend:
    resources = {s: s for s in ("draft", "middle", "target")}

    def __init__(self, limit=22, alternatives=True, wrong_at=3):
        self.limit, self.alternatives, self.wrong_at = limit, alternatives, wrong_at

    def middle_token(self, prefix):
        return 99 if len(prefix) == self.wrong_at else oracle(prefix)

    def draft(self, state):
        time.sleep(0.001)
        prefix = state.prefix
        for _ in range(min(4, self.limit - len(prefix))):
            prefix += (self.middle_token(prefix),)
        return DraftJob(state, prefix[len(state.prefix):])

    def middle(self, job):
        time.sleep(0.003)
        prefix = job.state.prefix
        alts = []
        for token in job.tokens:
            correct = self.middle_token(prefix)
            if len(prefix) == self.wrong_at and self.alternatives:
                alts.append(ReadyState(prefix + (oracle(prefix),), score=0.5, fork=len(prefix)))
            prefix += (correct,)
            if token != correct:
                break
        if len(prefix) < self.limit:
            prefix += (self.middle_token(prefix),)
        return [ReadyState(prefix, score=job.state.score, fork=job.state.fork)] + alts

    def recover(self, state, prefix):
        time.sleep(0.002)
        return ReadyState(prefix)

    def verify(self, committed, proposal):
        time.sleep(0.025)
        pending = len(proposal) - len(committed)
        prefix, accepted = committed, 0
        for token in proposal[len(committed):]:
            correct = oracle(prefix)
            if token != correct:
                return TargetResult(prefix + (correct,), accepted, pending, True)
            prefix += (token,)
            accepted += 1
        if len(prefix) < self.limit:
            prefix += (oracle(prefix),)
        return TargetResult(prefix, accepted, pending, False)


def reference(limit):
    prefix = (1,)
    while len(prefix) < limit:
        prefix += (oracle(prefix),)
    return prefix


class SchedulerTests(unittest.TestCase):
    def run_decode(self, cfg, backend=None, stops=()):
        backend = backend or FakeBackend()
        initial = ReadyState(reference(2))
        r = reserve_decode(backend, initial, backend.limit, stops, cfg)
        self.assertEqual(r.prefix, reference(len(r.prefix)))
        self.assertLessEqual(r.metrics["max_live_branches"], cfg.reserve_size + 1)
        for value in r.metrics["stage_idle_fraction"].values():
            self.assertTrue(0 <= value <= 1)
        return r

    def test_sync_no_reserve_and_async_reserve_are_lossless(self):
        for cfg in (ReserveConfig(0, synchronous=True),
                    ReserveConfig(0, target_fallback=False), ReserveConfig(2),
                    ReserveConfig(2, target_fallback=False, max_ahead=2)):
            with self.subTest(cfg=cfg):
                r = self.run_decode(cfg)
                self.assertEqual(len(r.prefix), 22)
                self.assertGreater(r.metrics["rollbacks"], 0)

    def test_prepared_branch_survives_target_correction(self):
        r = self.run_decode(ReserveConfig(2, target_fallback=False))
        self.assertGreater(r.metrics["branch_hits"], 0)
        self.assertGreater(r.metrics["reusable_tokens"], 0)
        self.assertTrue(r.metrics["recovery_delays"])

    def test_real_workers_overlap(self):
        r = self.run_decode(ReserveConfig(2, target_fallback=False))
        self.assertTrue(any(a["stage"] != b["stage"]
                            and max(a["started"], b["started"]) < min(a["ended"], b["ended"])
                            for a in r.trace for b in r.trace))

    def test_shared_resource_is_serialized(self):
        b = FakeBackend()
        b.resources = dict.fromkeys(b.resources, "one-device")
        r = self.run_decode(ReserveConfig(2), b)
        events = sorted(r.trace, key=lambda e: e["started"])
        for a, b in zip(events, events[1:]):
            self.assertLessEqual(a["ended"], b["started"])

    def test_target_can_use_raw_draft_without_middle_approval(self):
        class SlowMiddle(FakeBackend):
            def middle(self, job):
                time.sleep(0.06)
                return super().middle(job)
        r = self.run_decode(ReserveConfig(0, target_fallback=False), SlowMiddle())
        first_target = next(e for e in r.trace if e["stage"] == "target")
        first_middle = next(e for e in r.trace if e["stage"] == "middle")
        self.assertLess(first_target["started"], first_middle["ended"])

    def test_stop_token_in_rejected_suffix_does_not_stop(self):
        # 99 is produced by the middle but never belongs to the target trajectory.
        r = self.run_decode(ReserveConfig(2), stops=(99,))
        self.assertEqual(len(r.prefix), 22)

    def test_stop_token_and_immediate_stop(self):
        token = reference(6)[-1]
        r = self.run_decode(ReserveConfig(2), stops=(token,))
        expected = reference(22)
        first = next(i for i in range(1, len(expected)) if expected[i] == token)
        self.assertEqual(len(r.prefix), first + 1)
        initial = ReadyState((1, 7))
        r = reserve_decode(FakeBackend(), initial, 22, (7,))
        self.assertEqual(r.prefix, initial.prefix)
        self.assertEqual(r.trace, [])

    def test_worker_failure_propagates(self):
        class Broken(FakeBackend):
            def draft(self, state):
                raise ValueError("broken draft")
        with self.assertRaisesRegex(ValueError, "broken draft"):
            self.run_decode(ReserveConfig(0, target_fallback=False), Broken())

    def test_invalid_config(self):
        for kwargs in (dict(reserve_size=-1), dict(max_ahead=0), dict(target_window=0),
                       dict(reserve_size=2, synchronous=True)):
            with self.assertRaises(ValueError):
                ReserveConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
