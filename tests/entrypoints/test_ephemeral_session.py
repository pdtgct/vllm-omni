# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The canonical ephemeral-transcription orchestrator (PORT-EPH-001/002/004).

GPU-free CPU tier. Drives ``transcribe_ephemeral`` over a fake
``SessionFactory`` / ``SessionLease`` that satisfy the published
protocols by SHAPE (never importing or subclassing them) and record
every lifecycle call. Pins:

- exactly one ``TranscriptionResult`` on success: ``flush`` then
  ``finish`` then ``release``, no ``abort`` (PORT-EPH-002);
- the canonical geometry is the factory's, not the caller's -- the
  orchestrator takes no cadence and drives ``open_ephemeral``
  (PORT-REGIME-002);
- the pre-submit slice bound holds for arbitrary clip lengths incl. a
  zero-sample clip, and every sample is fed exactly once (no cadence
  arithmetic in the orchestrator, ING-FE-005/006);
- error / cancellation / admission-busy / deadline-expiry -> abort +
  release idempotently, NO result (PORT-EPH-002);
- ``flush`` AND ``finish`` are bounded together, so a stalled ``finish``
  cannot retain the slot; the finalization budget is evaluated FRESH at
  drain time so a slow feed cannot leave it stale;
- a raising or stalled ``finish`` is a session error: abort recovers
  (finish is attempted, then abort), then release; a raising abort does
  not mask the original failure; ``release`` runs on every path;
- an engine-originated ``TimeoutError`` is NOT misclassified as the
  orchestrator's own deadline expiry;
- the orchestrator source carries no cadence/chunk arithmetic
  (source-scan, PORT-EPH-001).

Loader-runnable: the module under test is pure stdlib, loaded by file
path under a bare name so no ``vllm_omni`` parent ``__init__`` (which
imports ``vllm``) is touched.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

_MODULE_PATH = Path(
    os.environ.get("EPHEMERAL_SESSION_PATH")
    or (
        Path(__file__).resolve().parents[2]
        / "vllm_omni/entrypoints/ephemeral_session.py"
    )
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ephemeral_session_under_test", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field resolution can find the
    # module by its ``__module__`` name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_EPH = _load_module()
transcribe_ephemeral = _EPH.transcribe_ephemeral
TranscriptionResult = _EPH.TranscriptionResult
SessionFactory = _EPH.SessionFactory
SessionLease = _EPH.SessionLease
AdmissionBusyError = _EPH.AdmissionBusyError
FinalizationTimeoutError = _EPH.FinalizationTimeoutError
_remaining_drain_budget = _EPH._remaining_drain_budget

pytestmark = [pytest.mark.cpu]

#: A representative pre-submit bound (samples). Numeric here in the
#: TEST is fine; only the ORCHESTRATOR source must be arithmetic-free.
_BOUND = 17920
_LOCALE = "auto"


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def _block(n: int) -> Any:
    return np.zeros(n, dtype=np.float32)


# ---- fakes: satisfy the protocols by shape, never by import/subclass ----------


class FakeSessionLease:
    """Records every lifecycle call; scriptable to raise / hang / go busy."""

    def __init__(
        self,
        *,
        transcript: str = "the final transcript",
        feed_raise: BaseException | None = None,
        flush_raise: BaseException | None = None,
        flush_hang_s: float | None = None,
        finish_raise: BaseException | None = None,
        finish_hang_s: float | None = None,
        abort_raise: BaseException | None = None,
    ) -> None:
        self.transcript = transcript
        self._feed_raise = feed_raise
        self._flush_raise = flush_raise
        self._flush_hang_s = flush_hang_s
        self._finish_raise = finish_raise
        self._finish_hang_s = finish_hang_s
        self._abort_raise = abort_raise
        self.calls: list[str] = []
        self.fed: list[int] = []
        self.finish_count = 0
        self.abort_count = 0
        self.release_count = 0

    async def feed(self, samples: Any) -> list[str]:
        self.calls.append("feed")
        self.fed.append(len(samples))
        if self._feed_raise is not None:
            raise self._feed_raise
        return []

    async def flush(self) -> str:
        self.calls.append("flush")
        if self._flush_hang_s is not None:
            await asyncio.sleep(self._flush_hang_s)
        if self._flush_raise is not None:
            raise self._flush_raise
        return self.transcript

    async def update_locale(self, locale: str) -> None:
        self.calls.append("update_locale")

    async def abort(self) -> None:
        self.calls.append("abort")
        self.abort_count += 1
        if self._abort_raise is not None:
            raise self._abort_raise

    async def finish(self) -> None:
        self.calls.append("finish")
        self.finish_count += 1
        if self._finish_hang_s is not None:
            await asyncio.sleep(self._finish_hang_s)
        if self._finish_raise is not None:
            raise self._finish_raise

    async def release(self) -> None:
        self.calls.append("release")
        self.release_count += 1


class FakeSessionFactory:
    """Mints one fake lease, or raises ``AdmissionBusyError`` when scripted.

    Exposes BOTH factory entry points so it conforms to the protocol by
    shape; the orchestrator drives ``open_ephemeral`` (canonical
    geometry, no cadence). ``opened_ephemeral`` records the ephemeral
    leases; ``opened`` records realtime ones -- the orchestrator must
    never touch the latter.
    """

    def __init__(
        self, lease: FakeSessionLease | None = None, *, busy: bool = False
    ) -> None:
        self._lease = lease if lease is not None else FakeSessionLease()
        self._busy = busy
        self.opened_ephemeral: list[str] = []
        self.opened: list[tuple[str, str]] = []

    async def open_ephemeral(self, *, locale: str) -> FakeSessionLease:
        self.opened_ephemeral.append(locale)
        if self._busy:
            raise AdmissionBusyError("shared pool full")
        return self._lease

    async def open(self, *, cadence: str, locale: str) -> FakeSessionLease:
        self.opened.append((cadence, locale))
        if self._busy:
            raise AdmissionBusyError("shared pool full")
        return self._lease


def _call(
    factory: FakeSessionFactory,
    pieces: Any,
    *,
    finalization_timeout_s: float = 5.0,
    submit_bound_samples: int = _BOUND,
    deadline: float | None = None,
) -> Any:
    return _run(
        transcribe_ephemeral(
            pieces,
            factory=factory,
            locale=_LOCALE,
            finalization_timeout_s=finalization_timeout_s,
            submit_bound_samples=submit_bound_samples,
            deadline=deadline,
        )
    )


# ---- structural conformance (PORT-EPH-004) ------------------------------------


def test_fakes_conform_to_protocols_by_shape_only() -> None:
    # @spec PORT-EPH-004
    lease = FakeSessionLease()
    factory = FakeSessionFactory(lease)
    # runtime_checkable Protocols: isinstance verifies the method shape.
    assert isinstance(lease, SessionLease)
    assert isinstance(factory, SessionFactory)
    # The fakes neither import nor subclass the protocols.
    assert SessionLease not in type(lease).__mro__
    assert SessionFactory not in type(factory).__mro__


# ---- success path: exactly one result, finish-then-release (PORT-EPH-002) -----


def test_success_yields_exactly_one_result_finish_then_release() -> None:
    # @spec PORT-EPH-001, PORT-EPH-002
    lease = FakeSessionLease(transcript="hello world")
    factory = FakeSessionFactory(lease)
    result = _call(factory, [_block(_BOUND)])
    assert isinstance(result, TranscriptionResult)
    assert result.text == "hello world"
    assert lease.finish_count == 1
    assert lease.abort_count == 0
    assert lease.release_count == 1
    # finish precedes release, which is the last call; abort never runs.
    assert lease.calls[-3:] == ["flush", "finish", "release"]


def test_canonical_geometry_is_the_factorys_not_the_callers() -> None:
    # @spec PORT-EPH-001, PORT-REGIME-002
    # The orchestrator takes NO cadence: it drives open_ephemeral with
    # locale only, so the canonical 1120-ms geometry lives once in the
    # model-aware factory and no transport can pick the wrong cadence.
    factory = FakeSessionFactory()
    _run(
        transcribe_ephemeral(
            [_block(1)],
            factory=factory,
            locale="xx-YY",
            finalization_timeout_s=5.0,
            submit_bound_samples=_BOUND,
        )
    )
    assert factory.opened_ephemeral == ["xx-YY"]
    assert factory.opened == []  # the realtime entry is never used here


def test_transcribe_ephemeral_takes_no_cadence_parameter() -> None:
    # @spec PORT-EPH-001
    # A cadence kwarg must be a hard error, not silently accepted: the
    # canonical geometry is not the orchestrator's to choose.
    import inspect

    params = inspect.signature(transcribe_ephemeral).parameters
    assert "cadence" not in params


# ---- slice bound: arbitrary lengths, every sample fed once (ING-FE-005/006) ---


@pytest.mark.parametrize(
    "length",
    [0, 1, _BOUND - 1, _BOUND, _BOUND + 1, 2 * _BOUND, 3 * _BOUND + 7, 100_000],
)
def test_slice_bound_holds_for_arbitrary_clip_lengths(length: int) -> None:
    # @spec PORT-EPH-001, ING-FE-005, ING-FE-006
    lease = FakeSessionLease()
    factory = FakeSessionFactory(lease)
    result = _call(factory, [_block(length)], submit_bound_samples=_BOUND)
    # Every feed respects the pre-submit bound...
    assert all(n <= _BOUND for n in lease.fed)
    # ...every sample is fed exactly once (none dropped, none duplicated)...
    assert sum(lease.fed) == length
    # ...in the minimal number of bound-sized slices (0 for a zero clip)...
    assert len(lease.fed) == (length + _BOUND - 1) // _BOUND
    # ...and a result is still produced (flush owns the final tail).
    assert isinstance(result, TranscriptionResult)
    assert lease.finish_count == 1


def test_zero_sample_clip_flushes_final_tail_without_feeding() -> None:
    # @spec PORT-EPH-002, PORT-SESS-003
    # Both an empty iterable and a single zero-length block: no feed runs,
    # flush drains the zero-sample final tail, one result is emitted.
    for pieces in ([], [_block(0)]):
        lease = FakeSessionLease(transcript="")
        factory = FakeSessionFactory(lease)
        result = _call(factory, pieces)
        assert "feed" not in lease.calls
        assert lease.calls == ["flush", "finish", "release"]
        assert result.text == ""


def test_multiple_pieces_each_bound_sliced() -> None:
    # @spec ING-FE-005, ING-FE-006
    lease = FakeSessionLease()
    factory = FakeSessionFactory(lease)
    _call(
        factory,
        [_block(_BOUND + 3), _block(0), _block(2 * _BOUND)],
        submit_bound_samples=_BOUND,
    )
    assert all(n <= _BOUND for n in lease.fed)
    assert sum(lease.fed) == (_BOUND + 3) + 0 + (2 * _BOUND)
    assert lease.fed == [_BOUND, 3, _BOUND, _BOUND]


# ---- error / cancel -> abort + release, no result (PORT-EPH-002) --------------


def test_feed_error_aborts_and_releases_no_result() -> None:
    # @spec PORT-EPH-002
    lease = FakeSessionLease(feed_raise=RuntimeError("engine feed boom"))
    factory = FakeSessionFactory(lease)
    with pytest.raises(RuntimeError, match="engine feed boom"):
        _call(factory, [_block(_BOUND)])
    assert lease.abort_count == 1
    assert lease.finish_count == 0
    assert lease.release_count == 1
    assert "flush" not in lease.calls  # never reached finalization


def test_flush_error_aborts_and_releases_no_result() -> None:
    # @spec PORT-EPH-002
    lease = FakeSessionLease(flush_raise=RuntimeError("engine flush boom"))
    factory = FakeSessionFactory(lease)
    with pytest.raises(RuntimeError, match="engine flush boom"):
        _call(factory, [_block(_BOUND)])
    assert lease.abort_count == 1
    assert lease.finish_count == 0
    assert lease.release_count == 1


def test_cancellation_aborts_and_releases_no_result() -> None:
    # @spec PORT-EPH-002
    lease = FakeSessionLease(feed_raise=asyncio.CancelledError())
    factory = FakeSessionFactory(lease)
    with pytest.raises(asyncio.CancelledError):
        _call(factory, [_block(_BOUND)])
    assert lease.abort_count == 1
    assert lease.finish_count == 0
    assert lease.release_count == 1


def test_admission_busy_propagates_with_nothing_leased() -> None:
    # @spec PORT-EPH-002, PORT-EPH-004
    lease = FakeSessionLease()
    factory = FakeSessionFactory(lease, busy=True)
    with pytest.raises(AdmissionBusyError):
        _call(factory, [_block(_BOUND)])
    # No lease was minted, so no lifecycle call ever ran.
    assert lease.calls == []
    assert lease.finish_count == 0
    assert lease.abort_count == 0
    assert lease.release_count == 0


# ---- finish never alongside abort; release always runs ------------------------


def test_raising_abort_does_not_mask_the_original_failure() -> None:
    # @spec PORT-EPH-002
    # The original error (why we aborted) dominates; a raising abort is
    # suppressed so it cannot mask the real failure, and release still
    # runs after the best-effort abort.
    lease = FakeSessionLease(
        feed_raise=RuntimeError("feed boom"),
        abort_raise=RuntimeError("abort boom"),
    )
    factory = FakeSessionFactory(lease)
    with pytest.raises(RuntimeError, match="feed boom"):
        _call(factory, [_block(_BOUND)])
    assert lease.abort_count == 1
    assert lease.finish_count == 0
    assert lease.release_count == 1  # release runs even if abort raises


# ---- deadline composition (A2: serving layer owns deadlines) ------------------


def test_remaining_drain_budget_is_evaluated_fresh() -> None:
    # @spec PORT-EPH-001
    # No transport deadline: the mandatory bound is used verbatim.
    assert _remaining_drain_budget(5.0, None, 100.0) == 5.0
    # Transport deadline at t=102 (absolute), now t=100: 2 s remain, and
    # that is earlier than the 5 s mandatory bound -> 2.0.
    assert _remaining_drain_budget(5.0, 102.0, 100.0) == 2.0
    # The mandatory bound is earlier than the transport's remaining time.
    assert _remaining_drain_budget(1.5, 109.0, 100.0) == 1.5
    # STALE-BUDGET REGRESSION: feeding advanced the clock from 100 to 108
    # against a transport deadline at 102 -> the budget is now NEGATIVE
    # (already out of time), not the original 2 s. Composition happens at
    # finalization, not before the feeds.
    assert _remaining_drain_budget(5.0, 102.0, 108.0) == -6.0


def test_non_finite_finalization_bound_is_rejected() -> None:
    # @spec PORT-EPH-002
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        factory = FakeSessionFactory()
        with pytest.raises(ValueError):
            _call(factory, [_block(1)], finalization_timeout_s=bad)


def test_stalled_finish_past_deadline_aborts_no_result() -> None:
    # @spec PORT-EPH-002
    # The R1-in-its-new-home regression: flush drains cleanly but finish
    # hangs. finish is now UNDER the same deadline as flush, so a stalled
    # finish can no longer retain the slot -- it times out, aborts, and
    # releases, emitting no result.
    lease = FakeSessionLease(finish_hang_s=10.0)
    factory = FakeSessionFactory(lease)
    with pytest.raises(FinalizationTimeoutError):
        _call(factory, [_block(_BOUND)], finalization_timeout_s=0.02)
    assert "flush" in lease.calls  # flush completed
    assert lease.abort_count == 1  # ...then the stalled finish was aborted
    assert lease.release_count == 1


def test_raising_finish_aborts_and_releases_no_result() -> None:
    # @spec PORT-EPH-002
    # PORT-EPH-002 requires abort on a session error; a raising finish is
    # a session error, so abort recovery runs (not release-without-abort).
    lease = FakeSessionLease(finish_raise=RuntimeError("finish boom"))
    factory = FakeSessionFactory(lease)
    with pytest.raises(RuntimeError, match="finish boom"):
        _call(factory, [_block(_BOUND)])
    assert lease.finish_count == 1  # finish was attempted
    assert lease.abort_count == 1  # ...and its failure triggered abort
    assert lease.release_count == 1  # released after cleanup


def test_hanging_flush_past_composed_deadline_aborts_no_result() -> None:
    # @spec PORT-EPH-002
    lease = FakeSessionLease(flush_hang_s=10.0)
    factory = FakeSessionFactory(lease)
    with pytest.raises(FinalizationTimeoutError):
        _call(factory, [_block(_BOUND)], finalization_timeout_s=0.02)
    assert lease.abort_count == 1
    assert lease.finish_count == 0
    assert lease.release_count == 1


def test_transport_deadline_dominates_composition() -> None:
    # @spec PORT-EPH-001, PORT-EPH-002
    # Big finalization bound but a tiny transport deadline: the composed
    # (smaller) deadline is what bounds the drain.
    lease = FakeSessionLease(flush_hang_s=10.0)
    factory = FakeSessionFactory(lease)
    with pytest.raises(FinalizationTimeoutError):
        _call(
            factory,
            [_block(_BOUND)],
            finalization_timeout_s=10.0,
            deadline=0.02,
        )
    assert lease.abort_count == 1
    assert lease.release_count == 1


def test_engine_timeout_not_misclassified_as_deadline_expiry() -> None:
    # @spec PORT-EPH-002
    # flush raises TimeoutError itself, WITHOUT the composed deadline
    # expiring (large bound, no hang): it is a FAILED finalization and
    # must propagate as TimeoutError, never as FinalizationTimeoutError.
    lease = FakeSessionLease(flush_raise=TimeoutError("engine stalled"))
    factory = FakeSessionFactory(lease)
    with pytest.raises(TimeoutError) as excinfo:
        _call(factory, [_block(_BOUND)], finalization_timeout_s=30.0)
    assert not isinstance(excinfo.value, FinalizationTimeoutError)
    assert lease.abort_count == 1
    assert lease.finish_count == 0
    assert lease.release_count == 1


# ---- source-scan: no cadence/chunk arithmetic in the orchestrator -------------


def test_orchestrator_source_has_no_cadence_arithmetic() -> None:
    # @spec PORT-EPH-001
    source = _MODULE_PATH.read_text()
    banned = (
        "SAMPLE_RATE",
        "chunk_ms",
        "chunk_samples",
        "RAW_SAMPLES_PER_CHUNK",
        "VALID_CHUNK_MS",
        "// 1000",
        "16000",
        "1120",
    )
    present = [token for token in banned if token in source]
    assert present == [], (
        f"orchestrator source contains cadence/chunk arithmetic: {present}; "
        "PORT owns cadence segmentation (ING-FE-006)"
    )


# ---- round-5: expired deadlines and bounded cleanup (PORT-EPH-002) ------------


class InstantLease:
    """A lease whose every method completes WITHOUT yielding — the exact
    shape that slips past asyncio.timeout(0), which only cancels at the
    next scheduling point."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def feed(self, samples: Any) -> list[str]:
        self.calls.append("feed")
        return []

    async def flush(self) -> str:
        self.calls.append("flush")
        return "late-success"

    async def update_locale(self, locale: str) -> None:
        self.calls.append("update_locale")

    async def abort(self) -> None:
        self.calls.append("abort")

    async def finish(self) -> None:
        self.calls.append("finish")

    async def release(self) -> None:
        self.calls.append("release")


def test_already_expired_deadline_never_returns_success() -> None:
    # @spec PORT-EPH-002
    # The transport was already out of time at entry. Even with a lease
    # whose flush/finish complete instantly (never yielding), no result
    # may be returned: a non-positive budget is rejected BEFORE any
    # terminal step, and the session is aborted and released.
    lease = InstantLease()
    factory = FakeSessionFactory(cast(Any, lease))
    with pytest.raises(FinalizationTimeoutError):
        _call(factory, [], deadline=-1.0)
    assert "flush" not in lease.calls  # finalization never started
    assert "finish" not in lease.calls
    assert "abort" in lease.calls
    assert lease.calls[-1] == "release"


def test_zero_remaining_budget_never_returns_success() -> None:
    # @spec PORT-EPH-002
    # deadline=0.0: the absolute instant is "now", so the budget computed
    # after feeding is <= 0 — same rejection, instant-completing lease.
    lease = InstantLease()
    factory = FakeSessionFactory(cast(Any, lease))
    with pytest.raises(FinalizationTimeoutError):
        _call(factory, [_block(1)], deadline=0.0)
    assert "finish" not in lease.calls
    assert "abort" in lease.calls
    assert lease.calls[-1] == "release"


def test_negative_budget_helper_is_rejected_before_any_step() -> None:
    # @spec PORT-EPH-002
    lease = InstantLease()
    with pytest.raises(FinalizationTimeoutError):
        _run(_EPH._finalize_within_deadline(cast(Any, lease), 0.0))
    assert lease.calls == []  # neither flush nor finish ever ran


def test_hanging_abort_cannot_retain_the_slot() -> None:
    # @spec PORT-EPH-002
    # Failure path with an abort that would hang forever: cleanup is
    # bounded, so the call still completes (primary error preserved) and
    # release still runs.
    class HangingAbortLease(InstantLease):
        async def feed(self, samples: Any) -> list[str]:
            self.calls.append("feed")
            raise RuntimeError("feed boom")

        async def abort(self) -> None:
            self.calls.append("abort")
            await asyncio.sleep(30.0)

    lease = HangingAbortLease()
    factory = FakeSessionFactory(cast(Any, lease))
    with pytest.raises(RuntimeError, match="feed boom"):
        _call(factory, [_block(1)], finalization_timeout_s=0.05)
    assert "abort" in lease.calls
    assert lease.calls[-1] == "release"  # release ran despite the hang


def test_release_failure_never_masks_the_primary_error() -> None:
    # @spec PORT-EPH-002
    class RaisingReleaseLease(InstantLease):
        async def flush(self) -> str:
            self.calls.append("flush")
            raise RuntimeError("primary flush boom")

        async def release(self) -> None:
            self.calls.append("release")
            raise RuntimeError("secondary release boom")

    lease = RaisingReleaseLease()
    factory = FakeSessionFactory(cast(Any, lease))
    with pytest.raises(RuntimeError, match="primary flush boom"):
        _call(factory, [_block(1)])
    assert "abort" in lease.calls
    assert "release" in lease.calls


def test_release_failure_after_clean_finish_is_suppressed() -> None:
    # @spec PORT-EPH-002
    # Slot accounting is our own object; its failure after a clean
    # finalization is secondary — the result stands (documented policy).
    class RaisingReleaseLease(InstantLease):
        async def release(self) -> None:
            self.calls.append("release")
            raise RuntimeError("secondary release boom")

    lease = RaisingReleaseLease()
    factory = FakeSessionFactory(cast(Any, lease))
    result = _call(factory, [_block(1)], finalization_timeout_s=5.0)
    assert result.text == "late-success"
    assert "finish" in lease.calls
    assert "abort" not in lease.calls


def test_cancellation_during_cleanup_cannot_skip_release() -> None:
    # @spec PORT-EPH-002
    # The caller cancels while abort is mid-hang: release must still run,
    # and the task must end cancelled (the cancellation is re-delivered
    # after cleanup, never swallowed on the success-less path).
    async def scenario() -> None:
        started = asyncio.Event()

        class SlowAbortLease(InstantLease):
            async def feed(self, samples: Any) -> list[str]:
                self.calls.append("feed")
                raise RuntimeError("feed boom")

            async def abort(self) -> None:
                self.calls.append("abort")
                started.set()
                await asyncio.sleep(0.5)

        lease = SlowAbortLease()
        factory = FakeSessionFactory(cast(Any, lease))
        task = asyncio.ensure_future(
            transcribe_ephemeral(
                [_block(1)],
                factory=factory,
                locale=_LOCALE,
                finalization_timeout_s=5.0,
                submit_bound_samples=_BOUND,
            )
        )
        await started.wait()  # cleanup (abort) is in flight
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "release" in lease.calls  # cancellation did not skip it

    _run(scenario())
