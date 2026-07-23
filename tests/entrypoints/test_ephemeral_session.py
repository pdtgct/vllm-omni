# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The canonical ephemeral-transcription orchestrator (PORT-EPH-001/002/004).

GPU-free CPU tier. Drives ``transcribe_ephemeral`` over a fake
``SessionFactory`` / ``SessionLease`` that satisfy the published
protocols by SHAPE (never importing or subclassing them) and record
every lifecycle call. Pins:

- exactly one ``TranscriptionResult`` on success; ``finish`` then
  ``release``, never ``abort`` (PORT-EPH-002);
- the pre-submit slice bound holds for arbitrary clip lengths incl. a
  zero-sample clip, and every sample is fed exactly once (no cadence
  arithmetic in the orchestrator, ING-FE-005/006);
- error / cancellation / admission-busy / deadline-expiry -> abort +
  release idempotently, NO result (PORT-EPH-002);
- an engine-originated ``TimeoutError`` is NOT misclassified as the
  orchestrator's own deadline expiry;
- ``finish`` and ``abort`` never both run; ``release`` runs on every
  path even if the terminal call raises;
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
from typing import Any

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
_compose_deadline = _EPH._compose_deadline

pytestmark = [pytest.mark.cpu]

#: A representative pre-submit bound (samples). Numeric here in the
#: TEST is fine; only the ORCHESTRATOR source must be arithmetic-free.
_BOUND = 17920
_CADENCE = "1120ms"
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
        abort_raise: BaseException | None = None,
    ) -> None:
        self.transcript = transcript
        self._feed_raise = feed_raise
        self._flush_raise = flush_raise
        self._flush_hang_s = flush_hang_s
        self._finish_raise = finish_raise
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
        if self._finish_raise is not None:
            raise self._finish_raise

    async def release(self) -> None:
        self.calls.append("release")
        self.release_count += 1


class FakeSessionFactory:
    """Mints one fake lease, or raises ``AdmissionBusyError`` when scripted."""

    def __init__(
        self, lease: FakeSessionLease | None = None, *, busy: bool = False
    ) -> None:
        self._lease = lease if lease is not None else FakeSessionLease()
        self._busy = busy
        self.opened: list[tuple[str, str]] = []

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
            cadence=_CADENCE,
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


def test_opaque_cadence_and_locale_pass_through_untouched() -> None:
    # @spec PORT-EPH-004
    factory = FakeSessionFactory()
    _run(
        transcribe_ephemeral(
            [_block(1)],
            factory=factory,
            cadence="weird-cadence",
            locale="xx-YY",
            finalization_timeout_s=5.0,
            submit_bound_samples=_BOUND,
        )
    )
    assert factory.opened == [("weird-cadence", "xx-YY")]


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


def test_finish_raising_still_releases_and_never_aborts() -> None:
    # @spec PORT-EPH-002
    lease = FakeSessionLease(finish_raise=RuntimeError("finish boom"))
    factory = FakeSessionFactory(lease)
    with pytest.raises(RuntimeError, match="finish boom"):
        _call(factory, [_block(_BOUND)])
    assert lease.finish_count == 1
    assert lease.abort_count == 0  # finish never runs alongside abort
    assert lease.release_count == 1  # release runs even if finish raises


def test_abort_raising_still_releases() -> None:
    # @spec PORT-EPH-002
    lease = FakeSessionLease(
        feed_raise=RuntimeError("feed boom"),
        abort_raise=RuntimeError("abort boom"),
    )
    factory = FakeSessionFactory(lease)
    with pytest.raises(RuntimeError):
        _call(factory, [_block(_BOUND)])
    assert lease.abort_count == 1
    assert lease.finish_count == 0
    assert lease.release_count == 1  # release runs even if abort raises


# ---- deadline composition (A2: serving layer owns deadlines) ------------------


def test_compose_deadline_takes_the_earlier_bound() -> None:
    # @spec PORT-EPH-001
    assert _compose_deadline(5.0, None) == 5.0
    assert _compose_deadline(5.0, 2.0) == 2.0  # transport deadline is earlier
    assert _compose_deadline(1.5, 9.0) == 1.5  # finalization bound is earlier
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            _compose_deadline(bad, None)


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
