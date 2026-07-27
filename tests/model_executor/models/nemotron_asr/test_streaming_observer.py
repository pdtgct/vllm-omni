# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: the model-package StreamingObserver protocol.

Specs: PORT-OBS-003 (Prometheus-free observer protocol owned by the
neutral ``streaming_transport`` module; unit-ready/minted/terminal-
disposition/overflow events; single-in-flight-handle park correlation;
ledgerless inertness), PORT-OBS-004 (chunk latency only for parked
units; final-tail ready stamp is caller-captured, never reconstructed),
PORT-OBS-005 (backlog +1 at ready, -1 exactly once at terminal
disposition), PORT-SESS-001 (the per-session accepted-audio SECONDS
budget bounds queue occupancy only; lifetime cumulative audio is never
capped by it).

``session.py``/``streaming.py`` are engine-free (dataclasses/asyncio/
numpy only), so this file loads them by the same importlib chain as
``test_realtime_session.py`` and runs on macOS/CPU, extended with the
neutral ``vllm_omni.metrics.streaming_transport`` module session.py now
imports its protocol/handle type from. The observer params threaded
through ``buffer_stream``/``NemotronRealtimeSession`` are Phase-5 stubs
— accepted, stored, but not yet wired to any event — so every
behavioral test below is EXPECTED RED (explicit missing-behavior
assertion) until Phase 6 wires the calls; the protocol-shape, inertness,
and park-correlation-design tests are real, passing pins.

Hygiene: the bare parent-package stubs this loader creates
(``vllm_omni``, ``vllm_omni.model_executor``, ``vllm_omni.model_executor.
models``, ``vllm_omni.metrics``) are restored via ``pytest.MonkeyPatch``
immediately after ``_load_chain()`` returns — NOT deferred to
``teardown_module``, which pytest's collect-then-run model makes too
late (a later-collected file needing a REAL ``vllm_omni.*`` import would
otherwise get ``ModuleNotFoundError: 'vllm_omni' is not a package``).
The genuine-content leaf modules (session, streaming, manifests,
configuration_nemotron_asr, streaming_transport) stay cached under their
full dotted names for the rest of the session — matching every sibling
loader-chain file's deliberate "reuse-if-present" class-identity
sharing; only the bare, path-less package placeholders are undone.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from collections import deque
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_PKG = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/nemotron_asr"
_METRICS_PKG = Path(__file__).resolve().parents[4] / "vllm_omni/metrics"
_BASE = "vllm_omni.model_executor.models.nemotron_asr"

_PARENT_STUBS = (
    "vllm_omni",
    "vllm_omni.model_executor",
    "vllm_omni.model_executor.models",
    "vllm_omni.metrics",
    _BASE,
)


def _load_chain() -> dict[str, Any]:
    mp = pytest.MonkeyPatch()
    try:
        for name in _PARENT_STUBS:
            if name not in sys.modules:
                mp.setitem(sys.modules, name, types.ModuleType(name))

        loaded: dict[str, Any] = {}

        def _load_leaf(dotted: str, path: Path) -> Any:
            existing = sys.modules.get(dotted)
            if existing is not None and getattr(existing, "__file__", None):
                return existing
            spec = importlib.util.spec_from_file_location(dotted, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = module  # persists: genuine file content, shared across files
            spec.loader.exec_module(module)
            return module

        loaded["streaming_transport"] = _load_leaf(
            "vllm_omni.metrics.streaming_transport",
            _METRICS_PKG / "streaming_transport.py",
        )
        for mod in ("manifests", "configuration_nemotron_asr", "session", "streaming"):
            loaded[mod] = _load_leaf(f"{_BASE}.{mod}", _PKG / f"{mod}.py")
        return loaded
    finally:
        mp.undo()  # restores ONLY the bare parent-package stubs set above


_MODULES = _load_chain()
_SESSION = _MODULES["session"]
_TRANSPORT = _MODULES["streaming_transport"]
NemotronRealtimeSession = _SESSION.NemotronRealtimeSession
ReceiptLedger = _SESSION.ReceiptLedger
StreamingObserver = _TRANSPORT.StreamingObserver
ChunkReadyHandle = _TRANSPORT.ChunkReadyHandle
buffer_stream = _MODULES["streaming"].buffer_stream

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PARK_ID = 13088
PLACEHOLDER_ID = 13089
PROMPTS = {"auto": 101, "en-US": 2}


def _hf(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "eos_token_id": PARK_ID,
        "audio_chunk_token_id": PLACEHOLDER_ID,
        "prompt_dictionary": dict(PROMPTS),
        "num_prompts": 128,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _session(**kwargs: Any) -> Any:
    return NemotronRealtimeSession.from_model_config(_hf(), **kwargs)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def _audio(*sizes: int) -> AsyncIterator[Any]:
    async def stream() -> AsyncIterator[Any]:
        for size in sizes:
            yield np.zeros(size, dtype=np.float32)

    return stream()


class _RecordingObserver:
    """A structurally-conforming fake with REAL per-session ready/in-flight
    tracking (correction 1's settled design) — needed to exercise, and
    prove coherent, the single-in-flight-handle park-correlation state
    machine, not merely record call arguments like the other events.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._waiting: dict[str, deque[Any]] = {}
        self._inflight: dict[str, Any] = {}

    def session_opened(self, *, cadence_ms: str) -> None:
        self.calls.append(("session_opened", {"cadence_ms": cadence_ms}))

    def session_finished(self, *, cadence_ms: str, reason: str) -> None:
        self.calls.append(("session_finished", {"cadence_ms": cadence_ms, "reason": reason}))

    def session_open_rejected(self, *, reason: str) -> None:
        self.calls.append(("session_open_rejected", {"reason": reason}))

    def accepted_audio_seconds(self, *, cadence_ms: str, seconds: float) -> None:
        self.calls.append(("accepted_audio_seconds", {"cadence_ms": cadence_ms, "seconds": seconds}))

    def unit_ready(
        self,
        *,
        session_key: str = "default",
        cadence_ms: str,
        chunk_type: str,
        ready_stamp_s: float,
    ) -> Any:
        handle = ChunkReadyHandle(
            session_key=session_key,
            cadence_ms=cadence_ms,
            chunk_type=chunk_type,
            ready_stamp_s=ready_stamp_s,
        )
        self._waiting.setdefault(session_key, deque()).append(handle)
        self.calls.append(
            (
                "unit_ready",
                {
                    "session_key": session_key,
                    "cadence_ms": cadence_ms,
                    "chunk_type": chunk_type,
                    "ready_stamp_s": ready_stamp_s,
                },
            )
        )
        return handle

    def unit_minted(self, handle: Any) -> None:
        waiting = self._waiting.get(handle.session_key)
        if waiting is not None and handle in waiting:
            waiting.remove(handle)
        self._inflight[handle.session_key] = handle
        self.calls.append(("unit_minted", {"handle": handle}))

    def complete_inflight(self, session_key: str) -> Any:
        handle = self._inflight.get(session_key)
        self._inflight[session_key] = None
        self.calls.append(("complete_inflight", {"session_key": session_key, "resolved": handle}))
        return handle

    def unit_parked(self, handle: Any, *, park_stamp_s: float) -> None:
        self.calls.append(("unit_parked", {"handle": handle, "park_stamp_s": park_stamp_s}))

    def unit_cleared(self, handle: Any, *, outcome: str) -> None:
        session_key = handle.session_key
        waiting = self._waiting.get(session_key)
        if waiting is not None and handle in waiting:
            waiting.remove(handle)
        if self._inflight.get(session_key) is handle:
            self._inflight.pop(session_key, None)
        self.calls.append(("unit_cleared", {"handle": handle, "outcome": outcome}))

    def overflow(self, *, kind: str) -> None:
        self.calls.append(("overflow", {"kind": kind}))

    def clear_all_outstanding(self, session_key: str, *, outcome: str) -> None:
        handles = list(self._waiting.get(session_key, ()))
        inflight = self._inflight.get(session_key)
        if inflight is not None:
            handles.append(inflight)
        for handle in handles:
            self.unit_cleared(handle, outcome=outcome)

    # ---- test-only introspection -------------------------------------
    def outstanding(self, session_key: str) -> list[Any]:
        return list(self._waiting.get(session_key, ()))


# ---- protocol shape (PORT-OBS-003) --------------------------------------------


# @spec PORT-OBS-003
def test_recording_fake_satisfies_the_streaming_observer_protocol() -> None:
    fake = _RecordingObserver()
    assert isinstance(fake, StreamingObserver)


# @spec PORT-OBS-003
def test_protocol_declares_every_documented_event_method() -> None:
    for name in (
        "session_opened",
        "session_finished",
        "session_open_rejected",
        "accepted_audio_seconds",
        "unit_ready",
        "unit_minted",
        "complete_inflight",
        "unit_parked",
        "unit_cleared",
        "overflow",
    ):
        assert hasattr(StreamingObserver, name), f"StreamingObserver missing {name}"


# @spec PORT-OBS-003
def test_protocol_no_longer_declares_pop_next_ready() -> None:
    """The REJECTED FIFO-correlation design (correction 1): a ready-order
    FIFO cannot serve as the park-correlation authority, so the rejected
    ``pop_next_ready`` method must not reappear on the protocol."""
    assert not hasattr(StreamingObserver, "pop_next_ready")


# ---- THE required interleaving test (P0 correction 1) — real, GREEN ----------
# Proves the settled single-in-flight-handle design against the FAKE's
# real tracking logic (a faithful reference for what the production
# adapter must implement) and demonstrates why a ready-order FIFO would
# misclassify the carrierless echo.


# @spec PORT-OBS-003
def test_park_correlation_interleaving_matches_the_settled_design() -> None:
    """ready A, ready B -> minted A -> park (completes A) -> carrierless
    echo (ignored; B still outstanding) -> minted B -> park (completes
    B); every handle gets exactly one disposition."""
    fake = _RecordingObserver()
    session_key = "sess-interleave"

    handle_a = fake.unit_ready(session_key=session_key, cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0)
    handle_b = fake.unit_ready(session_key=session_key, cadence_ms="560", chunk_type="regular", ready_stamp_s=0.56)
    assert fake.outstanding(session_key) == [handle_a, handle_b]

    fake.unit_minted(handle_a)
    assert fake.outstanding(session_key) == [handle_b], "B remains waiting once A is minted"

    resolved_a = fake.complete_inflight(session_key)
    assert resolved_a is handle_a
    fake.unit_parked(resolved_a, park_stamp_s=0.6)

    # Carrierless async park echo: no unit is in-flight (A already
    # completed, B never minted) -> must resolve to None, and B must
    # remain untouched. A ready-order FIFO would instead have wrongly
    # popped and completed B here.
    echo = fake.complete_inflight(session_key)
    assert echo is None
    assert fake.outstanding(session_key) == [handle_b], "B must still be outstanding after the ignored echo"

    fake.unit_minted(handle_b)
    assert fake.outstanding(session_key) == []
    resolved_b = fake.complete_inflight(session_key)
    assert resolved_b is handle_b
    fake.unit_parked(resolved_b, park_stamp_s=1.2)

    parked = [c for c in fake.calls if c[0] == "unit_parked"]
    assert len(parked) == 2
    disposed = {c[1]["handle"] for c in parked}
    assert disposed == {handle_a, handle_b}, "every handle received exactly one disposition"


# ---- inertness: no observer -> identical behavior (real, GREEN) --------------


# @spec PORT-OBS-003
def test_absent_observer_produces_baseline_behavior_with_zero_observer_traffic() -> None:
    """No observer supplied: the segmenter's full detection/delivery/
    budget path runs to completion with zero observer traffic (there is
    no observer object to call — this is the durable PORT-OBS-003
    contract: "absent observer -> pre-metrics baseline behavior",
    forever, not just pre-Phase-6), and produces byte-identical envelopes
    across two independent back-to-back runs (baseline determinism).

    Lead-authorized fix (Phase-6 round 2, Q1b): retires the sibling
    "supplied-but-unwired" leg this test used to carry — that leg pinned
    the Phase-5 stub's literal non-consumption of a supplied observer,
    an invariant that necessarily breaks once Phase 6 wires real
    observation (the point of this phase). The absent-observer leg below
    is what PORT-OBS-003 actually requires long-term.
    """

    async def run_without_observer() -> list[Any]:
        session = _session(with_ledger=True, observer=None)
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(8_960 * 2), queue, session)
        envelopes = []
        async for prompt in agen:
            if "multi_modal_data" in prompt:
                envelopes.append(prompt["multi_modal_data"]["audio"].copy())
            queue.put_nowait([PARK_ID])
        return envelopes

    first = _run(run_without_observer())
    second = _run(run_without_observer())
    assert len(first) == len(second) == 3  # 2 regular + 1 final-tail
    envelope_header_fields = _MODULES["manifests"].ENVELOPE_HEADER_FIELDS
    admission_slot = envelope_header_fields.index("admission_ms_mod")
    for a, b in zip(first, second, strict=True):
        # The header's admission_ms_mod slot is a genuine wall-clock stamp
        # (design §Ingress-deadline plumbing) — the two back-to-back runs
        # are expected to differ there by a millisecond or so; mask it out
        # so this test compares only observer-independent content.
        a_masked, b_masked = a.copy(), b.copy()
        a_masked[admission_slot] = 0.0
        b_masked[admission_slot] = 0.0
        np.testing.assert_array_equal(a_masked, b_masked)


# ---- regular ready at cadence completion (PORT-OBS-004) — RED ----------------


# @spec PORT-OBS-003, PORT-OBS-004
def test_regular_unit_ready_is_observed_at_cadence_completion() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        session = _session(with_ledger=True, observer=fake)
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(8_960 * 2), queue, session)
        async for prompt in agen:
            if "multi_modal_data" in prompt:
                queue.put_nowait([PARK_ID])
            else:
                queue.put_nowait([PARK_ID])
        return fake.calls

    calls = _run(scenario())
    ready_calls = [c for c in calls if c[0] == "unit_ready" and c[1]["chunk_type"] == "regular"]
    # Two regular cadences were completed; each should mint one ready event.
    assert len(ready_calls) == 2


# @spec PORT-OBS-003, PORT-OBS-004
def test_final_tail_ready_stamp_is_passed_through_verbatim() -> None:
    """The caller-captured finalize-acceptance stamp must reach the
    final-tail unit_ready call EXACTLY — never reconstructed inside the
    generator at resumption."""
    sentinel_stamp = 123456.789

    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        session = _session(with_ledger=True, observer=fake)
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(
            _audio(4_480),
            queue,
            session,
            final_tail_ready_stamp_s=sentinel_stamp,
        )
        async for prompt in agen:
            queue.put_nowait([PARK_ID])
        return fake.calls

    calls = _run(scenario())
    final_ready = [c for c in calls if c[0] == "unit_ready" and c[1]["chunk_type"] == "final_tail"]
    assert len(final_ready) == 1
    assert final_ready[0][1]["ready_stamp_s"] == sentinel_stamp


# ---- ledgerless session still emits ready (PORT-OBS-003) — RED ---------------


# @spec PORT-OBS-003
def test_ledgerless_session_emits_the_same_ready_events() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        # with_ledger=False: the native path constructs a bare, ledgerless
        # session — observation must not depend on the ledger being armed.
        session = _session(with_ledger=False, observer=fake)
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(8_960), queue, session)
        async for prompt in agen:
            queue.put_nowait([PARK_ID])
        return fake.calls

    calls = _run(scenario())
    ready_calls = [c for c in calls if c[0] == "unit_ready"]
    assert len(ready_calls) >= 1


# ---- terminal-disposition OWNERSHIP (PORT-OBS-003) -----------------------------
#
# buffer_stream/the segmenter does NOT own terminal observation — the
# native adapter (test_realtime_streaming_metrics.py) or the leased-path
# lease consumer (test_nemotron_session_observer.py) does. This file
# therefore asserts only READY-event emission (above); the "every ready
# unit gets exactly one park-or-clear disposition" invariant is pinned
# on the CONSUMER side in those two files, not here.


# @spec PORT-OBS-003
def test_clearing_a_unit_twice_is_idempotent() -> None:
    """Session-level abort/close terminal paths are idempotent, so a
    double-fire (e.g. an abort racing a natural end) must not double-clear
    the same unit. Wired through the real constructor params
    (``session.from_model_config(observer=...)``) rather than a bare
    disconnected ``ReceiptLedger()``, so a Phase-6 implementation that
    emits ``unit_cleared`` from the session's own ledger/observer pairing
    (``session.ledger`` / ``session.observer``) makes this pass —
    ``ReceiptLedger.fail()`` is already documented idempotent."""

    async def scenario() -> tuple[Any, _RecordingObserver]:
        fake = _RecordingObserver()
        session = _session(with_ledger=True, observer=fake)
        ledger = session.ledger
        assert session.observer is fake  # the real wiring point, not a bare ledger
        ticket = ledger.mint(final_tail=False, admission_ms_mod=0)
        error = RuntimeError("boom")
        ledger.fail(error)
        ledger.fail(error)  # idempotent no-op on the ledger itself
        return ticket, fake

    ticket, fake = _run(scenario())
    assert ticket.done.cancelled() is False
    ticket.done.exception()  # consume, silence "never retrieved" warning
    # Once wired, exactly one unit_cleared call should have fired despite
    # fail() being invoked twice.
    assert len([c for c in fake.calls if c[0] == "unit_cleared"]) == 1


# ---- overflow observed by kind (PORT-OBS-005) — RED ---------------------------


# @spec PORT-OBS-003, PORT-OBS-005
def test_carrier_ledger_overflow_is_observed_by_kind() -> None:
    """PORT's bounded accepted-audio queue (kind="input_queue") does not
    exist yet in this codebase slice — see the handoff report. The
    receipt ledger's own backlog bound (kind="carrier") IS implemented
    today (ReceiptLedger.mint's RuntimeError), so this pins that half of
    PORT-OBS-005's overflow contract against the real bound-trip. Wired
    through ``session.from_model_config(observer=..., max_pending_carriers=1)``
    — the session's OWN armed ledger — rather than a bare disconnected
    ``ReceiptLedger()``."""

    async def scenario() -> _RecordingObserver:
        fake: _RecordingObserver = _RecordingObserver()
        session = _session(with_ledger=True, observer=fake, max_pending_carriers=1)
        ledger = session.ledger
        assert session.observer is fake
        ledger.mint(final_tail=False, admission_ms_mod=0)
        with pytest.raises(RuntimeError):
            ledger.mint(final_tail=False, admission_ms_mod=1)
        # mypy narrows `fake`'s type via the preceding `is` comparison
        # against `session.observer: StreamingObserver | None`; a known
        # `is`-narrowing quirk, not a real type-safety issue (verified in
        # isolation) — see the class docstring for the identical
        # `unit_cleared` narrowing this file already works around.
        return fake  # type: ignore[no-any-return]

    fake = _run(scenario())
    overflow_calls = [c for c in fake.calls if c[0] == "overflow" and c[1]["kind"] == "carrier"]
    assert len(overflow_calls) == 1


# ---- native open pinned at observer-bearing session construction --------------
# (PORT-OBS-006) — RED. NOT RealtimeConnection.__init__ (that counts
# WebSockets, the exact thing PORT-OBS-006 forbids) — see
# test_realtime_streaming_metrics.py's connection-construction NEGATIVE
# check for the other half of this pin.


# @spec PORT-OBS-006
def test_session_construction_is_the_native_open_boundary() -> None:
    """Native open is observer-bearing model-session construction after
    successful validation and before engine request creation. Building a
    ``NemotronRealtimeSession`` (the classmethod every native/leased path
    funnels through) with an observer must report ``session_opened``
    exactly once, at construction — never later, never zero times."""
    fake = _RecordingObserver()
    session = _session(with_ledger=False, observer=fake)

    assert session.observer is fake
    opened = [c for c in fake.calls if c[0] == "session_opened"]
    assert opened == [("session_opened", {"cadence_ms": "560"})]


# @spec PORT-OBS-006
def test_construction_failure_reports_no_session_opened() -> None:
    """A construction that fails validation (unknown cadence) must not
    report session_opened — open is only for a SUCCESSFUL validated
    construction."""
    fake = _RecordingObserver()
    with pytest.raises(ValueError):
        NemotronRealtimeSession.from_model_config(_hf(), cadence="not-a-cadence", observer=fake)
    assert fake.calls == []


# ---- accepted-audio budget (PORT-SESS-001, amended Decision 1) ----------------
# native/segmenter-side cases. Leased-path cases (including "session
# then follows ordinary terminal clearing", a lease-consumer concern per
# this file's ownership fix above) live in test_nemotron_session_observer.py.


# @spec PORT-SESS-001
def test_accepted_audio_budget_defaults_to_thirty_seconds() -> None:
    session = _session(with_ledger=False)
    assert session.accepted_audio_budget_s == 30.0


# @spec PORT-SESS-001
def test_native_piece_exceeding_the_budget_is_rejected_whole_before_acceptance() -> None:
    """PORT-SESS-001 (amended): PORT's accepted-audio queue is a
    per-session SECONDS budget (default 30 s, ``accepted_audio_budget_s`),
    checked strictly BEFORE acceptance on the native path. A piece that
    would push accepted audio past a near-zero budget must be rejected
    atomically, before any of its complete cadences are accepted — never
    a partial/truncated accept."""

    async def scenario() -> _RecordingObserver:
        fake = _RecordingObserver()
        session = _session(with_ledger=True, observer=fake, accepted_audio_budget_s=0.001)
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(8_960 * 5), queue, session)
        async for _prompt in agen:
            queue.put_nowait([PARK_ID])
        return fake

    with pytest.raises(RuntimeError):
        _run(scenario())


# @spec PORT-SESS-001
def test_native_rejected_piece_accrues_no_accepted_audio_seconds() -> None:
    async def scenario() -> _RecordingObserver:
        fake = _RecordingObserver()
        session = _session(with_ledger=True, observer=fake, accepted_audio_budget_s=0.001)
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(8_960 * 5), queue, session)
        try:
            async for _prompt in agen:
                queue.put_nowait([PARK_ID])
        except RuntimeError:
            pass
        return fake

    fake = _run(scenario())
    assert [c for c in fake.calls if c[0] == "accepted_audio_seconds"] == []


# @spec PORT-SESS-001, PORT-OBS-005
def test_native_rejected_piece_emits_exactly_one_input_queue_overflow() -> None:
    async def scenario() -> _RecordingObserver:
        fake = _RecordingObserver()
        session = _session(with_ledger=True, observer=fake, accepted_audio_budget_s=0.001)
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(8_960 * 5), queue, session)
        try:
            async for _prompt in agen:
                queue.put_nowait([PARK_ID])
        except RuntimeError:
            pass
        return fake

    fake = _run(scenario())
    overflow = [c for c in fake.calls if c[0] == "overflow" and c[1]["kind"] == "input_queue"]
    assert len(overflow) == 1


# @spec PORT-SESS-001
def test_native_occupancy_lifecycle_never_caps_lifetime_cumulative_audio() -> None:
    """PORT-SESS-001 (amended): the budget bounds QUEUE OCCUPANCY only —
    audio drained by CHUNK consumption releases its budget, and lifetime
    cumulative session audio is never capped by it. Accept pieces to near
    the budget, drain them via CHUNK consumption (parking each ready
    unit), then accept another piece that would push LIFETIME cumulative
    audio over the budget but fits comfortably in the now-drained queue —
    it must be accepted. A duration-cap implementation (rejecting because
    lifetime total exceeds the budget) must FAIL this test."""

    async def scenario() -> int:
        fake = _RecordingObserver()
        # Budget covers exactly 2 regular cadences of queue occupancy.
        session = _session(with_ledger=True, observer=fake, accepted_audio_budget_s=1.12)
        queue: asyncio.Queue = asyncio.Queue()
        # 5 regular cadences total (5 * 0.56s = 2.8s), each drained via a
        # park before the next is fed — occupancy never exceeds 2
        # cadences at a time, but LIFETIME cumulative audio (2.8s) is
        # already well past the 1.12s budget by the third cadence.
        agen = buffer_stream(_audio(8_960, 8_960, 8_960, 8_960, 8_960), queue, session)
        accepted = 0
        async for prompt in agen:
            if "multi_modal_data" in prompt:
                accepted += 1
            queue.put_nowait([PARK_ID])  # drain immediately: occupancy releases
        return accepted

    accepted = _run(scenario())
    # All 5 regular cadences plus the final tail must be accepted; none
    # rejected for exceeding a (nonexistent) lifetime cap.
    assert accepted == 6


# ---- model package imports no prometheus_client (PORT-OBS-003) ---------------


# @spec PORT-OBS-003
def test_model_package_imports_no_prometheus_client() -> None:
    import re

    banned = re.compile(r"^\s*(import prometheus_client|from prometheus_client)", re.MULTILINE)
    offenders = []
    for path in sorted(_PKG.glob("*.py")):
        text = path.read_text()
        if banned.search(text):
            offenders.append(path.name)
    assert offenders == [], (
        f"nemotron_asr package files import prometheus_client: {offenders}; "
        "the observer protocol must stay Prometheus-free (PORT-OBS-003)"
    )
