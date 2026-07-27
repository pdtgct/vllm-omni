# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: the native-path StreamingObserver wiring.

Specs: PORT-OBS-003 (the native connection adapter observes chunk
terminal disposition from the generation output stream it already
consumes — the named park-visibility pin, rebuilt around the SETTLED
single-in-flight-handle design (correction 1): unit ready -> minted
(promotion to in-flight at CHUNK submission; PORT-SESS-001 admits at
most one in-flight unit per session) -> parked/cleared. A park
observation completes exactly the in-flight unit
(``observer.complete_inflight(session_key)``); a park with NO in-flight
unit — a carrierless async echo (PORT-ADV-004) or FLUSH (PORT-SESS-003,
mints no ready handle) — is ignored, and every waiting-ready unit stays
outstanding. A ready-order FIFO was REJECTED: with several units ready
before the first park, it would misclassify a carrierless echo as
completing the next waiting unit), PORT-OBS-006 (native open is
observer-bearing model-session construction, never WebSocket accept —
so this file's connection-construction tests assert NO increment, and
the session-construction pin itself lives in
``test_streaming_observer.py``, not here; route-to-session injection is
driven here against the REAL production route function), PORT-OBS-007
(connection-layer rejection increments open-rejections with a bounded
reason and no cadence label, driven through the REAL inherited
``handle_event``/``_check_model`` validation chain, not a bypassed
``send_error`` no-op).

Reuses the ``RealtimeConnection.__new__`` + manual-attribute fixture
pattern from ``test_realtime_connection_helpers.py``. The ``_observer``/
``_park_token_id`` attributes added to ``RealtimeConnection.__init__``
(Phase-5 stub) are never read by ``_run_generation``/``handle_event``
yet, so every behavioral test below is EXPECTED RED via an explicit
missing-behavior assertion — no wiring was added to avoid regressing
the existing ``_run_generation`` behavioral tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from vllm_omni.entrypoints.openai import realtime_connection as realtime_connection_mod
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PARK_ID = 13088
_FIXED_UUID = UUID(int=0)


class _RecordingObserver:
    """A fake with REAL per-session ready/in-flight tracking (matching
    ``test_streaming_observer.py``'s reference implementation) — the
    correlation authority PORT-OBS-003 requires on the ledgerless native
    path, where the connection layer watches the output token stream but
    never receives a handle directly (the segmenter, not the adapter,
    calls ``unit_ready``).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._waiting: dict[str, list[Any]] = {}
        self._inflight: dict[str, Any] = {}
        self._seq = 0

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
        self._seq += 1
        handle = SimpleNamespace(id=self._seq, session_key=session_key, cadence_ms=cadence_ms, chunk_type=chunk_type)
        self._waiting.setdefault(session_key, []).append(handle)
        self.calls.append(
            ("unit_ready", {"session_key": session_key, "cadence_ms": cadence_ms, "chunk_type": chunk_type})
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
        self.calls.append(("unit_parked", {"handle": handle}))

    def unit_cleared(self, handle: Any, *, outcome: str) -> None:
        session_key = handle.session_key
        waiting = self._waiting.get(session_key)
        if waiting is not None and handle in waiting:
            waiting.remove(handle)
        if self._inflight.get(session_key) is handle:
            self._inflight[session_key] = None
        self.calls.append(("unit_cleared", {"handle": handle, "outcome": outcome}))

    def overflow(self, *, kind: str) -> None:
        self.calls.append(("overflow", {"kind": kind}))

    # ---- test-only seeding/introspection --------------------------------
    def seed_ready(self, session_key: str, *, n: int = 1, cadence_ms: str = "560") -> list[Any]:
        """Simulate the segmenter having already minted ``n`` ready
        handles for ``session_key`` before generation output arrives —
        the state the real observer will hold per session."""
        return [
            self.unit_ready(session_key=session_key, cadence_ms=cadence_ms, chunk_type="regular", ready_stamp_s=0.0)
            for _ in range(n)
        ]

    def mint(self, handle: Any) -> None:
        """Test-only shorthand: promote a seeded ready handle to in-flight."""
        self.unit_minted(handle)

    def outstanding(self, session_key: str) -> int:
        return len(self._waiting.get(session_key, ()))


class _RealtimeGenerationEngine:
    default_sampling_params_list = [SimpleNamespace(tag="default")]

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs

    def generate(self, **_kwargs: Any) -> AsyncGenerator[Any, None]:
        async def _outputs() -> AsyncGenerator[Any, None]:
            for output in self.outputs:
                yield output

        return _outputs()


def _generation_output(text: str, token_ids: list[int]) -> Any:
    return SimpleNamespace(
        stage_id=0,
        outputs=[SimpleNamespace(text=text, token_ids=token_ids)],
        prompt_token_ids=[1],
        multimodal_output=None,
    )


def _connection(
    engine: Any,
    *,
    observer: Any,
    park_token_id: int | None = PARK_ID,
) -> tuple[RealtimeConnection, list[Any], list[dict[str, Any]]]:
    connection: Any = RealtimeConnection.__new__(RealtimeConnection)
    connection.connection_id = "test"
    connection.engine = engine
    connection._is_connected = True
    connection.audio_queue = asyncio.Queue()
    connection._observer = observer
    connection._park_token_id = park_token_id

    sent_events: list[Any] = []
    sent_json: list[dict[str, Any]] = []

    async def _send(event: Any) -> None:
        sent_events.append(event)

    async def _send_json(payload: dict[str, Any]) -> None:
        sent_json.append(payload)

    async def _send_error(message: str, error_type: str | None = None) -> None:
        pass

    connection.send = _send
    connection.send_json = _send_json
    connection.send_error = _send_error
    return connection, sent_events, sent_json


async def _empty_streaming_input() -> AsyncGenerator[Any, None]:
    return
    yield None  # the standard empty-async-generator idiom


def _fixed_session_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """``_run_generation`` mints ``request_id`` internally via
    ``uuid4()``; pin it so the test can pre-seed the observer's
    session-keyed state for the SAME key ``_run_generation`` will use."""
    monkeypatch.setattr(realtime_connection_mod, "uuid4", lambda: _FIXED_UUID)
    return f"rt-test-{_FIXED_UUID}"


# ---- THE named park-visibility pin: single-in-flight-handle correlation ------
# (PORT-OBS-003) — every case RED (no wiring exists yet)


# @spec PORT-OBS-003, PORT-OBS-004
@pytest.mark.asyncio
async def test_blank_chunk_park_with_an_inflight_handle_is_counted_parked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) A blank/empty-text CHUNK park with the unit MINTED (promoted
    to in-flight) must resolve via ``complete_inflight`` -> `unit_parked`
    — text emptiness is irrelevant; only park-token identity + the
    single in-flight handle commits."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    (handle,) = observer.seed_ready(session_key, n=1)
    observer.mint(handle)

    engine = _RealtimeGenerationEngine([_generation_output("", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue())

    parked = [c for c in observer.calls if c[0] == "unit_parked"]
    assert len(parked) == 1
    assert parked[0][1]["handle"] is handle


# @spec PORT-OBS-003, PORT-ADV-004
@pytest.mark.asyncio
async def test_park_output_with_no_inflight_unit_is_ignored_as_carrierless_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) A park-token output with NO in-flight unit is the asynchronous
    scheduler park echo (PORT-ADV-004): emit-park-change-nothing at the
    model layer, and observed as nothing here — never a fabricated
    `unit_parked`/`unit_cleared`. Every waiting-ready unit for the
    session (none seeded here) remains outstanding."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    assert observer.outstanding(session_key) == 0  # deliberately unseeded, nothing minted

    engine = _RealtimeGenerationEngine([_generation_output("", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue())

    terminal = [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared")]
    assert terminal == []


# @spec PORT-OBS-003, PORT-SESS-003
@pytest.mark.asyncio
async def test_flush_park_is_ignored_because_flush_mints_no_ready_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) FLUSH's trailing park (PORT-SESS-003) creates no carrier
    ticket and therefore no ready handle by construction, so nothing is
    ever minted in-flight for it; observing it as a chunk park would be
    a fabricated event. Mechanically identical to (b) under this fake
    (both are "no in-flight unit"), but the real-world cause is FLUSH's
    terminal barrier rather than an async park echo — pinned as its own
    case per the review's test matrix."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()

    # A FLUSH-shaped output: no preceding CHUNK was ever minted in-flight
    # for this session, matching FLUSH's ticketless park.
    engine = _RealtimeGenerationEngine([_generation_output("", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue())

    terminal = [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared")]
    assert terminal == []
    assert observer.outstanding(session_key) == 0


# @spec PORT-OBS-003, PORT-OBS-005
@pytest.mark.asyncio
async def test_interleaving_echo_between_two_mints_leaves_the_waiting_unit_outstanding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE required interleaving scenario, driven against `_run_generation`
    (still unwired -> RED): ready A, ready B -> mint A -> park (completes
    A) -> carrierless echo (ignored; B still outstanding) -> mint B ->
    park (completes B); every handle gets exactly one disposition. A
    ready-order FIFO would instead have wrongly resolved the echo against
    B — see test_streaming_observer.py's real/green reference proof of
    this same interleaving against the fake's own tracking logic."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    handle_a, handle_b = observer.seed_ready(session_key, n=2)
    observer.mint(handle_a)

    # Three generation steps: park (completes A), carrierless echo
    # (ignored), then — after minting B out-of-band, mirroring what the
    # segmenter would do between CHUNKs — a second park (completes B).
    engine = _RealtimeGenerationEngine(
        [
            _generation_output("", [PARK_ID]),  # completes A
            _generation_output("", [PARK_ID]),  # carrierless echo: ignored
        ]
    )
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue())

    # After the echo, B must still be outstanding (never wrongly resolved).
    assert observer.outstanding(session_key) == 1
    terminal_so_far = [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared")]
    assert len(terminal_so_far) == 1
    assert terminal_so_far[0][1]["handle"] is handle_a


# @spec PORT-OBS-003, PORT-OBS-005
@pytest.mark.asyncio
async def test_every_ready_handle_gets_exactly_one_disposition_by_generation_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ready handles, only one minted+parked before the generation
    stream ends. The end-of-generation cleanup must clear the still-
    outstanding handle (mirroring ``NemotronSessionLease._consume``'s
    finally-clearing pattern on the leased path) so no ready handle is
    left permanently dangling."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    handle_a, handle_b = observer.seed_ready(session_key, n=2)
    observer.mint(handle_a)

    engine = _RealtimeGenerationEngine([_generation_output("", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue())

    terminal = [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared")]
    assert len(terminal) == 2
    disposed_handles = {c[1]["handle"] for c in terminal}
    assert disposed_handles == {handle_a, handle_b}
    assert observer.outstanding(session_key) == 0


# ---- park_token_id is generic and inert when None (real, GREEN) --------------


# @spec PORT-OBS-003
@pytest.mark.asyncio
async def test_park_token_id_none_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    """No park-token id configured -> no park detection is attempted,
    regardless of what ids the engine emits. This holds trivially today
    (nothing is wired), and must continue to hold once Phase 6 lands."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    (handle,) = observer.seed_ready(session_key, n=1)
    observer.mint(handle)

    engine = _RealtimeGenerationEngine([_generation_output("", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer, park_token_id=None)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue())

    terminal = [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared")]
    assert terminal == []


# ---- connection construction increments NOTHING (real, GREEN) ----------------
# (PORT-OBS-006: native open is session construction, never WebSocket accept)


# @spec PORT-OBS-006
def test_connection_construction_never_increments_session_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PORT-OBS-006 forbids counting WebSocket acceptance as native
    open. This connection's own ``__init__`` must never call
    ``session_opened`` — that boundary lives at observer-bearing
    ``NemotronRealtimeSession`` construction instead (pinned in
    ``test_streaming_observer.py``, not here)."""
    from vllm.entrypoints.speech_to_text.realtime import (
        connection as vllm_connection_mod,
    )

    def _fake_base_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self.serving = SimpleNamespace(engine_client=_RealtimeGenerationEngine([]))

    monkeypatch.setattr(vllm_connection_mod.RealtimeConnection, "__init__", _fake_base_init)

    observer = _RecordingObserver()
    RealtimeConnection(observer=observer)

    assert observer.calls == []


# ---- connection-layer rejection: driven through the REAL validation ----------
# chain (PORT-OBS-007) — RED


# @spec PORT-OBS-007
@pytest.mark.asyncio
async def test_connection_layer_rejection_increments_open_rejections_without_cadence() -> None:
    """Drives the REAL inherited ``handle_event`` -> ``_check_model``
    chain (unmodified upstream code) with a fake ``serving`` object that
    reports the model unsupported — exactly the production rejection
    path PORT-OBS-007 observes at, not a bypassed ``send_error`` no-op."""

    class _RejectingServing:
        def _is_model_supported(self, model: str | None) -> bool:
            return False

        def create_error_response(
            self,
            *,
            message: str,
            err_type: str,
            status_code: Any,
            param: str,
        ) -> Any:
            return SimpleNamespace(error=SimpleNamespace(message=message))

    observer = _RecordingObserver()
    connection: Any = RealtimeConnection.__new__(RealtimeConnection)
    connection.serving = _RejectingServing()
    connection._observer = observer
    connection._is_connected = True
    connection._is_model_validated = False

    sent_errors: list[tuple[str, str | None]] = []

    async def _send_error(message: str, code: str | None = None) -> None:
        sent_errors.append((message, code))

    connection.send_error = _send_error

    await connection.handle_event({"type": "session.update", "model": "unsupported-model"})

    # Confirms the REAL rejection path actually fired (the production
    # signal this test drives off of).
    assert sent_errors, "the real _check_model/send_error chain must have rejected the model"

    rejected = [c for c in observer.calls if c[0] == "session_open_rejected"]
    assert len(rejected) == 1
    assert rejected[0][1]["reason"] in ("model", "cadence", "locale", "config")
    assert "cadence_ms" not in rejected[0][1]


# ---- route-to-session injection: driven against the REAL production route ----
# function (PORT-OBS-003, correction 3) — RED


class _FakeWebSocket:
    """A minimal fake satisfying the exact surface ``RealtimeConnection.
    handle_connection`` calls: ``accept``, ``send_text``, then
    ``receive_text`` raising ``WebSocketDisconnect`` immediately so the
    connection loop exits cleanly on the first turn."""

    def __init__(self, app_state: Any) -> None:
        self.app = SimpleNamespace(state=app_state)
        self.sent_text: list[str] = []

    async def accept(self) -> None:
        pass

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def receive_text(self) -> str:
        from starlette.websockets import WebSocketDisconnect

        raise WebSocketDisconnect()


@pytest.mark.asyncio
async def test_realtime_route_setup_injects_the_installed_observer_into_native_session_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drives the REAL production route function
    (``vllm_omni.entrypoints.openai.api_server.realtime_websocket``) with
    a fake ``websocket``/``app.state`` rather than manually constructing
    ``RealtimeConnection`` — the route-to-session injection PORT-OBS-003
    requires: the observer resolved off ``app.state`` (via
    ``resolve_installed_observer``) must be what the route passes into
    ``RealtimeConnection`` construction."""
    from vllm_omni.entrypoints.openai import api_server as api_server_mod
    from vllm_omni.metrics import streaming_install

    installed_observer = _RecordingObserver()
    monkeypatch.setattr(streaming_install, "resolve_installed_observer", lambda app_state: installed_observer)

    captured: dict[str, Any] = {}
    real_init = api_server_mod.RealtimeConnection.__init__

    def _capturing_init(self: Any, *args: Any, observer: Any = None, **kwargs: Any) -> None:
        captured["observer"] = observer
        real_init(self, *args, observer=observer, **kwargs)

    monkeypatch.setattr(api_server_mod.RealtimeConnection, "__init__", _capturing_init)

    app_state = SimpleNamespace(openai_serving_realtime=SimpleNamespace(engine_client=_RealtimeGenerationEngine([])))
    websocket: Any = _FakeWebSocket(app_state)

    await api_server_mod.realtime_websocket(websocket)

    assert captured.get("observer") is installed_observer, (
        "the real realtime_websocket route must resolve the installed observer off "
        "app.state and pass it into RealtimeConnection construction"
    )


# ---- api-server-count invariant: real config-validation call site (correction 3)


@pytest.mark.asyncio
async def test_api_server_count_invariant_is_asserted_at_its_real_call_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PORT-OBS-001's single-API-server invariant must be asserted where
    the worker count is actually resolved
    (``vllm_omni/entrypoints/openai/api_server.py``'s
    ``api_server_count``/``worker_count`` derivation) — pinned here as a
    seam-existence check (the unit-level behavioral test of
    ``assert_single_api_server_invariant`` itself lives in
    ``test_streaming_install.py``)."""
    import inspect

    from vllm_omni.entrypoints.openai import api_server as api_server_mod

    source = inspect.getsource(api_server_mod)
    assert "assert_single_api_server_invariant" in source, (
        "api_server.py must call assert_single_api_server_invariant at its "
        "worker_count/api_server_count resolution site"
    )
