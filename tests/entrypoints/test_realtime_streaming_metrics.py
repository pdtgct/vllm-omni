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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from vllm_omni.entrypoints.openai import realtime_connection as realtime_connection_mod
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.entrypoints.session_lifecycle import SessionLifecycleDeadline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PARK_ID = 13088
_FIXED_UUID = UUID(int=0)


@dataclass(frozen=True)
class _FakeHandle:
    """A hashable fake handle (Phase-6 round 2, Q1d, lead-authorized):
    frozen — same as round-3 F6 already did for the real
    ``ChunkReadyHandle`` — so callers may key sets/dicts by handle
    identity (e.g. "every ready handle got exactly one disposition",
    checked over a ``set`` of disposed handles)."""

    id: int
    session_key: str
    cadence_ms: str
    chunk_type: str


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

    def session_opened(self, *, session_key: str, cadence_ms: str) -> None:
        self.calls.append(("session_opened", {"session_key": session_key, "cadence_ms": cadence_ms}))

    def session_finished(self, *, session_key: str, reason: str) -> None:
        self.calls.append(("session_finished", {"session_key": session_key, "reason": reason}))

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
        handle = _FakeHandle(id=self._seq, session_key=session_key, cadence_ms=cadence_ms, chunk_type=chunk_type)
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

    def clear_all_outstanding(self, session_key: str, *, outcome: str) -> int:
        handles = list(self._waiting.get(session_key, ()))
        inflight = self._inflight.get(session_key)
        if inflight is not None:
            handles.append(inflight)
        for handle in handles:
            self.unit_cleared(handle, outcome=outcome)
        return len(handles)

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
                # Lead-authorized extension (Phase-6 round 2, Q1c): a
                # callable step is a scripted out-of-band side effect
                # (e.g. minting a handle between generation outputs,
                # mirroring what the segmenter does between CHUNKs) run
                # synchronously in-stream rather than yielded as an
                # output — existing callers that only pass output
                # objects are unaffected.
                if callable(output):
                    output()
                    continue
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

    async def _inert_expiry(_kind: str) -> None:
        return None

    connection._session_lifecycle = SessionLifecycleDeadline(
        idle_timeout_s=None,
        finalization_timeout_s=None,
        on_expire=_inert_expiry,
    )

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

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

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

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

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

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    terminal = [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared")]
    assert terminal == []
    assert observer.outstanding(session_key) == 0


# @spec PORT-OBS-003, PORT-OBS-005
@pytest.mark.asyncio
async def test_interleaving_echo_between_two_mints_leaves_the_waiting_unit_outstanding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE required interleaving scenario, driven end-to-end against
    `_run_generation`: ready A, ready B -> mint A -> park (completes A)
    -> carrierless echo (ignored; B still outstanding, resolved to
    `None`, never wrongly completing B) -> mint B out-of-band (mirroring
    what the segmenter does between CHUNKs) -> park (completes B); every
    handle gets exactly one disposition, and both are `unit_parked` (no
    clears). A ready-order FIFO would instead have wrongly resolved the
    echo against B — see test_streaming_observer.py's real/green
    reference proof of this same interleaving against the fake's own
    tracking logic.

    Lead-authorized fix (Phase-6 round 2, Q1c): the fixture now actually
    implements the third step ("mint B out-of-band... a second park
    completes B") its docstring always described but never scripted —
    the out-of-band mint is a scripted callable step in the generation
    stream, giving a natural mid-stream checkpoint (asserted inside the
    callable, before B is minted) as well as the final end-state.
    """
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    handle_a, handle_b = observer.seed_ready(session_key, n=2)
    observer.mint(handle_a)

    def _mint_b_after_echo_mid_stream() -> None:
        # Mid-stream checkpoint: after A's park and the carrierless
        # echo, B must still be outstanding — parked ONLY by the later
        # park below, never by the echo.
        assert observer.outstanding(session_key) == 1
        terminal_so_far = [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared")]
        assert len(terminal_so_far) == 1
        assert terminal_so_far[0][1]["handle"] is handle_a
        observer.mint(handle_b)

    engine = _RealtimeGenerationEngine(
        [
            _generation_output("", [PARK_ID]),  # completes A
            _generation_output("", [PARK_ID]),  # carrierless echo: ignored
            _mint_b_after_echo_mid_stream,  # out-of-band mint + mid-stream assertions
            _generation_output("", [PARK_ID]),  # completes B
        ]
    )
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    # End state: A parked, B parked, no clears, nothing outstanding.
    terminal = [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared")]
    assert len(terminal) == 2
    assert all(c[0] == "unit_parked" for c in terminal)
    assert terminal[0][1]["handle"] is handle_a
    assert terminal[1][1]["handle"] is handle_b
    assert observer.outstanding(session_key) == 0


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

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    terminal = [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared")]
    assert len(terminal) == 2
    disposed_handles = {c[1]["handle"] for c in terminal}
    assert disposed_handles == {handle_a, handle_b}
    assert observer.outstanding(session_key) == 0


# ---- generation-end cleanup outcome is cause-mapped (lead-settled, Q&A round 2) -----


# @spec PORT-OBS-005, PORT-OBS-006
@pytest.mark.asyncio
async def test_clean_end_with_outstanding_units_is_lifecycle_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Amended (A27 topology cascade): an ostensibly normal end with
    still-outstanding units is lifecycle divergence — the remaining
    work is cleared as "error" and the session finishes as "error",
    never "completed"."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    observer.seed_ready(session_key, n=1)  # never minted, never parked

    engine = _RealtimeGenerationEngine([])  # generation ends immediately, cleanly
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    cleared = [c for c in observer.calls if c[0] == "unit_cleared"]
    assert len(cleared) == 1
    assert cleared[0][1]["outcome"] == "error"
    finishes = [c for c in observer.calls if c[0] == "session_finished"]
    assert [f[1]["reason"] for f in finishes] == ["error"]
    assert finishes[0][1]["session_key"] == session_key


# @spec PORT-OBS-005, PORT-OBS-006
@pytest.mark.asyncio
async def test_generation_end_cleanup_clears_as_error_on_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lead-settled (Phase-6 round 2): the exception path clears any
    still-outstanding units with outcome "error"."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    observer.seed_ready(session_key, n=1)  # never minted, never parked

    class _RaisingEngine:
        default_sampling_params_list = [SimpleNamespace(tag="default")]

        def generate(self, **_kwargs: Any) -> AsyncGenerator[Any, None]:
            async def _outputs() -> AsyncGenerator[Any, None]:
                raise RuntimeError("boom")
                yield  # pragma: no cover - unreachable, satisfies the generator shape

            return _outputs()

    connection, _sent_events, _sent_json = _connection(_RaisingEngine(), observer=observer)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    cleared = [c for c in observer.calls if c[0] == "unit_cleared"]
    assert len(cleared) == 1
    assert cleared[0][1]["outcome"] == "error"


# ---- park_token_id is generic and inert when None (real, GREEN) --------------


# @spec PORT-OBS-003
@pytest.mark.asyncio
async def test_park_token_id_none_is_inert_for_detection_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Amended (A27 topology cascade): no park-token id -> no park
    DETECTION is attempted, but cleanup and lifecycle completion depend
    only on the observer/session key — outstanding units are still
    cleared and the session still finishes."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    (handle,) = observer.seed_ready(session_key, n=1)
    observer.mint(handle)

    engine = _RealtimeGenerationEngine([_generation_output("", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer, park_token_id=None)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    assert [c for c in observer.calls if c[0] == "unit_parked"] == []
    cleared = [c for c in observer.calls if c[0] == "unit_cleared"]
    assert len(cleared) == 1  # the minted handle, cleared at generation end
    finishes = [c for c in observer.calls if c[0] == "session_finished"]
    assert len(finishes) == 1
    assert finishes[0][1]["session_key"] == session_key


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


# ---- streaming-path-only install gate (PORT-OBS-001/002, lead-authorized fix) ------
#
# Drives the REAL ``api_server._install_streaming_observer_and_build_realtime_
# serving`` (the exact call site inlined into ``omni_init_app_state``,
# extracted to a standalone function specifically so this gate is
# independently testable without driving that ~900-line function's full
# engine/model/tool-server setup).


def _fake_state_with_registry(model_name: str = "gate-test-model", *, log_stats: bool = True) -> Any:
    registry = SimpleNamespace(model_name=lambda: model_name)
    return SimpleNamespace(openai_serving_models=registry, log_stats=log_stats)


def _fake_engine_client() -> Any:
    """Minimal fake satisfying OpenAIServing.__init__'s real attribute
    reads (model_config/renderer/input_processor; vllm_config is
    fingerprint-optional and safely absent)."""
    return SimpleNamespace(
        model_config=SimpleNamespace(),
        renderer=None,
        input_processor=None,
        vllm_config=None,
    )


# @spec PORT-OBS-001, PORT-OBS-003
def test_install_and_serving_construction_are_unconditional_at_the_pin() -> None:
    """The install seam takes no task vocabulary and always installs.

    Regression pin (2026-07-28 GPU round): a prior revision gated this
    seam on ``"realtime" in supported_tasks``, porting upstream vLLM's
    ``factories.py`` condition across the engine boundary. But
    ``AsyncOmniEngine`` derives its task vocabulary only from
    ``{"generate", "speech"}`` (is_comprehension / audio final output),
    so the gate was False in every real deployment: ``/v1/realtime``
    answered "Realtime API is not available" and no observer was ever
    installed — while unit tests passed by feeding the seam a
    ``"realtime"`` membership the engine can never produce. The seam
    now matches the pre-metrics base (unconditional construction) and
    PORT-OBS-003's actual contract: one observer per serving app state.
    """
    from vllm_omni.entrypoints.openai import api_server as api_server_mod
    from vllm_omni.metrics import streaming_install

    state = _fake_state_with_registry()
    api_server_mod._install_streaming_observer_and_build_realtime_serving(
        state,
        _fake_engine_client(),
        request_logger=None,
    )

    installed = streaming_install.resolve_installed_observer(state)
    assert installed is not None
    assert state.openai_serving_realtime is not None
    assert state.openai_serving_realtime._observer is installed


# @spec PORT-OBS-001, PORT-OBS-009
def test_duplicate_install_fails_loudly_rather_than_silently_rebinding() -> None:
    state = _fake_state_with_registry()
    engine_client = _fake_engine_client()
    from vllm_omni.entrypoints.openai import api_server as api_server_mod

    api_server_mod._install_streaming_observer_and_build_realtime_serving(
        state,
        engine_client,
        request_logger=None,
    )
    with pytest.raises(RuntimeError, match="already installed"):
        api_server_mod._install_streaming_observer_and_build_realtime_serving(
            state,
            engine_client,
            request_logger=None,
        )


# @spec PORT-OBS-008, PORT-OBS-009
def test_batch_stat_sink_attaches_through_the_engines_orchestrator_binding() -> None:
    """The sink attach reads ``engine_client.orchestrator`` — the
    binding ``AsyncOmniEngine``'s bootstrap thread now sets before the
    engine reports ready. Regression pin (2026-07-28 GPU round): the
    ``Orchestrator`` used to be a bootstrap-closure local that was
    never bound to the engine, so ``getattr(engine, "orchestrator",
    None)`` silently skipped the attach in every real serve and
    ``chunk_batch_size`` could never record. An engine exposing no
    binding must degrade observability only — never fail app-state
    init."""
    from vllm_omni.entrypoints.openai import api_server as api_server_mod
    from vllm_omni.metrics import streaming_install

    class _RecordingOrchestrator:
        def __init__(self) -> None:
            self.received: Any = None

        def set_streaming_metrics(self, metrics: Any) -> None:
            self.received = metrics

    state = _fake_state_with_registry()
    engine_client = _fake_engine_client()
    engine_client.orchestrator = _RecordingOrchestrator()

    api_server_mod._install_streaming_observer_and_build_realtime_serving(
        state,
        engine_client,
        request_logger=None,
    )

    installed = streaming_install.resolve_installed_observer(state)
    assert installed is not None
    assert engine_client.orchestrator.received is installed.metrics

    # Absent binding: install + serving still complete (already covered
    # implicitly above via _fake_engine_client(), asserted explicitly
    # here for the no-raise contract).
    bare_state = _fake_state_with_registry(model_name="no-orch-model")
    api_server_mod._install_streaming_observer_and_build_realtime_serving(
        bare_state,
        _fake_engine_client(),
        request_logger=None,
    )
    assert bare_state.openai_serving_realtime is not None


# @spec PORT-OBS-007, PORT-OBS-010
def test_manager_projection_uses_the_same_installed_metrics_instance() -> None:
    """The persistent-state service receives the app-owned sink, never a
    second metrics wrapper with a different model identity or log-stats flag."""
    from vllm_omni.entrypoints.openai import api_server as api_server_mod
    from vllm_omni.metrics import streaming_install

    class _RecordingService:
        def __init__(self) -> None:
            self.received: Any = None
            self.runtime_config = SimpleNamespace(accepted_audio_budget_s=30.0)

        def install_metrics(self, metrics: Any) -> None:
            self.received = metrics

    state = _fake_state_with_registry(model_name="state-projection-model")
    state.persistent_state_service = _RecordingService()

    api_server_mod._install_streaming_observer_and_build_realtime_serving(
        state,
        _fake_engine_client(),
        request_logger=None,
    )

    installed = streaming_install.resolve_installed_observer(state)
    assert installed is not None
    assert state.persistent_state_service.received is installed.metrics
    assert (
        state.openai_serving_realtime.runtime_config
        is state.persistent_state_service.runtime_config
    )


# @spec PORT-INT-005, PORT-SESS-005
def test_realtime_serving_accepts_one_immutable_runtime_envelope() -> None:
    import inspect

    from vllm_omni.entrypoints.openai.serving_realtime import (
        NemotronServingRealtime,
    )

    parameters = inspect.signature(NemotronServingRealtime.__init__).parameters
    assert "runtime_config" in parameters
    assert {
        "accepted_audio_budget_s",
        "accepted_audio_capacity_samples",
        "max_retained_transcript_bytes",
        "max_session_duration_s",
        "session_configuration_timeout_s",
        "session_idle_timeout_s",
        "session_finalization_timeout_s",
    }.isdisjoint(parameters)


def test_async_omni_engine_declares_the_orchestrator_binding() -> None:
    """`AsyncOmniEngine.__init__` must initialize ``self.orchestrator``
    and the bootstrap must bind the constructed instance — the attach
    seam above depends on this attribute existing on the real engine,
    which is exactly what the pre-fix code lacked."""
    import inspect

    from vllm_omni.engine import async_omni_engine as eng_mod

    src = inspect.getsource(eng_mod.AsyncOmniEngine)
    assert "self.orchestrator: Orchestrator | None = None" in src
    assert "self.orchestrator = orchestrator" in src


# @spec PORT-OBS-008, PORT-OBS-009
def test_async_omni_wrapper_passes_the_orchestrator_binding_through() -> None:
    """The API server's engine client is ``AsyncOmni`` (the wrapper),
    not ``AsyncOmniEngine`` — ``build_async_omni`` yields the wrapper.
    The sink attach therefore needs the wrapper to mirror the engine's
    ``orchestrator`` binding, as a live property (the engine binds it
    on its bootstrap thread; an ``__init__`` snapshot would race).

    Regression pin (2026-07-28 GPU re-run): with the binding only on
    the engine, the real serve logged 'sink NOT attached' — the seam
    read the wrapper, which had no passthrough."""
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    wrapper: Any = AsyncOmni.__new__(AsyncOmni)
    sentinel = object()
    wrapper.engine = SimpleNamespace(orchestrator=None)
    assert wrapper.orchestrator is None
    wrapper.engine.orchestrator = sentinel
    assert wrapper.orchestrator is sentinel


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


# ---- native serving-level injection: NemotronServingRealtime (Phase-6 round 2, Q2) --
#
# Lead-decided (Q2): no contextvar, no vLLM-core change. A fork-owned
# ``NemotronServingRealtime`` subclass overrides ``transcribe_realtime``
# with the same body as upstream, threading observer/budget into the
# widened ``buffer_realtime_audio`` optional keyword-only params. These
# tests drive the REAL override end-to-end (a fake ``model_cls`` in
# place of the actual Nemotron model class, since only the reach of the
# kwargs is being proven here — the model package's own
# ``buffer_realtime_audio``/``buffer_stream`` wiring is proven
# separately in test_streaming_observer.py).


class _FakeBufferRealtimeModelCls:
    """Records the kwargs ``transcribe_realtime`` passes through."""

    captured: dict[str, Any]

    def __init__(self) -> None:
        self.captured = {}

    async def buffer_realtime_audio(
        self,
        audio_stream: Any,
        input_stream: Any,
        model_config: Any,
        *,
        observer: Any = None,
        accepted_audio_budget_s: float | None = None,
        session_key: str | None = None,
    ) -> AsyncGenerator[Any, None]:
        self.captured["observer"] = observer
        self.captured["accepted_audio_budget_s"] = accepted_audio_budget_s
        self.captured["session_key"] = session_key
        self.captured["model_config"] = model_config
        return
        yield  # pragma: no cover - the standard empty-async-generator idiom


def _serving_realtime(*, observer: Any, accepted_audio_budget_s: float | None) -> Any:
    """Construct ``NemotronServingRealtime`` bypassing ``OpenAIServing.
    __init__``'s vLLM-core base-class plumbing (the same ``__new__`` +
    manual-attribute pattern this file already uses for
    ``RealtimeConnection``) — only the OVERRIDDEN ``transcribe_realtime``
    method is under test here, not upstream's own constructor."""
    from vllm_omni.entrypoints.openai.serving_realtime import (
        NemotronServingRealtime,
    )

    serving: Any = NemotronServingRealtime.__new__(NemotronServingRealtime)
    serving.model_config = SimpleNamespace()
    serving.renderer = None
    fake_model_cls = _FakeBufferRealtimeModelCls()
    # ``model_cls`` is a ``functools.cached_property`` on the base class;
    # pre-seeding the instance ``__dict__`` short-circuits it without
    # touching vLLM's real model registry.
    serving.__dict__["model_cls"] = fake_model_cls
    serving._observer = observer
    serving.runtime_config = (
        None
        if accepted_audio_budget_s is None
        else SimpleNamespace(
            accepted_audio_budget_s=accepted_audio_budget_s
        )
    )
    return serving, fake_model_cls


async def _empty_audio_stream() -> AsyncGenerator[Any, None]:
    return
    yield None  # pragma: no cover - the standard empty-async-generator idiom


# @spec PORT-OBS-003, PORT-SESS-001
@pytest.mark.asyncio
async def test_serving_transcribe_realtime_threads_observer_and_budget_end_to_end() -> None:
    """The installed observer and configured budget must reach
    ``buffer_realtime_audio`` exactly — proving the full native-path
    injection chain the lead's Q2 decision established."""
    observer = _RecordingObserver()
    serving, fake_model_cls = _serving_realtime(observer=observer, accepted_audio_budget_s=12.5)

    async for _ in serving.transcribe_realtime(_empty_audio_stream(), asyncio.Queue()):
        pass  # pragma: no cover - the fake yields nothing

    assert fake_model_cls.captured["observer"] is observer
    assert fake_model_cls.captured["accepted_audio_budget_s"] == 12.5


# @spec PORT-OBS-003, PORT-SESS-001
@pytest.mark.asyncio
async def test_serving_transcribe_realtime_is_inert_with_no_observer_installed() -> None:
    """No observer installed (the common case for a server that never
    streams Nemotron-ASR audio) -> `None`/`None` reaches
    ``buffer_realtime_audio`` exactly like calling it with no kwargs at
    all — the widened signature changes nothing for an uninstrumented
    deployment."""
    serving, fake_model_cls = _serving_realtime(observer=None, accepted_audio_budget_s=None)

    async for _ in serving.transcribe_realtime(_empty_audio_stream(), asyncio.Queue()):
        pass  # pragma: no cover - the fake yields nothing

    assert fake_model_cls.captured["observer"] is None
    assert fake_model_cls.captured["accepted_audio_budget_s"] is None


# @spec PORT-OBS-003
@pytest.mark.asyncio
async def test_unwidened_model_receives_the_exact_upstream_call() -> None:
    """A model whose ``buffer_realtime_audio`` carries the un-widened
    upstream signature (``qwen3_omni`` at the pin, and every upstream
    ``SupportsRealtime`` implementation) must receive the exact
    upstream call — no widened kwargs.

    Regression pin (2026-07-28 GPU round, latent behind the dead
    install gate): the override passed ``observer``/
    ``accepted_audio_budget_s`` unconditionally, a ``TypeError`` for
    any non-Nemotron realtime model the pre-metrics base served fine.
    Only a signature that explicitly declares both params opts in.
    """

    class _UnwidenedModelCls:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def buffer_realtime_audio(
            self, audio_stream: Any, input_stream: Any, model_config: Any
        ) -> AsyncGenerator[Any, None]:
            self.calls.append({"model_config": model_config})
            return
            yield  # pragma: no cover - the standard empty-async-generator idiom

    from vllm_omni.entrypoints.openai.serving_realtime import (
        NemotronServingRealtime,
    )

    serving: Any = NemotronServingRealtime.__new__(NemotronServingRealtime)
    serving.model_config = SimpleNamespace()
    serving.renderer = None
    model_cls = _UnwidenedModelCls()
    serving.__dict__["model_cls"] = model_cls
    serving._observer = _RecordingObserver()
    serving.runtime_config = SimpleNamespace(accepted_audio_budget_s=30.0)

    # Must not raise TypeError despite an installed observer + budget.
    async for _ in serving.transcribe_realtime(_empty_audio_stream(), asyncio.Queue()):
        pass  # pragma: no cover - the fake yields nothing

    assert len(model_cls.calls) == 1


# ---------------------------------------------------------------------------
# A27 topology cascade: correlation identity, terminal-reason taxonomy,
# model-gated park-token resolution (lead-approved amendments 1, 3, 4).
# ---------------------------------------------------------------------------


class _CapturingServing:
    """Fake serving that records the session_key it is handed."""

    def __init__(self) -> None:
        self.session_keys: list[Any] = []

    def transcribe_realtime(
        self, audio_stream: Any, input_stream: Any, *, session_key: str | None = None
    ) -> Any:
        self.session_keys.append(session_key)
        return _empty_streaming_input()


# @spec PORT-OBS-003
@pytest.mark.asyncio
async def test_start_generation_mints_one_identity_for_serving_and_engine() -> None:
    """Amendment 1: the engine request id is minted once in
    start_generation, handed to transcribe_realtime as session_key, and
    handed to the engine as request_id — one identity, both paths."""

    captured_request_ids: list[str] = []

    class _IdCapturingEngine:
        default_sampling_params_list = [SimpleNamespace(tag="default")]

        def generate(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
            captured_request_ids.append(kwargs["request_id"])

            async def _outputs() -> AsyncGenerator[Any, None]:
                return
                yield  # pragma: no cover

            return _outputs()

    serving = _CapturingServing()
    connection: Any = RealtimeConnection.__new__(RealtimeConnection)
    connection.connection_id = "identity-test"
    connection.engine = _IdCapturingEngine()
    connection.serving = serving
    connection._is_connected = True
    connection.audio_queue = asyncio.Queue()
    connection._observer = None
    connection._park_token_id = None
    connection.generation_task = None

    async def _inert_expiry(_kind: str) -> None:
        return None

    connection._session_lifecycle = SessionLifecycleDeadline(
        idle_timeout_s=None,
        finalization_timeout_s=None,
        on_expire=_inert_expiry,
    )

    async def _audio() -> AsyncGenerator[Any, None]:
        return
        yield  # pragma: no cover

    connection.audio_stream_generator = _audio

    async def _send(event: Any) -> None:
        pass

    connection.send = _send
    connection.send_json = _send
    connection.send_error = _send

    await connection.start_generation()
    assert connection.generation_task is not None
    await connection.generation_task

    assert len(serving.session_keys) == 1
    assert len(captured_request_ids) == 1
    assert serving.session_keys[0] == captured_request_ids[0]
    assert str(serving.session_keys[0]).startswith("rt-identity-test-")


# @spec PORT-OBS-006
@pytest.mark.asyncio
async def test_completed_reason_requires_exhaustion_with_zero_outstanding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal exhaustion with every unit already disposed -> exactly one
    session_finished with reason "completed" and no divergence clears."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    (handle,) = observer.seed_ready(session_key, n=1)
    observer.mint(handle)

    engine = _RealtimeGenerationEngine([_generation_output("hi", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    parked = [c for c in observer.calls if c[0] == "unit_parked"]
    assert len(parked) == 1
    finishes = [c for c in observer.calls if c[0] == "session_finished"]
    assert [f[1]["reason"] for f in finishes] == ["completed"]
    assert finishes[0][1]["session_key"] == session_key
    assert [c for c in observer.calls if c[0] == "unit_cleared"] == []


# @spec PORT-OBS-006
@pytest.mark.asyncio
async def test_cancelled_error_records_aborted_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Amendment 4: asyncio.CancelledError is handled separately —
    recorded as "aborted", then re-raised."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    observer.seed_ready(session_key, n=1)

    class _CancelledEngine:
        default_sampling_params_list = [SimpleNamespace(tag="default")]

        def generate(self, **_kwargs: Any) -> AsyncGenerator[Any, None]:
            async def _outputs() -> AsyncGenerator[Any, None]:
                raise asyncio.CancelledError()
                yield  # pragma: no cover

            return _outputs()

    connection, _sent_events, _sent_json = _connection(_CancelledEngine(), observer=observer)

    with pytest.raises(asyncio.CancelledError):
        await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    cleared = [c for c in observer.calls if c[0] == "unit_cleared"]
    assert [c[1]["outcome"] for c in cleared] == ["aborted"]
    finishes = [c for c in observer.calls if c[0] == "session_finished"]
    assert [f[1]["reason"] for f in finishes] == ["aborted"]


# @spec PORT-OBS-006
@pytest.mark.asyncio
async def test_engine_error_finishes_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    observer.seed_ready(session_key, n=1)

    class _RaisingEngine:
        default_sampling_params_list = [SimpleNamespace(tag="default")]

        def generate(self, **_kwargs: Any) -> AsyncGenerator[Any, None]:
            async def _outputs() -> AsyncGenerator[Any, None]:
                raise RuntimeError("engine boom")
                yield  # pragma: no cover

            return _outputs()

    connection, _sent_events, _sent_json = _connection(_RaisingEngine(), observer=observer)
    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    finishes = [c for c in observer.calls if c[0] == "session_finished"]
    assert [f[1]["reason"] for f in finishes] == ["error"]
    assert finishes[0][1]["session_key"] == session_key


# @spec PORT-OBS-006
@pytest.mark.asyncio
async def test_client_disconnect_finishes_aborted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop's disconnect break is an abort, not an error and not a
    completion."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    (handle,) = observer.seed_ready(session_key, n=1)
    observer.mint(handle)

    connection_holder: list[Any] = []

    def _disconnect() -> None:
        connection_holder[0]._is_connected = False

    engine = _RealtimeGenerationEngine(
        [_generation_output("partial", [7]), _disconnect, _generation_output("late", [8])]
    )
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)
    connection_holder.append(connection)

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    cleared = [c for c in observer.calls if c[0] == "unit_cleared"]
    assert [c[1]["outcome"] for c in cleared] == ["aborted"]
    finishes = [c for c in observer.calls if c[0] == "session_finished"]
    assert [f[1]["reason"] for f in finishes] == ["aborted"]


# @spec PORT-OBS-006
@pytest.mark.asyncio
async def test_exactly_one_terminal_section_even_when_terminal_send_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    (handle,) = observer.seed_ready(session_key, n=1)
    observer.mint(handle)

    engine = _RealtimeGenerationEngine([_generation_output("t", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    async def _failing_send_json(payload: dict[str, Any]) -> None:
        raise RuntimeError("socket already closed")

    connection.send_json = _failing_send_json

    await connection._run_generation(_empty_streaming_input(), asyncio.Queue(), request_id=session_key)

    finishes = [c for c in observer.calls if c[0] == "session_finished"]
    assert len(finishes) == 1


# ---------------------------------------------------------------------------
# Amendment 3: model-gated park-token resolution on the serving.
# ---------------------------------------------------------------------------


class _WidenedModelClsWithConfig:
    """Nemotron-shaped: widened signature, so token resolution applies."""

    async def buffer_realtime_audio(
        self,
        audio_stream: Any,
        input_stream: Any,
        model_config: Any,
        *,
        observer: Any = None,
        accepted_audio_budget_s: float | None = None,
        session_key: str | None = None,
    ) -> AsyncGenerator[Any, None]:
        return
        yield  # pragma: no cover


class _UnwidenedModelClsShape:
    async def buffer_realtime_audio(
        self, audio_stream: Any, input_stream: Any, model_config: Any
    ) -> AsyncGenerator[Any, None]:
        return
        yield  # pragma: no cover


def _serving_with_model_cls(model_cls: Any, hf_config: Any) -> Any:
    from vllm_omni.entrypoints.openai.serving_realtime import (
        NemotronServingRealtime,
    )

    serving: Any = NemotronServingRealtime.__new__(NemotronServingRealtime)
    serving.model_config = SimpleNamespace(hf_config=hf_config)
    serving.renderer = None
    serving.__dict__["model_cls"] = model_cls
    serving._observer = None
    serving._accepted_audio_budget_s = None
    return serving


# @spec PORT-OBS-003
def test_park_token_resolves_for_the_recognized_streaming_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Architecture-identity gate (review F5): the recognized Nemotron
    architecture resolves through the model package's own authority."""
    import vllm_omni.entrypoints.openai.serving_realtime as serving_mod

    monkeypatch.setattr(
        serving_mod, "_resolve_park_token_id", lambda hf_config: 777, raising=False
    )
    serving = _serving_with_model_cls(
        _WidenedModelClsWithConfig(),
        hf_config=SimpleNamespace(architectures=["Nemotron3_5AsrForRNNT"]),
    )
    assert serving.park_token_id == 777


# @spec PORT-OBS-003
def test_park_token_resolution_fails_loudly_when_authority_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recognized streaming model but no resolvable token -> loud
    failure, never a silent None (silent None was the A27 GPU-round
    defect class)."""
    import vllm_omni.entrypoints.openai.serving_realtime as serving_mod

    def _raise(hf_config: Any) -> int:
        raise ValueError("park token authority cannot resolve")

    monkeypatch.setattr(serving_mod, "_resolve_park_token_id", _raise, raising=False)
    serving = _serving_with_model_cls(
        _WidenedModelClsWithConfig(),
        hf_config=SimpleNamespace(architectures=["Nemotron3_5AsrForRNNT"]),
    )
    with pytest.raises(ValueError):
        _ = serving.park_token_id


# @spec PORT-OBS-003
def test_park_token_is_none_and_inert_for_other_realtime_models() -> None:
    serving = _serving_with_model_cls(_UnwidenedModelClsShape(), hf_config=SimpleNamespace())
    assert serving.park_token_id is None


# @spec PORT-OBS-003
def test_unrelated_widened_model_gets_no_park_token() -> None:
    """Review F5: a future realtime model may declare the same three
    generic observer params for call-shape compatibility WITHOUT
    inheriting Nemotron park semantics — identity, not signature,
    gates the resolver."""
    serving = _serving_with_model_cls(
        _WidenedModelClsWithConfig(),
        hf_config=SimpleNamespace(architectures=["SomeFutureRealtimeModel"]),
    )
    assert serving._model_declares_widened_buffer_kwargs is True
    assert serving.park_token_id is None


# @spec PORT-OBS-003
def test_widened_guard_requires_session_key_too() -> None:
    """Amendment 1: a model declaring only observer+budget (but not
    session_key) predates the correlation-identity contract and must be
    treated as un-widened."""

    class _PartiallyWidened:
        async def buffer_realtime_audio(
            self,
            audio_stream: Any,
            input_stream: Any,
            model_config: Any,
            *,
            observer: Any = None,
            accepted_audio_budget_s: float | None = None,
        ) -> AsyncGenerator[Any, None]:
            return
            yield  # pragma: no cover

    serving = _serving_with_model_cls(_PartiallyWidened(), hf_config=SimpleNamespace())
    assert serving._model_declares_widened_buffer_kwargs is False


# @spec PORT-OBS-006
@pytest.mark.asyncio
async def test_terminal_send_failure_after_finalization_still_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reviewer F3: transport delivery is not part of model completion.
    The engine generator exhausts successfully (model finalized), then
    the TranscriptionDone send fails — the session must finish
    completed, never error."""
    session_key = _fixed_session_key(monkeypatch)
    observer = _RecordingObserver()
    (handle,) = observer.seed_ready(session_key, n=1)
    observer.mint(handle)

    # Empty delta text: the only ``send`` is the post-exhaustion
    # TranscriptionDone, so the failure is strictly after finalization.
    engine = _RealtimeGenerationEngine([_generation_output("", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    async def _failing_send(event: Any) -> None:
        raise RuntimeError("socket write failed after generation finished")

    connection.send = _failing_send

    await connection._run_generation(
        _empty_streaming_input(), asyncio.Queue(), request_id=session_key
    )

    finishes = [c for c in observer.calls if c[0] == "session_finished"]
    assert [f[1]["reason"] for f in finishes] == ["completed"]
    assert [c for c in observer.calls if c[0] == "unit_cleared"] == []


# @spec PORT-OBS-003
@pytest.mark.asyncio
async def test_keyless_generation_with_observer_disables_observation_loudly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reviewer F6: with an observer installed, a generation without the
    threaded correlation id must not mint a second identity and observe
    under it — observation is disabled loudly; keyless compatibility
    remains only on the unobserved path."""
    observer = _RecordingObserver()
    observer.seed_ready("whatever", n=1)

    engine = _RealtimeGenerationEngine([_generation_output("t", [PARK_ID])])
    connection, _sent_events, _sent_json = _connection(engine, observer=observer)

    with caplog.at_level("WARNING"):
        await connection._run_generation(_empty_streaming_input(), asyncio.Queue())

    assert any("correlation" in rec.message for rec in caplog.records)
    assert [c for c in observer.calls if c[0] in ("unit_parked", "unit_cleared", "session_finished")] == []
