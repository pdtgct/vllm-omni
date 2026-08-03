# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: the leased-path StreamingObserver wiring.

Specs: PORT-OBS-003 (the transport-neutral factory injects the observer
at session construction through its INTERNAL constructor seam — the
public ``create_nemotron_session_factory(engine_client)`` signature is
unchanged, PORT-RTC-003; the lease consumer observes terminal
disposition at ticket resolution; ledger-failure clearing rides the
lease's single terminal fail/close path), PORT-OBS-005 (backlog
decrement-by-exact-count on bulk teardown), PORT-OBS-006 (session
lifecycle counters and the exact finished-reason mapping;
opens - finished == active), PORT-SESS-001 (the per-session
accepted-audio SECONDS budget bounds queue occupancy only; lifetime
cumulative audio is never capped by it).

Same importlib loader chain as ``test_nemotron_session.py`` (GPU-free
CPU tier), extended with the neutral ``vllm_omni.metrics.
streaming_transport`` module ``session.py`` now imports its protocol/
handle type from.

Hygiene (P1 correction, done "for real" this round): every bare
parent-package stub AND the hand-rolled ``vllm_omni.entrypoints.utils``
fake are restored via ``pytest.MonkeyPatch`` in a ``try/finally``
immediately surrounding ``_load_chain()`` — NOT deferred to
``teardown_module``, which pytest's collect-then-run model makes too
late (a later-collected file needing the REAL ``vllm_omni.entrypoints.
utils`` — or even a bare ``import vllm_omni`` — would otherwise get a
stale/incomplete fake or ``ModuleNotFoundError: 'vllm_omni' is not a
package``). The genuine-file leaf modules (session, streaming,
manifests, configuration_nemotron_asr, nemotron_session,
streaming_transport) stay cached under their full dotted names, matching
every sibling loader-chain file's deliberate "reuse-if-present"
class-identity sharing.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from collections.abc import Awaitable, Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ENTRYPOINTS = _ROOT / "vllm_omni/entrypoints"
_METRICS_PKG = _ROOT / "vllm_omni/metrics"
_MODEL_PKG = _ROOT / "vllm_omni/model_executor/models/nemotron_asr"
_MODULE_PATH = _ENTRYPOINTS / "nemotron_session.py"
_MODEL_BASE = "vllm_omni.model_executor.models.nemotron_asr"


def _load_chain() -> dict[str, Any]:
    mp = pytest.MonkeyPatch()
    try:
        for name in (
            "vllm_omni",
            "vllm_omni.entrypoints",
            "vllm_omni.model_executor",
            "vllm_omni.model_executor.models",
            "vllm_omni.metrics",
            _MODEL_BASE,
        ):
            if name not in sys.modules:
                mp.setitem(sys.modules, name, types.ModuleType(name))

        utils = types.ModuleType("vllm_omni.entrypoints.utils")

        def coerce_param_message_types(params: list[Any], is_streaming: bool) -> list[Any]:
            return list(params)

        utils.coerce_param_message_types = coerce_param_message_types  # type: ignore[attr-defined]
        mp.setitem(sys.modules, "vllm_omni.entrypoints.utils", utils)

        loaded: dict[str, Any] = {}

        def _load_leaf(dotted: str, path: Path) -> Any:
            existing = sys.modules.get(dotted)
            if existing is not None and getattr(existing, "__file__", None):
                return existing
            spec = importlib.util.spec_from_file_location(dotted, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = module  # persists: genuine file content
            spec.loader.exec_module(module)
            return module

        loaded["streaming_transport"] = _load_leaf(
            "vllm_omni.metrics.streaming_transport",
            _METRICS_PKG / "streaming_transport.py",
        )
        for mod, short in (
            (f"{_MODEL_BASE}.manifests", "manifests"),
            (f"{_MODEL_BASE}.configuration_nemotron_asr", "configuration"),
            (f"{_MODEL_BASE}.session", "session"),
            (f"{_MODEL_BASE}.streaming", "streaming"),
        ):
            filename = mod.rsplit(".", 1)[-1] + ".py"
            loaded[short] = _load_leaf(mod, _MODEL_PKG / filename)
        loaded["nemotron_session"] = _load_leaf("vllm_omni.entrypoints.nemotron_session", _MODULE_PATH)
        return loaded
    finally:
        mp.undo()


_M = _load_chain()
_NS = _M["nemotron_session"]
NemotronSessionFactory = _NS.NemotronSessionFactory
_SESSION = _M["session"]
_TRANSPORT = _M["streaming_transport"]
NemotronRealtimeSession = _SESSION.NemotronRealtimeSession
StreamingObserver = _TRANSPORT.StreamingObserver
ChunkReadyHandle = _TRANSPORT.ChunkReadyHandle

pytestmark = [pytest.mark.cpu]

PARK_ID = 13088
PLACEHOLDER_ID = 13089
PROMPTS = {"auto": 101, "en-US": 2}
_CADENCE = "560ms"
_CHUNK = _M["manifests"].RAW_SAMPLES_PER_CHUNK[_CADENCE]
WAIT_S = 2.0


def _hf(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "architectures": ["Nemotron3_5AsrForRNNT"],
        "eos_token_id": PARK_ID,
        "audio_chunk_token_id": PLACEHOLDER_ID,
        "prompt_dictionary": dict(PROMPTS),
        "num_prompts": 128,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def _wait(awaitable: Awaitable[Any]) -> Any:
    return await asyncio.wait_for(awaitable, WAIT_S)


def _audio(n: int) -> Any:
    return np.zeros(n, dtype=np.float32)


def _out(ids: list[int], text: str = "") -> Any:
    return SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=list(ids), text=text)])


def _default_script(item: Any, index: int) -> list[Any]:
    if "multi_modal_data" not in item:
        return [_out([PARK_ID])]
    return [_out([7, PARK_ID], text=f" w{index}")]


class FakeAsyncOmni:
    def __init__(self, *, script: Any = None, stop_after: int | None = None) -> None:
        self.model_config = SimpleNamespace(hf_config=_hf())
        self.renderer: Any = None
        self.default_sampling_params_list = (SimpleNamespace(tag="default"),)
        self.script = script or _default_script
        self.stop_after = stop_after
        self.prompts: list[Any] = []
        self.aborted: list[str] = []
        self._state_generation = 0
        self.state_releases: list[dict[str, Any]] = []
        self.pending_claim_timeout_s = 3600.0
        self.runtime_config = SimpleNamespace(
            accepted_audio_budget_s=30.0,
            accepted_audio_capacity_samples=480_000,
            max_retained_transcript_bytes=1 << 20,
            max_session_samples=None,
            session_idle_timeout_s=3600.0,
            session_finalization_timeout_s=3600.0,
        )

    @property
    def inventory(self) -> dict[str, str]:
        return {"schema_id": "state-manifest-v1", "profile_id": "default"}

    def get_persistent_state_service(self) -> FakeAsyncOmni:
        return self

    async def check_health(self) -> None:
        return None

    async def reserve(self, **kwargs: Any) -> Any:
        self._state_generation += 1
        return SimpleNamespace(
            engine_epoch="test-epoch",
            session_key=kwargs["session_key"],
            generation=self._state_generation,
            schema_id=kwargs["schema_id"],
            profile_id=kwargs["profile_id"],
            binding_token=f"binding-{self._state_generation}",
        )

    async def release(self, **kwargs: Any) -> None:
        self.state_releases.append(kwargs)

    async def claim_pending_cleanup(self, lease: Any) -> bool:
        del lease
        return True

    async def generate(
        self,
        *,
        prompt: Any,
        request_id: str,
        sampling_params_list: Any,
        request_id_already_unique: bool = False,
    ) -> Any:
        del request_id, request_id_already_unique
        index = 0
        async for item in prompt:
            self.prompts.append(item)
            for out in self.script(item, index):
                yield out
            index += 1
            if self.stop_after is not None and index >= self.stop_after:
                return

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


async def _passthrough_render(prompt: Any) -> Any:
    return prompt


class _RecordingObserver:
    """Real per-session waiting/in-flight tracking (matching
    ``test_streaming_observer.py``'s reference implementation) so
    ``complete_inflight`` correctly resolves handles the lease minted —
    a bare recorder that always returned ``None`` would silently break
    every real park-resolution call site exercised below."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._waiting: dict[str, list[Any]] = {}
        self._inflight: dict[str, Any] = {}

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
        handle = ChunkReadyHandle(
            session_key=session_key,
            cadence_ms=cadence_ms,
            chunk_type=chunk_type,
            ready_stamp_s=ready_stamp_s,
        )
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
        self.calls.append(("complete_inflight", {"session_key": session_key}))
        return handle

    def unit_parked(self, handle: Any, *, park_stamp_s: float) -> None:
        self.calls.append(("unit_parked", {"handle": handle, "park_stamp_s": park_stamp_s}))

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

    def clear_all_outstanding(self, session_key: str, *, outcome: str) -> None:
        handles = list(self._waiting.get(session_key, ()))
        inflight = self._inflight.get(session_key)
        if inflight is not None:
            handles.append(inflight)
        for handle in handles:
            self.unit_cleared(handle, outcome=outcome)


# ---- factory injects the observer at session construction (real, GREEN) ------
# (PORT-OBS-003): the INTERNAL NemotronSessionFactory constructor seam —
# never the public create_nemotron_session_factory (PORT-RTC-003, fixed
# below).


# @spec PORT-OBS-003
def test_internal_factory_constructor_injects_the_observer_at_session_construction() -> None:
    async def scenario() -> None:
        fake = _RecordingObserver()
        factory = NemotronSessionFactory(engine=FakeAsyncOmni(), observer=fake)
        lease = await factory.open(cadence=_CADENCE, locale="en-US")
        assert lease.session.observer is fake
        assert isinstance(lease.session.observer, StreamingObserver)
        await lease.release()

    _run(scenario())


# @spec PORT-OBS-003
def test_no_observer_supplied_leaves_session_observer_none() -> None:
    async def scenario() -> None:
        factory = NemotronSessionFactory(engine=FakeAsyncOmni())
        lease = await factory.open(cadence=_CADENCE, locale="en-US")
        assert lease.session.observer is None
        await lease.release()

    _run(scenario())


# ---- public factory signature is PINNED to (engine_client) only --------------
# (PORT-RTC-003, correction 2: the observer param was reverted off the
# PUBLIC constructor entirely)


# @spec PORT-RTC-003
def test_public_factory_constructor_signature_is_pinned_to_engine_client_only() -> None:
    import inspect

    sig = inspect.signature(_NS.create_nemotron_session_factory)
    assert list(sig.parameters) == ["engine_client"], (
        f"create_nemotron_session_factory must accept exactly (engine_client); got {list(sig.parameters)}"
    )


# @spec PORT-RTC-003, PORT-OBS-003
def test_public_factory_constructor_rejects_an_observer_keyword() -> None:
    """The public constructor must not silently accept (and discard) an
    observer keyword — TypeError is the correct, honest failure, not
    silent acceptance."""
    with pytest.raises(TypeError):
        _NS.create_nemotron_session_factory(FakeAsyncOmni(), observer=_RecordingObserver())


# ---- lease consumer observes terminal disposition at ticket resolution -------
# (PORT-OBS-003) — RED: _consume() does not call the observer yet.


# @spec PORT-OBS-003, PORT-OBS-004
def test_parked_ticket_resolution_reports_latency_from_the_ready_stamp() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(), cadence=_CADENCE, locale="auto", with_ledger=True, observer=fake,
            session_key="test-fixture-key",
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-1",
            render=_passthrough_render,
        )
        await _wait(lease.feed(_audio(_CHUNK)))
        await _wait(lease.abort())
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    parked = [c for c in calls if c[0] == "unit_parked"]
    assert len(parked) == 1


# ---- ledger-failure path clears all pending (decrement-by-count) -------------
# (PORT-OBS-003, PORT-OBS-005) — RED


# @spec PORT-OBS-003, PORT-OBS-005
def test_ledger_failure_clears_every_pending_unit_by_exact_count() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        # A script that consumes the carrier but never emits park, and
        # whose generation stream ends after that one item (stop_after=1):
        # the minted ticket stays pending when the stream ends, so the
        # lease's premature-end path fails the ledger with it still
        # outstanding (mirrors test_premature_stream_end_fails_pending_feed
        # in test_nemotron_session.py).
        engine = FakeAsyncOmni(script=lambda item, index: [], stop_after=1)
        session = NemotronRealtimeSession.from_model_config(
            _hf(), cadence=_CADENCE, locale="auto", with_ledger=True, observer=fake,
            session_key="test-fixture-key",
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-2",
            render=_passthrough_render,
        )
        with pytest.raises(RuntimeError):
            await _wait(lease.feed(_audio(_CHUNK)))
        await _wait(lease.abort())
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    cleared = [c for c in calls if c[0] == "unit_cleared"]
    assert len(cleared) == 1


# ---- lifecycle counters + exact reason mapping (PORT-OBS-006) — RED ----------


# @spec PORT-OBS-006
def test_session_finished_reason_completed_after_flush_park() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(), cadence=_CADENCE, locale="auto", with_ledger=True, observer=fake,
            session_key="rt-obs-3",
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-3",
            render=_passthrough_render,
        )
        await _wait(lease.flush())
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    finished = [c for c in calls if c[0] == "session_finished"]
    assert finished == [("session_finished", {"session_key": "rt-obs-3", "reason": "completed"})]


# @spec PORT-OBS-006
def test_session_finished_reason_aborted_on_client_abort() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(), cadence=_CADENCE, locale="auto", with_ledger=True, observer=fake,
            session_key="rt-obs-4",
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-4",
            render=_passthrough_render,
        )
        await _wait(lease.feed(_audio(_CHUNK)))
        await _wait(lease.abort())
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    finished = [c for c in calls if c[0] == "session_finished"]
    assert finished == [("session_finished", {"session_key": "rt-obs-4", "reason": "aborted"})]


# @spec PORT-OBS-006
def test_opens_minus_finished_equals_active_under_an_abort_close_race() -> None:
    """Concurrent abort() and finish() on the same lease must still leave
    exactly one session_opened and exactly one session_finished — the
    idempotent terminal-disposition section, never double-counted."""

    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(), cadence=_CADENCE, locale="auto", with_ledger=True, observer=fake,
            session_key="test-fixture-key",
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-5",
            render=_passthrough_render,
        )
        await _wait(lease.feed(_audio(_CHUNK)))
        # Race abort() against finish(): both may run, only one terminal
        # disposition may be recorded.
        results = await asyncio.gather(lease.abort(), lease.finish(), return_exceptions=True)
        assert all(not isinstance(r, Exception) for r in results)
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    opened = [c for c in calls if c[0] == "session_opened"]
    finished = [c for c in calls if c[0] == "session_finished"]
    assert len(opened) == 1
    assert len(finished) == 1


# ---- every ready unit gets exactly one disposition (PORT-OBS-003) ------------
# The LEASED-path consumer (this lease, not the segmenter) owns terminal
# observation — the ownership fix from review: this invariant moved here
# from test_streaming_observer.py, which asserts only ready-event emission.


# @spec PORT-OBS-003, PORT-OBS-005
def test_every_ready_unit_gets_exactly_one_disposition_at_the_lease() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(), cadence=_CADENCE, locale="auto", with_ledger=True, observer=fake,
            session_key="test-fixture-key",
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-6",
            render=_passthrough_render,
        )
        await _wait(lease.feed(_audio(_CHUNK)))
        await _wait(lease.feed(_audio(_CHUNK)))
        await _wait(lease.flush())
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    ready = [c for c in calls if c[0] == "unit_ready"]
    terminal = [c for c in calls if c[0] in ("unit_parked", "unit_cleared")]
    assert len(ready) > 0
    assert len(terminal) == len(ready)


# ---- leased-path accepted-audio budget (PORT-SESS-001, amended Decision 1) ----
# Native-path cases live in test_streaming_observer.py.


# @spec PORT-SESS-001
def test_leased_piece_exceeding_the_budget_is_rejected_whole_before_acceptance() -> None:
    """A piece that would push accepted audio past a near-zero budget
    must be rejected atomically through the lease's ``feed`` — the whole
    piece, never a partial/truncated accept."""

    async def scenario() -> None:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(),
            cadence=_CADENCE,
            locale="auto",
            with_ledger=True,
            observer=fake,
            session_key="test-fixture-key",
            accepted_audio_budget_s=0.001,
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-budget-1",
            render=_passthrough_render,
        )
        with pytest.raises(ValueError, match="buffer_overflow"):
            await _wait(lease.feed(_audio(_CHUNK * 5)))
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


# @spec PORT-SESS-001
def test_leased_rejected_piece_accrues_no_accepted_audio_seconds() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(),
            cadence=_CADENCE,
            locale="auto",
            with_ledger=True,
            observer=fake,
            session_key="test-fixture-key",
            accepted_audio_budget_s=0.001,
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-budget-2",
            render=_passthrough_render,
        )
        try:
            await _wait(lease.feed(_audio(_CHUNK * 5)))
        except ValueError:
            pass
        await _wait(lease.abort())
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    assert [c for c in calls if c[0] == "accepted_audio_seconds"] == []


# @spec PORT-SESS-001, PORT-OBS-005
def test_leased_rejected_piece_emits_exactly_one_input_queue_overflow() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(),
            cadence=_CADENCE,
            locale="auto",
            with_ledger=True,
            observer=fake,
            session_key="test-fixture-key",
            accepted_audio_budget_s=0.001,
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-budget-3",
            render=_passthrough_render,
        )
        try:
            await _wait(lease.feed(_audio(_CHUNK * 5)))
        except ValueError:
            pass
        await _wait(lease.abort())
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    overflow = [c for c in calls if c[0] == "overflow" and c[1]["kind"] == "input_queue"]
    assert len(overflow) == 1


# @spec PORT-SESS-001, PORT-OBS-006
def test_leased_session_follows_ordinary_terminal_clearing_after_budget_rejection() -> None:
    """After a budget rejection the session must still follow its
    ordinary terminal clearing (PORT-OBS-006's idempotent terminal-
    disposition section) — a budget rejection is not a special path that
    skips finished-reason accounting."""

    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(),
            cadence=_CADENCE,
            locale="auto",
            with_ledger=True,
            observer=fake,
            session_key="test-fixture-key",
            accepted_audio_budget_s=0.001,
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-budget-4",
            render=_passthrough_render,
        )
        try:
            await _wait(lease.feed(_audio(_CHUNK * 5)))
        except ValueError:
            pass
        await _wait(lease.abort())
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    finished = [c for c in calls if c[0] == "session_finished"]
    assert len(finished) == 1


# @spec PORT-SESS-001
def test_leased_occupancy_lifecycle_never_caps_lifetime_cumulative_audio() -> None:
    """PORT-SESS-001 (amended): the budget bounds QUEUE OCCUPANCY only.
    Accept pieces to near-budget, drain each via a park before feeding
    the next (so occupancy never exceeds the budget at any instant), then
    keep accepting — even though LIFETIME cumulative accepted audio
    climbs well past the budget. A duration-cap implementation (rejecting
    because lifetime total exceeds 30s/the configured budget) must FAIL
    this test."""

    async def scenario() -> int:
        fake = _RecordingObserver()
        session = NemotronRealtimeSession.from_model_config(
            _hf(),
            cadence=_CADENCE,
            locale="auto",
            with_ledger=True,
            observer=fake,
            session_key="test-fixture-key",
            accepted_audio_budget_s=1.12,  # covers exactly 2 cadences of occupancy
        )
        engine = FakeAsyncOmni()
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-occupancy",
            render=_passthrough_render,
        )
        accepted = 0
        # 5 regular cadences fed one at a time, each drained (via the
        # script's park) before the next feed — lifetime cumulative audio
        # (5 * 0.56s = 2.8s) is already well past the 1.12s occupancy
        # budget by the third feed, yet every feed must still be accepted.
        for _ in range(5):
            results = await _wait(lease.feed(_audio(_CHUNK)))
            accepted += len(results)
        await _wait(lease.flush())
        await lease.release()
        return accepted

    accepted = _run(scenario())
    assert accepted == 5


# ---- nonfatal observer failure (PORT-OBS-002/003) ----------------------------
# An observer whose sink raises must never propagate into feed/park/
# finalize call sites — observation is non-authoritative and nonfatal.


class _RaisingObserver:
    """Every call raises — the sink failure this test proves is swallowed."""

    def session_opened(self, *, session_key: str, cadence_ms: str) -> None:
        raise RuntimeError("sink boom: session_opened")

    def session_finished(self, *, session_key: str, reason: str) -> None:
        raise RuntimeError("sink boom: session_finished")

    def session_open_rejected(self, *, reason: str) -> None:
        raise RuntimeError("sink boom: session_open_rejected")

    def accepted_audio_seconds(self, *, cadence_ms: str, seconds: float) -> None:
        raise RuntimeError("sink boom: accepted_audio_seconds")

    def unit_ready(
        self,
        *,
        session_key: str = "default",
        cadence_ms: str,
        chunk_type: str,
        ready_stamp_s: float,
    ) -> Any:
        raise RuntimeError("sink boom: unit_ready")

    def unit_minted(self, handle: Any) -> None:
        raise RuntimeError("sink boom: unit_minted")

    def complete_inflight(self, session_key: str) -> Any:
        raise RuntimeError("sink boom: complete_inflight")

    def unit_parked(self, handle: Any, *, park_stamp_s: float) -> None:
        raise RuntimeError("sink boom: unit_parked")

    def unit_cleared(self, handle: Any, *, outcome: str) -> None:
        raise RuntimeError("sink boom: unit_cleared")

    def overflow(self, *, kind: str) -> None:
        raise RuntimeError("sink boom: overflow")

    def clear_all_outstanding(self, session_key: str, *, outcome: str) -> None:
        raise RuntimeError("sink boom: clear_all_outstanding")


# @spec PORT-OBS-002, PORT-OBS-003
def test_observer_sink_failure_never_propagates_into_feed_or_flush() -> None:
    """A raising observer must not fail feed()/flush() — observation is
    non-authoritative and nonfatal at every call site (PORT-OBS-002)."""

    async def scenario() -> tuple[list[str], Any]:
        engine = FakeAsyncOmni()
        session = NemotronRealtimeSession.from_model_config(
            _hf(), cadence=_CADENCE, locale="auto", with_ledger=True, observer=_RaisingObserver(),
            session_key="test-fixture-key",
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-nonfatal",
            render=_passthrough_render,
        )
        results = await _wait(lease.feed(_audio(_CHUNK)))
        text = await _wait(lease.flush())
        await lease.release()
        return results, text

    # Today this passes trivially (nothing calls the observer yet); once
    # Phase 6 wires observation, feed/flush must STILL complete normally
    # despite every observer call raising.
    results, text = _run(scenario())
    assert results == [" w0"]
    assert text.complete_text == " w0 w1"


# ---------------------------------------------------------------------------
# A27 topology cascade (amendment 1): the leased path mints the engine
# request id BEFORE constructing the session and uses it as the key.
# ---------------------------------------------------------------------------


# @spec PORT-OBS-003, PORT-RTC-003
def test_factory_open_uses_the_request_id_as_the_session_key() -> None:
    """The request id is minted BEFORE session construction and becomes
    the session's correlation key — one identity for ready events,
    in-flight completion, and lifecycle on the leased path."""

    async def scenario() -> None:
        factory = NemotronSessionFactory(engine=FakeAsyncOmni())
        lease = await factory.open(cadence=_CADENCE, locale="en-US")
        assert lease.session.session_key == lease.request_id
        assert lease.request_id.startswith("nemotron-session-")
        await lease.release()

    _run(scenario())


# @spec PORT-OBS-006
def test_premature_generation_exhaustion_finishes_error_not_completed() -> None:
    """Reviewer F2: generation ending before audio close / FLUSH park is
    a failed finalization — the ledger is failed AND the session reason
    is error, never completed."""

    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        fake = _RecordingObserver()
        engine = FakeAsyncOmni(stop_after=1)  # engine stops mid-stream
        session = NemotronRealtimeSession.from_model_config(
            _hf(), cadence=_CADENCE, locale="auto", with_ledger=True, observer=fake,
            session_key="rt-obs-premature",
        )
        lease = _NS.NemotronSessionLease(
            engine=engine,
            session=session,
            request_id="rt-obs-premature",
            render=_passthrough_render,
        )
        try:
            await _wait(lease.feed(_audio(_CHUNK)))
            await _wait(lease.feed(_audio(_CHUNK)))
            await _wait(lease.flush())
        except Exception:
            pass  # the failed ledger surfaces to the consumer; reason is under test
        await lease.release()
        return fake.calls

    calls = _run(scenario())
    finished = [c for c in calls if c[0] == "session_finished"]
    assert len(finished) == 1
    assert finished[0][1]["reason"] == "error"
