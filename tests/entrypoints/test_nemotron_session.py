# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The transport-neutral Nemotron session module.

GPU-free CPU tier. Drives ``NemotronSessionFactory`` and
``NemotronSessionLease`` over a FAKE ``AsyncOmni`` (scripted
``generate`` output stream, recorded ``abort``) while the segmenter,
session, and receipt ledger are the REAL modules loaded through the
importlib chain. Pins:

- ``feed`` returns one cumulative hypothesis per completed cadence via
  REAL ledger tickets (real ``buffer_stream`` + real
  ``NemotronRealtimeSession(with_ledger=True)``) — no cadence
  arithmetic in the binding (ING-FE-006, the F6/R2 ledger adoption);
- engine/generation failure -> ``ledger.fail`` -> a blocked ``feed``
  raises the ORIGINAL error, never hangs (round-4 R2 / round-5 V3);
- ``flush`` drains to stream end and returns the final transcript,
  final tail included (PORT-SESS-003); zero-feed flush still runs the
  explicit zero-sample final-tail transaction;
- the prototype's liveness guards survive the ledger port: feed after
  flush/abort/generation-end raises, never hangs (ING-LIFE-010);
- ``feed`` notifies exact piece acceptance before carrier completion;
- the factory accepts explicit cadence/locale model controls and
  installs no serving counter or implicit single-shot geometry;
- ``update_locale`` delegates to ``session.select_prompt`` (the one
  validator, PORT-LID-001) and the next mint stamps the new index;
- structural protocol conformance: the concrete factory/lease satisfy
  the ``SessionFactory``/``SessionLease`` runtime protocols WITHOUT
  importing endpoint-specific protocols (PORT-RTC-007);
- source-scan: no cadence/chunk arithmetic in the binding.

Loader-runnable: every module is loaded by file path under its
canonical dotted name (parents stubbed), so no ``vllm_omni`` package
``__init__`` (which imports ``vllm``) is touched; the binding's vllm
imports are lazy by design.
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
from uuid import uuid4

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ENTRYPOINTS = _ROOT / "vllm_omni/entrypoints"
_MODEL_PKG = _ROOT / "vllm_omni/model_executor/models/nemotron_asr"
_MODULE_PATH = _ENTRYPOINTS / "nemotron_session.py"
_MODEL_BASE = "vllm_omni.model_executor.models.nemotron_asr"

#: Calls the chain-level fake ``coerce_param_message_types`` records,
#: as ``(params, is_streaming)`` — the binding must coerce the engine
#: defaults for streaming exactly as the realtime connection does.
_COERCE_CALLS: list[tuple[list[Any], bool]] = []


def _load_chain() -> dict[str, Any]:
    """Load the real model + entrypoints modules engine-free.

    Same importlib chain as test_realtime_session.py, extended with the
    entrypoints modules under test. ``vllm_omni.entrypoints.utils`` is a
    FAKE (its real body imports vllm): a recording identity coercion, so
    the binding's lazy sampling-params import resolves and is assertable.
    """
    for name in (
        "vllm_omni",
        "vllm_omni.entrypoints",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        _MODEL_BASE,
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    utils = types.ModuleType("vllm_omni.entrypoints.utils")

    def coerce_param_message_types(params: list[Any], is_streaming: bool) -> list[Any]:
        _COERCE_CALLS.append((list(params), is_streaming))
        return list(params)

    utils.coerce_param_message_types = coerce_param_message_types  # type: ignore[attr-defined]
    sys.modules["vllm_omni.entrypoints.utils"] = utils

    loaded: dict[str, Any] = {}
    modules = [
        (f"{_MODEL_BASE}.manifests", _MODEL_PKG / "manifests.py", "manifests"),
        (
            f"{_MODEL_BASE}.configuration_nemotron_asr",
            _MODEL_PKG / "configuration_nemotron_asr.py",
            "configuration",
        ),
        (f"{_MODEL_BASE}.session", _MODEL_PKG / "session.py", "session"),
        (f"{_MODEL_BASE}.streaming", _MODEL_PKG / "streaming.py", "streaming"),
        (
            "vllm_omni.entrypoints.nemotron_session",
            _MODULE_PATH,
            "nemotron_session",
        ),
    ]
    for dotted, path, short in modules:
        # Reuse-if-present: multiple test files chain-load these same
        # canonical names at COLLECTION time, and the module under test
        # resolves its lazy imports through sys.modules at CALL time —
        # an unconditional overwrite here would split class identity
        # between chains and break every isinstance/behavioral check in
        # whichever file collected first.
        existing = sys.modules.get(dotted)
        if existing is not None and getattr(existing, "__file__", None):
            loaded[short] = existing
            continue
        spec = importlib.util.spec_from_file_location(dotted, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = module
        spec.loader.exec_module(module)
        loaded[short] = module
    return loaded


_M = _load_chain()
_NS = _M["nemotron_session"]
NemotronSessionFactory = _NS.NemotronSessionFactory
NemotronSessionLease = _NS.NemotronSessionLease
SessionFactory = _NS.SessionFactory
SessionLease = _NS.SessionLease
_SESSION = _M["session"]
NemotronRealtimeSession = _SESSION.NemotronRealtimeSession
DEFAULT_CADENCE = _SESSION.DEFAULT_CADENCE
RAW_SAMPLES_PER_CHUNK = _M["manifests"].RAW_SAMPLES_PER_CHUNK

pytestmark = [pytest.mark.cpu]

PARK_ID = 13088
PLACEHOLDER_ID = 13089
EOU_ID = 13090
FLUSH_ID = 13091
PROMPTS = {"auto": 101, "en-US": 2, "de-DE": 7}
#: The admitted cadence the lease tests run at; sample counts come from
#: the published manifests, never re-derived (arithmetic lives in TESTS
#: only as fixture sizing).
_CADENCE = "560ms"
_CHUNK = RAW_SAMPLES_PER_CHUNK[_CADENCE]
WAIT_S = 2.0
_LOCALE = "auto"


def _hf(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "num_asr_labels": 13087,
        "vocab_size": 13092,
        "eos_token_id": PARK_ID,
        "audio_chunk_token_id": PLACEHOLDER_ID,
        "eou_token_id": EOU_ID,
        "flush_token_id": FLUSH_ID,
        "endpoint_history_capacity_frames": 12,
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
    """Bound every await so a regression HANGS the test never."""
    return await asyncio.wait_for(awaitable, WAIT_S)


def _audio(n: int) -> Any:
    return np.zeros(n, dtype=np.float32)


# ---- fakes ---------------------------------------------------------------------


def _out(
    ids: list[int],
    text: str = "",
    *,
    segment_completion: Any | None = None,
) -> Any:
    return SimpleNamespace(
        stage_id=0,
        outputs=[SimpleNamespace(token_ids=list(ids), text=text)],
        segment_completion=segment_completion,
    )


def _default_script(item: Any, index: int) -> list[Any]:
    if "multi_modal_data" not in item:
        return [_out([PARK_ID])]
    return [_out([7, PARK_ID], text=f" w{index}")]


class FakeAsyncOmni:
    """AsyncOmni stand-in: scripted generate outputs, recorded abort.

    ``generate`` mirrors the real keyword surface (``prompt`` async
    generator, ``request_id``, ``sampling_params_list``,
    ``request_id_already_unique``); the script maps each consumed
    rendered prompt to output batches. Like the real ``AsyncOmni``,
    a call WITHOUT ``request_id_already_unique=True`` suffixes the
    request id — the recorded ``request_ids`` therefore reflect the
    identity the engine actually tracks, which the scheduler's
    persistent-state claim requires to equal the lease's
    request/session key (PORT-STATE-019).
    """

    def __init__(self, *, script: Any = None, stop_after: int | None = None) -> None:
        self.model_config = SimpleNamespace(hf_config=_hf())
        self.renderer: Any = None
        self.default_sampling_params_list = (SimpleNamespace(tag="default"),)
        self.script = script or _default_script
        self.stop_after = stop_after
        self.prompts: list[Any] = []
        self.aborted: list[str] = []
        self.request_ids: list[str] = []
        self.sampling: Any = None
        self._state_generation = 0
        self.state_releases: list[dict[str, Any]] = []
        self.pending_cleanup_calls: list[Any] = []
        self.pending_cleanup_wins = True
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
        self.pending_cleanup_calls.append(lease)
        return self.pending_cleanup_wins

    def _track_generate(
        self,
        request_id: str,
        request_id_already_unique: bool,
    ) -> None:
        """Mirror ``AsyncOmni.generate``'s id handling exactly."""
        if not request_id_already_unique:
            request_id = f"{request_id}-{uuid4().hex[:8]}"
        self.request_ids.append(request_id)

    async def generate(
        self,
        *,
        prompt: Any,
        request_id: str,
        sampling_params_list: Any,
        request_id_already_unique: bool = False,
    ) -> Any:
        self._track_generate(request_id, request_id_already_unique)
        self.sampling = sampling_params_list
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
    """Test render: the segmenter's prompt goes to the engine as-is."""
    return prompt


def _make_lease(
    engine: FakeAsyncOmni,
    *,
    cadence: str = _CADENCE,
    locale: str = _LOCALE,
) -> tuple[Any, Any]:
    session = NemotronRealtimeSession.from_model_config(_hf(), cadence=cadence, locale=locale, with_ledger=True)
    lease = NemotronSessionLease(
        engine=engine,
        session=session,
        request_id="rt-test-1",
        render=_passthrough_render,
    )
    return lease, None


# ---- factory: caller-selected model controls, no parallel gate ----------------


# @spec PORT-RTC-001, PORT-RTC-003, PORT-LID-001, PORT-SESS-002
def test_open_realtime_validates_caller_cadence_and_locale() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        factory = NemotronSessionFactory(engine=engine)
        lease = await factory.open(cadence=_CADENCE, locale="en-US")

        assert lease.session.geometry.cadence == _CADENCE
        assert lease.session.prompt_index == PROMPTS["en-US"]
        assert lease.session.ledger is not None
        await lease.release()

    _run(scenario())


# @spec PORT-RTC-001, PORT-RTC-003
def test_factory_rejects_unknown_controls_without_minting_a_lease() -> None:
    engine = FakeAsyncOmni()
    factory = NemotronSessionFactory(engine=engine)
    with pytest.raises(ValueError, match="locale"):
        _run(factory.open(cadence=_CADENCE, locale="not-a-locale"))
    with pytest.raises(ValueError, match="cadence"):
        _run(factory.open(cadence="not-a-cadence", locale=_LOCALE))


def test_factory_leases_get_distinct_request_ids() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        factory = NemotronSessionFactory(engine=engine)
        a = await factory.open(cadence=_CADENCE, locale=_LOCALE)
        b = await factory.open(cadence=_CADENCE, locale=_LOCALE)
        assert a.request_id != b.request_id
        await a.release()
        await b.release()

    _run(scenario())


# ---- feed: real ledger tickets over the real segmenter (F6/R2) -----------------


# @spec PORT-RTC-002, ING-FE-006
def test_feed_returns_one_cumulative_hypothesis_per_completed_cadence() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        assert await _wait(lease.feed(_audio(_CHUNK))) == [" w0"]
        assert await _wait(lease.feed(_audio(_CHUNK))) == [" w0 w1"]
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


# @spec PORT-RTC-002, ING-FE-006
def test_feed_sub_cadence_piece_returns_no_hypothesis() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        assert await _wait(lease.feed(_audio(100))) == []
        # The residual completes with the remainder of a full cadence.
        assert await _wait(lease.feed(_audio(_CHUNK - 100))) == [" w0"]
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


# @spec PORT-RTC-002, ING-FE-006
def test_feed_burst_returns_hypotheses_in_cadence_order() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        results = await _wait(lease.feed(_audio(2 * _CHUNK)))
        assert results == [" w0", " w0 w1"]
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


# @spec PORT-SEG-006, PORT-SEG-007, PORT-RTC-002
def test_feed_projects_committed_segments_before_returning_park_hypothesis() -> None:
    async def scenario() -> None:
        completion = SimpleNamespace(
            generation=1,
            text=" hello",
            reason="model",
        )

        def segmenting(item: Any, index: int) -> list[Any]:
            if "multi_modal_data" not in item:
                return [_out([PARK_ID])]
            return [
                _out(
                    [7, EOU_ID, PARK_ID],
                    text=" hello",
                    segment_completion=completion,
                )
            ]

        lease, _ = _make_lease(FakeAsyncOmni(script=segmenting))
        observed: list[Any] = []
        hypotheses = await _wait(
            lease.feed(_audio(_CHUNK), on_segment=observed.append)
        )

        assert hypotheses == [" hello"]
        assert observed == [completion]
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


# @spec PORT-RTC-004, ING-FE-005
def test_feed_notifies_piece_acceptance_before_carrier_completion() -> None:
    """The callback releases transport credit at receipt, not at park."""

    async def scenario() -> None:
        allow_park = asyncio.Event()
        accepted = asyncio.Event()
        accepted_samples: list[int] = []

        class PausedEngine(FakeAsyncOmni):
            async def generate(
                self,
                *,
                prompt: Any,
                request_id: str,
                sampling_params_list: Any,
                request_id_already_unique: bool = False,
            ) -> Any:
                self._track_generate(request_id, request_id_already_unique)
                self.sampling = sampling_params_list
                async for item in prompt:
                    self.prompts.append(item)
                    await allow_park.wait()
                    yield _out([7, PARK_ID], text=" accepted")

        def on_accepted(sample_count: int) -> None:
            accepted_samples.append(sample_count)
            accepted.set()

        lease, _ = _make_lease(PausedEngine())
        feed = asyncio.create_task(lease.feed(_audio(_CHUNK), on_accepted=on_accepted))

        await _wait(accepted.wait())
        assert accepted_samples == [_CHUNK]
        assert not feed.done(), "carrier completion must still be waiting on park"

        allow_park.set()
        assert await _wait(feed) == [" accepted"]
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


def test_feed_coerces_engine_default_sampling_for_streaming() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        _COERCE_CALLS.clear()
        await _wait(lease.feed(_audio(_CHUNK)))
        assert len(_COERCE_CALLS) == 1
        params, is_streaming = _COERCE_CALLS[0]
        assert is_streaming is True
        assert params == list(engine.default_sampling_params_list)
        assert engine.sampling == params
        assert engine.request_ids == ["rt-test-1"]
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


# @spec PORT-RTC-003
def test_factory_opens_selected_cadence_without_a_parallel_limiter() -> None:
    async def scenario() -> None:
        factory = NemotronSessionFactory(engine=FakeAsyncOmni())
        assert not hasattr(factory, "open_ephemeral")

        lease = await factory.open(cadence=_CADENCE, locale="en-US")
        assert lease.session.geometry.cadence == _CADENCE
        assert lease.session.prompt_index == PROMPTS["en-US"]
        await lease.release()
        await lease.release()

    _run(scenario())


# @spec PORT-STATE-019, PORT-RTC-003
def test_factory_lease_submits_engine_request_under_the_lease_session_key() -> None:
    """The engine-tracked request id must equal the reserved session_key.

    ``AsyncOmni.generate`` suffixes the request id unless the caller
    declares it already unique. The persistent-state binding travels
    under the lease's session_key, and the scheduler keys the claimed
    binding by the ENGINE request id — a suffixed submission therefore
    fails ``prepare_attn``'s identity check ("persistent-state binding
    request mismatch") on the bounded-stream lane. The fake engine
    mirrors the real suffixing contract, so this test fails if the
    lease ever lets the default rewrite happen.
    """

    async def scenario() -> None:
        engine = FakeAsyncOmni(
            script=lambda item, index: [_out([7, PARK_ID], text=" ok")]
        )
        factory = NemotronSessionFactory(engine=engine)
        lease = await factory.open(cadence=_CADENCE, locale="en-US")

        async def render(prompt: Any) -> Any:
            return SimpleNamespace(prompt=prompt)

        lease._render = render
        assert lease._state_lease.session_key == lease.request_id

        await lease.feed(_audio(_CHUNK))

        assert engine.request_ids == [lease.request_id]
        await lease.abort()
        await lease.release()

    _run(scenario())


# @spec PORT-STATE-014, PORT-RTC-003, PORT-RTC-005
def test_factory_pending_claim_timeout_releases_through_lease_owner() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        engine.pending_claim_timeout_s = 0.01
        factory = NemotronSessionFactory(engine=engine)
        lease = await factory.open(cadence=_CADENCE, locale="en-US")

        await asyncio.sleep(0.03)

        assert len(engine.pending_cleanup_calls) == 1
        assert len(engine.state_releases) == 1
        assert engine.state_releases[0]["reason"] == "pending_claim_timeout"
        await lease.release()
        assert len(engine.state_releases) == 1

    _run(scenario())


# @spec PORT-SESS-005, PORT-STATE-014
def test_factory_idle_timeout_aborts_and_releases_once() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        engine.runtime_config.session_idle_timeout_s = 0.01
        factory = NemotronSessionFactory(engine=engine)
        lease = await factory.open(cadence=_CADENCE, locale="en-US")

        await asyncio.sleep(0.03)

        assert len(engine.state_releases) == 1
        assert engine.state_releases[0]["reason"] == "idle_timeout"
        with pytest.raises(TimeoutError, match="idle_timeout"):
            await lease.feed(_audio(_CHUNK))
        assert engine.request_ids == []
        await lease.release()
        assert len(engine.state_releases) == 1

    _run(scenario())


# @spec PORT-SESS-005
def test_rearming_idle_timeout_fences_the_superseded_timer() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        factory = NemotronSessionFactory(engine=engine)
        lease = await factory.open(cadence=_CADENCE, locale="en-US")
        stale_generation = lease._session_lifecycle.generation

        lease._arm_session_lifecycle_timeout("idle")
        lease._session_lifecycle._deadline_reached(
            stale_generation,
            "idle",
        )
        await asyncio.sleep(0)

        assert lease._session_lifecycle.expired is False
        assert engine.state_releases == []
        await lease.release()

    _run(scenario())


# @spec PORT-SESS-005
def test_factory_feed_rearms_deadline_without_replacing_asyncio_tasks() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        factory = NemotronSessionFactory(engine=engine)
        lease = await factory.open(cadence=_CADENCE, locale="en-US")

        class RenderedPrompt:
            def __init__(self, prompt: Any) -> None:
                self.prompt = prompt

            def __contains__(self, key: str) -> bool:
                return key in self.prompt

        async def render(prompt: Any) -> Any:
            return RenderedPrompt(prompt)

        lease._render = render

        try:
            await _wait(lease.feed(_audio(_CHUNK)))
            tasks_after_first_accept = asyncio.all_tasks()
            for _ in range(3):
                await _wait(lease.feed(_audio(_CHUNK)))

            assert asyncio.all_tasks() == tasks_after_first_accept
        finally:
            await lease.abort()
            await lease.release()

    _run(scenario())


# @spec PORT-INT-005
def test_factory_rejects_state_service_without_runtime_envelope() -> None:
    engine = FakeAsyncOmni()
    engine.runtime_config = None

    with pytest.raises(ValueError, match="resolved streaming runtime envelope"):
        NemotronSessionFactory(engine=engine)


# @spec PORT-SESS-005, PORT-STATE-014
def test_factory_finalization_timeout_aborts_without_terminal_result() -> None:
    async def scenario() -> None:
        class HeldGenerationEngine(FakeAsyncOmni):
            def __init__(self) -> None:
                super().__init__()
                self.generation_started = asyncio.Event()
                self.hold_generation = asyncio.Event()

            async def generate(
                self,
                *,
                prompt: Any,
                request_id: str,
                sampling_params_list: Any,
                request_id_already_unique: bool = False,
            ) -> Any:
                self._track_generate(request_id, request_id_already_unique)
                self.sampling = sampling_params_list
                async for item in prompt:
                    self.prompts.append(item)
                    self.generation_started.set()
                    await self.hold_generation.wait()
                if False:  # pragma: no cover - marks this as async generator
                    yield None

        engine = HeldGenerationEngine()
        engine.runtime_config.session_finalization_timeout_s = 3600.0
        factory = NemotronSessionFactory(engine=engine)
        lease = await factory.open(cadence=_CADENCE, locale="en-US")

        async def render(prompt: Any) -> Any:
            return SimpleNamespace(prompt=prompt)

        lease._render = render

        flush = asyncio.create_task(lease.flush())
        await asyncio.wait_for(engine.generation_started.wait(), timeout=1.0)
        lease._session_lifecycle._finalization_timeout_s = 0.01
        lease._arm_session_lifecycle_timeout("finalization")
        with pytest.raises(TimeoutError, match="finalization_timeout"):
            await flush

        assert engine.aborted == [lease.request_id]
        assert len(engine.state_releases) == 1
        assert engine.state_releases[0]["reason"] == "finalization_timeout"
        assert lease._terminal_result is None

    _run(scenario())


# ---- flush: drain to stream end (PORT-SESS-003) --------------------------------


# @spec PORT-DEC-009, PORT-SESS-003, PORT-RTC-002, PORT-RTC-005
def test_flush_drains_final_tail_and_returns_final_transcript() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        assert await _wait(lease.feed(_audio(_CHUNK))) == [" w0"]
        terminal = await _wait(lease.flush())
        assert terminal.complete_text == " w0 w1"
        assert terminal.completion.reason == "terminal"
        assert terminal.completion.text == " w0 w1"
        # FLUSH follows the cadence and final tail, but mints no ticket
        # and contributes no transcript text.
        assert len(engine.prompts) == 3
        assert "multi_modal_data" not in engine.prompts[-1]
        await _wait(lease.finish())  # no-op after flush
        assert engine.aborted == []
        await lease.release()

    _run(scenario())


# @spec PORT-DEC-009, PORT-SESS-003, PORT-RTC-002, PORT-RTC-005
def test_flush_without_feed_runs_zero_sample_final_tail() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        terminal = await _wait(lease.flush())
        assert terminal.complete_text == " w0"
        assert terminal.completion.reason == "terminal"
        assert terminal.completion.text == " w0"
        assert len(engine.prompts) == 2
        envelope = engine.prompts[0]["multi_modal_data"]["audio"]
        # Envelope header: [version, n_samples, ...] — the explicit
        # zero-sample final-tail transaction (PORT-SESS-003).
        assert envelope[1] == 0.0
        assert "multi_modal_data" not in engine.prompts[1]
        await lease.release()

    _run(scenario())


def test_finish_without_flush_closes_gracefully() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await _wait(lease.feed(_audio(_CHUNK)))
        await _wait(lease.finish())
        # Graceful end: the final-tail transaction ran, no engine abort.
        assert len(engine.prompts) == 3
        assert engine.aborted == []
        await lease.release()

    _run(scenario())


# ---- failure propagation: ledger.fail carries the ORIGINAL error ---------------


# @spec PORT-RTC-002, PORT-RTC-006
def test_engine_failure_releases_blocked_feed_with_original_error() -> None:
    def boom(item: Any, index: int) -> list[Any]:
        raise ValueError("engine exploded")

    async def scenario() -> None:
        engine = FakeAsyncOmni(script=boom)
        lease, _ = _make_lease(engine)
        with pytest.raises(ValueError, match="engine exploded"):
            await _wait(lease.feed(_audio(_CHUNK)))
        # The stored original error also guards later calls (never hang).
        with pytest.raises(ValueError, match="engine exploded"):
            await _wait(lease.feed(_audio(_CHUNK)))
        with pytest.raises(ValueError, match="engine exploded"):
            await _wait(lease.flush())
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


# @spec PORT-RTC-002, PORT-RTC-006
def test_premature_stream_end_fails_pending_feed() -> None:
    def silent(item: Any, index: int) -> list[Any]:
        return []  # consumes the prompt, never emits the park

    async def scenario() -> None:
        engine = FakeAsyncOmni(script=silent, stop_after=1)
        lease, _ = _make_lease(engine)
        with pytest.raises(RuntimeError, match="generation ended"):
            await _wait(lease.feed(_audio(_CHUNK)))
        with pytest.raises(RuntimeError, match="generation ended"):
            await _wait(lease.flush())
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


# ---- liveness guards (ING-LIFE-010, preserved from the prototype) --------------


# @spec PORT-RTC-005, PORT-RTC-006
def test_feed_after_flush_raises_instead_of_hanging() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await _wait(lease.flush())
        with pytest.raises(RuntimeError, match="finaliz"):
            await _wait(lease.feed(_audio(_CHUNK)))
        await lease.release()

    _run(scenario())


# @spec PORT-RTC-005, PORT-RTC-006
def test_feed_and_flush_after_abort_raise() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await _wait(lease.feed(_audio(_CHUNK)))
        await _wait(lease.abort())
        with pytest.raises(RuntimeError, match="aborted"):
            await _wait(lease.feed(_audio(_CHUNK)))
        with pytest.raises(RuntimeError, match="aborted"):
            await _wait(lease.flush())
        await lease.release()

    _run(scenario())


# ---- abort: cancel + engine abort + idempotence --------------------------------


def test_abort_cancels_consumer_and_aborts_engine_request() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await _wait(lease.feed(_audio(_CHUNK)))
        await _wait(lease.abort())
        assert engine.aborted == ["rt-test-1"]
        await _wait(lease.abort())  # idempotent
        assert engine.aborted == ["rt-test-1"]
        await lease.release()

    _run(scenario())


def test_abort_after_natural_end_skips_engine_abort() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await _wait(lease.flush())
        await _wait(lease.abort())
        # Generation already ended; there is no engine request to abort.
        assert engine.aborted == []
        await lease.release()

    _run(scenario())


def test_abort_before_any_feed_is_safe() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await _wait(lease.abort())
        assert engine.aborted == []
        await lease.release()

    _run(scenario())


# ---- release: retained future-reservation seam, idempotent ---------------------


# @spec PORT-RTC-003, PORT-RTC-005
def test_release_is_idempotent_without_a_parallel_reservation() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await lease.release()
        await lease.release()

    _run(scenario())


# ---- update_locale: one validator, stamped at the next mint --------------------


# @spec PORT-LID-001
def test_update_locale_delegates_to_select_prompt() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await _wait(lease.feed(_audio(_CHUNK)))
        await lease.update_locale("de-DE")
        assert lease.session.prompt_index == PROMPTS["de-DE"]
        await _wait(lease.feed(_audio(_CHUNK)))
        # Envelope header slot 4 is the stamped prompt index: the first
        # mint carries the admission locale, the second the update.
        first = engine.prompts[0]["multi_modal_data"]["audio"]
        second = engine.prompts[1]["multi_modal_data"]["audio"]
        assert first[4] == float(PROMPTS[_LOCALE])
        assert second[4] == float(PROMPTS["de-DE"])
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


# @spec PORT-LID-001
def test_update_locale_rejects_unknown_and_prior_selection_stands() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine, locale="en-US")
        with pytest.raises(ValueError, match="locale"):
            await lease.update_locale("xx-XX")
        assert lease.session.prompt_index == PROMPTS["en-US"]
        await lease.release()

    _run(scenario())


# ---- render wiring: the lazy core precedent (realtime/serving.py) --------------


def test_render_factory_mirrors_core_parse_render_wrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_calls: list[tuple[Any, Any]] = []
    rendered: list[Any] = []

    class FakeStreamingInput:
        def __init__(self, *, prompt: Any) -> None:
            self.prompt = prompt

    def parse_model_prompt(model_config: Any, prompt: Any) -> Any:
        parse_calls.append((model_config, prompt))
        return ("parsed", prompt)

    class FakeRenderer:
        async def render_cmpl_async(self, prompts: list[Any]) -> list[Any]:
            rendered.append(prompts)
            return [("engine-input", prompts[0])]

    for name in ("vllm", "vllm.engine", "vllm.renderers", "vllm.renderers.inputs"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    protocol = types.ModuleType("vllm.engine.protocol")
    protocol.StreamingInput = FakeStreamingInput  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.engine.protocol", protocol)
    preprocess = types.ModuleType("vllm.renderers.inputs.preprocess")
    preprocess.parse_model_prompt = parse_model_prompt  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.renderers.inputs.preprocess", preprocess)

    engine = FakeAsyncOmni()
    engine.renderer = FakeRenderer()
    render = _NS._render_factory(engine)
    result = _run(render({"prompt": "p0"}))
    assert parse_calls == [(engine.model_config, {"prompt": "p0"})]
    assert rendered == [[("parsed", {"prompt": "p0"})]]
    assert isinstance(result, FakeStreamingInput)
    assert result.prompt == ("engine-input", ("parsed", {"prompt": "p0"}))


# ---- structural protocol conformance (PORT-RTC-007) ----------------------------


# @spec PORT-RTC-007
def test_concrete_factory_and_lease_satisfy_protocols_by_shape() -> None:
    engine = FakeAsyncOmni()
    factory = NemotronSessionFactory(engine=engine)
    assert isinstance(factory, SessionFactory)
    lease, _ = _make_lease(engine)
    assert isinstance(lease, SessionLease)
    _run(lease.release())


# @spec PORT-RTC-007
def test_runtime_path_imports_no_endpoint_contract() -> None:
    source = _MODULE_PATH.read_text()
    assert "ephemeral_session" not in source
    assert "AdmissionBusyError" not in source
    assert "class SessionFactory(Protocol)" in source
    assert "class SessionLease(Protocol)" in source


# ---- source-scan: the binding holds no cadence arithmetic (ING-FE-006) ---------


# @spec PORT-RTC-004, ING-FE-006
def test_binding_source_has_no_cadence_arithmetic() -> None:
    source = _MODULE_PATH.read_text()
    banned = (
        "chunk_samples",
        "chunk_ms",
        "_fed",
        "SAMPLE_RATE",
        "RAW_SAMPLES_PER_CHUNK",
        "// ",
        "16000",
        "8960",
        "17920",
        "1120",
        "560ms",
    )
    present = [token for token in banned if token in source]
    assert present == [], (
        f"the engine binding contains cadence/chunk arithmetic: {present}; "
        "PORT owns cadence segmentation and the ledger replaces prediction "
        "(ING-FE-006, PORT-RTC-002)"
    )
