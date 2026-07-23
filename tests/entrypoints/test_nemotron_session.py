# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The transport-neutral Nemotron session module (RFC-1 brief §A/§B).

GPU-free CPU tier. Drives the concrete engine binding —
``ServingConcurrencyLimiter`` + ``NemotronSessionFactory`` +
``NemotronSessionLease`` — over a FAKE ``AsyncOmni`` (scripted
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
- limiter cap -> ``AdmissionBusyError``; release frees exactly once;
  no slot leak on a post-acquire factory failure; the limiter
  docstring carries the three DECIDED disclaimers (round-5, brief §B);
- the factory mints the canonical ephemeral geometry (the model
  package's ``EPHEMERAL_CADENCE``) without the caller naming it
  (PORT-REGIME-002);
- ``update_locale`` delegates to ``session.select_prompt`` (the one
  validator, PORT-LID-001) and the next mint stamps the new index;
- structural protocol conformance: the concrete factory/lease satisfy
  the ``SessionFactory``/``SessionLease`` runtime protocols WITHOUT
  importing them in the module's runtime path (PORT-EPH-004);
- source-scan: no cadence/chunk arithmetic in the binding.

Loader-runnable: every module is loaded by file path under its
canonical dotted name (parents stubbed), so no ``vllm_omni`` package
``__init__`` (which imports ``vllm``) is touched; the binding's vllm
imports are lazy by design.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
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

    def coerce_param_message_types(
        params: list[Any], is_streaming: bool
    ) -> list[Any]:
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
            "vllm_omni.entrypoints.ephemeral_session",
            _ENTRYPOINTS / "ephemeral_session.py",
            "ephemeral_session",
        ),
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
ServingConcurrencyLimiter = _NS.ServingConcurrencyLimiter
NemotronSessionFactory = _NS.NemotronSessionFactory
NemotronSessionLease = _NS.NemotronSessionLease
_EPH = _M["ephemeral_session"]
AdmissionBusyError = _EPH.AdmissionBusyError
SessionFactory = _EPH.SessionFactory
SessionLease = _EPH.SessionLease
_SESSION = _M["session"]
NemotronRealtimeSession = _SESSION.NemotronRealtimeSession
EPHEMERAL_CADENCE = _SESSION.EPHEMERAL_CADENCE
DEFAULT_CADENCE = _SESSION.DEFAULT_CADENCE
CADENCES = _M["manifests"].CADENCES
RAW_SAMPLES_PER_CHUNK = _M["manifests"].RAW_SAMPLES_PER_CHUNK

pytestmark = [pytest.mark.cpu]

PARK_ID = 13088
PLACEHOLDER_ID = 13089
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
    """Bound every await so a regression HANGS the test never."""
    return await asyncio.wait_for(awaitable, WAIT_S)


def _audio(n: int) -> Any:
    return np.zeros(n, dtype=np.float32)


# ---- fakes ---------------------------------------------------------------------


def _out(ids: list[int], text: str = "") -> Any:
    return SimpleNamespace(
        stage_id=0,
        outputs=[SimpleNamespace(token_ids=list(ids), text=text)],
    )


def _default_script(item: Any, index: int) -> list[Any]:
    return [_out([7, PARK_ID], text=f" w{index}")]


class FakeAsyncOmni:
    """AsyncOmni stand-in: scripted generate outputs, recorded abort.

    ``generate`` mirrors the real keyword surface (``prompt`` async
    generator, ``request_id``, ``sampling_params_list``); the script
    maps each consumed rendered prompt to output batches.
    """

    def __init__(
        self, *, script: Any = None, stop_after: int | None = None
    ) -> None:
        self.model_config = SimpleNamespace(hf_config=_hf())
        self.renderer: Any = None
        self.default_sampling_params_list = (SimpleNamespace(tag="default"),)
        self.script = script or _default_script
        self.stop_after = stop_after
        self.prompts: list[Any] = []
        self.aborted: list[str] = []
        self.request_ids: list[str] = []
        self.sampling: Any = None

    async def generate(
        self, *, prompt: Any, request_id: str, sampling_params_list: Any
    ) -> Any:
        self.request_ids.append(request_id)
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
    limiter: Any = None,
) -> tuple[Any, Any]:
    limiter = limiter or ServingConcurrencyLimiter(max_concurrent=4)
    limiter.acquire()
    session = NemotronRealtimeSession.from_model_config(
        _hf(), cadence=cadence, locale=locale, with_ledger=True
    )
    lease = NemotronSessionLease(
        engine=engine,
        session=session,
        limiter=limiter,
        request_id="rt-test-1",
        render=_passthrough_render,
    )
    return lease, limiter


# ---- ServingConcurrencyLimiter (round-5 DECIDED, brief §B) ---------------------


def test_limiter_sheds_at_cap_and_frees_exactly_once() -> None:
    limiter = ServingConcurrencyLimiter(max_concurrent=2)
    limiter.acquire()
    limiter.acquire()
    with pytest.raises(AdmissionBusyError):
        limiter.acquire()
    limiter.release()
    limiter.acquire()  # freed slot is reusable
    limiter.release()
    limiter.release()
    # Double-release guard: nothing held -> loud protocol error, and the
    # count can never go negative (which would silently widen the cap).
    with pytest.raises(RuntimeError, match="release"):
        limiter.release()
    with pytest.raises(ValueError):
        ServingConcurrencyLimiter(max_concurrent=0)


def test_limiter_docstring_carries_the_three_decided_disclaimers() -> None:
    doc = ServingConcurrencyLimiter.__doc__ or ""
    assert "load shedding" in doc.lower()
    assert "ADMITTED" in doc
    assert "residency" in doc.lower()
    assert "PORT-STATE-004" in doc
    # The owed engine work is named, so nobody mistakes this for it.
    assert "reservation" in doc.lower()


# ---- factory: admission + canonical geometry (PORT-REGIME-002) -----------------


def test_open_ephemeral_mints_canonical_cadence_without_caller_naming_it() -> None:
    assert EPHEMERAL_CADENCE in CADENCES
    engine = FakeAsyncOmni()
    factory = NemotronSessionFactory(
        engine=engine, limiter=ServingConcurrencyLimiter(max_concurrent=2)
    )
    lease = _run(factory.open_ephemeral(locale=_LOCALE))
    geometry = lease.session.geometry
    assert geometry.cadence == EPHEMERAL_CADENCE
    assert geometry.chunk_samples == RAW_SAMPLES_PER_CHUNK[EPHEMERAL_CADENCE]
    assert lease.session.ledger is not None
    _run(lease.release())


def test_open_realtime_admits_caller_cadence_and_locale() -> None:
    engine = FakeAsyncOmni()
    factory = NemotronSessionFactory(
        engine=engine, limiter=ServingConcurrencyLimiter(max_concurrent=2)
    )
    lease = _run(factory.open(cadence=_CADENCE, locale="en-US"))
    assert lease.session.geometry.cadence == _CADENCE
    assert lease.session.prompt_index == PROMPTS["en-US"]
    _run(lease.release())


def test_factory_sheds_when_limiter_full_and_frees_on_release() -> None:
    engine = FakeAsyncOmni()
    limiter = ServingConcurrencyLimiter(max_concurrent=1)
    factory = NemotronSessionFactory(engine=engine, limiter=limiter)
    lease = _run(factory.open_ephemeral(locale=_LOCALE))
    with pytest.raises(AdmissionBusyError):
        _run(factory.open_ephemeral(locale=_LOCALE))
    _run(lease.release())
    lease2 = _run(factory.open_ephemeral(locale=_LOCALE))  # slot came back
    _run(lease2.release())


def test_factory_releases_slot_on_post_acquire_failure() -> None:
    engine = FakeAsyncOmni()
    limiter = ServingConcurrencyLimiter(max_concurrent=1)
    factory = NemotronSessionFactory(engine=engine, limiter=limiter)
    with pytest.raises(ValueError, match="locale"):
        _run(factory.open_ephemeral(locale="not-a-locale"))
    # The failed open leaked nothing: the single slot is still free.
    lease = _run(factory.open_ephemeral(locale=_LOCALE))
    _run(lease.release())


def test_factory_leases_get_distinct_request_ids() -> None:
    engine = FakeAsyncOmni()
    factory = NemotronSessionFactory(
        engine=engine, limiter=ServingConcurrencyLimiter(max_concurrent=2)
    )
    a = _run(factory.open_ephemeral(locale=_LOCALE))
    b = _run(factory.open_ephemeral(locale=_LOCALE))
    assert a.request_id != b.request_id
    _run(a.release())
    _run(b.release())


# ---- feed: real ledger tickets over the real segmenter (F6/R2) -----------------


def test_feed_returns_one_cumulative_hypothesis_per_completed_cadence() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        assert await _wait(lease.feed(_audio(_CHUNK))) == [" w0"]
        assert await _wait(lease.feed(_audio(_CHUNK))) == [" w0 w1"]
        await _wait(lease.abort())
        await lease.release()

    _run(scenario())


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


def test_feed_burst_returns_hypotheses_in_cadence_order() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        results = await _wait(lease.feed(_audio(2 * _CHUNK)))
        assert results == [" w0", " w0 w1"]
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


# ---- flush: drain to stream end (PORT-SESS-003) --------------------------------


def test_flush_drains_final_tail_and_returns_final_transcript() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        assert await _wait(lease.feed(_audio(_CHUNK))) == [" w0"]
        assert await _wait(lease.flush()) == " w0 w1"
        # Two prompts reached the engine: the cadence and the final tail.
        assert len(engine.prompts) == 2
        await _wait(lease.finish())  # no-op after flush
        assert engine.aborted == []
        await lease.release()

    _run(scenario())


def test_flush_without_feed_runs_zero_sample_final_tail() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        assert await _wait(lease.flush()) == " w0"
        assert len(engine.prompts) == 1
        envelope = engine.prompts[0]["multi_modal_data"]["audio"]
        # Envelope header: [version, n_samples, ...] — the explicit
        # zero-sample final-tail transaction (PORT-SESS-003).
        assert envelope[1] == 0.0
        await lease.release()

    _run(scenario())


def test_finish_without_flush_closes_gracefully() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await _wait(lease.feed(_audio(_CHUNK)))
        await _wait(lease.finish())
        # Graceful end: the final-tail transaction ran, no engine abort.
        assert len(engine.prompts) == 2
        assert engine.aborted == []
        await lease.release()

    _run(scenario())


# ---- failure propagation: ledger.fail carries the ORIGINAL error ---------------


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


def test_feed_after_flush_raises_instead_of_hanging() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        lease, _ = _make_lease(engine)
        await _wait(lease.flush())
        with pytest.raises(RuntimeError, match="finaliz"):
            await _wait(lease.feed(_audio(_CHUNK)))
        await lease.release()

    _run(scenario())


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


# ---- release: the limiter slot, exactly once, idempotent -----------------------


def test_release_frees_limiter_slot_exactly_once() -> None:
    async def scenario() -> None:
        engine = FakeAsyncOmni()
        limiter = ServingConcurrencyLimiter(max_concurrent=1)
        lease, _ = _make_lease(engine, limiter=limiter)
        with pytest.raises(AdmissionBusyError):
            limiter.acquire()
        await lease.release()
        await lease.release()  # idempotent: frees once, never twice
        limiter.acquire()  # exactly one slot came back
        with pytest.raises(AdmissionBusyError):
            limiter.acquire()
        limiter.release()

    _run(scenario())


# ---- update_locale: one validator, stamped at the next mint --------------------


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
    monkeypatch.setitem(
        sys.modules, "vllm.renderers.inputs.preprocess", preprocess
    )

    engine = FakeAsyncOmni()
    engine.renderer = FakeRenderer()
    render = _NS._render_factory(engine)
    result = _run(render({"prompt": "p0"}))
    assert parse_calls == [(engine.model_config, {"prompt": "p0"})]
    assert rendered == [[("parsed", {"prompt": "p0"})]]
    assert isinstance(result, FakeStreamingInput)
    assert result.prompt == ("engine-input", ("parsed", {"prompt": "p0"}))


# ---- structural protocol conformance (PORT-EPH-004) ----------------------------


def test_concrete_factory_and_lease_satisfy_protocols_by_shape() -> None:
    engine = FakeAsyncOmni()
    factory = NemotronSessionFactory(
        engine=engine, limiter=ServingConcurrencyLimiter(max_concurrent=1)
    )
    assert isinstance(factory, SessionFactory)
    lease, _ = _make_lease(engine)
    assert isinstance(lease, SessionLease)
    _run(lease.release())


def test_runtime_path_never_imports_the_protocols() -> None:
    source = _MODULE_PATH.read_text()
    # AdmissionBusyError is the ONE sanctioned ephemeral_session import;
    # the protocols are satisfied by shape, never imported (PORT-EPH-004).
    assert re.search(
        r"from vllm_omni\.entrypoints\.ephemeral_session import[^\n]*"
        r"\bAdmissionBusyError\b",
        source,
    )
    for line in source.splitlines():
        if "import" not in line:
            continue
        assert "SessionFactory" not in line, line
        assert "SessionLease" not in line, line


# ---- source-scan: the binding holds no cadence arithmetic (ING-FE-006) ---------


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
