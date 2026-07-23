# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Nemotron ``/v1/audio/transcriptions`` adapter (RFC-1 brief §C/§D).

Two tiers in one file, split exactly as the brief's test plan requires:

GPU-free CPU tier (runs on the macOS loader path): every decision that
is PURE lives in the import-light rules sibling
(``nemotron_transcription_rules.py`` — stdlib + the engine-free
orchestrator module only), loaded by file path with stubbed parents:

- the honest-subset rejection table: ``json``/``text`` accepted;
  ``verbose_json``/``srt``/``vtt``, ``stream=True``, timestamp
  granularities, beam-search controls, ``hotwords``, non-empty
  ``prompt``, ``to_language``, and non-default sampling knobs (the
  decode is pinned greedy, PORT-DEC-005) each draw a NAMED rejection,
  never silent acceptance;
- language normalization: ``None``/empty -> ``"auto"`` (brief §C);
  everything else passes through to the checkpoint-locale gate;
- orchestrator error mapping: ``AdmissionBusyError`` -> the shared
  named capacity error (HTTP 429, PORT-STATE-004);
  ``FinalizationTimeoutError`` -> 504; anything else is NOT mapped
  (propagates);
- source-scans: the adapter holds no cadence arithmetic and never
  imports ``nemotron_asr_ingress`` (Tenet 3); it passes
  ``deadline=None`` (HTTP carries no transport deadline); the model
  file gains the three ``SupportsTranscription`` classmethods but NOT
  the ``supports_transcription`` classvar (task stays off by default,
  round-5 decision 3); the api_server wiring is flag-guarded and traps
  translation to an explicit ``None`` handler.

Pod tier (``vllm`` present; skipped here): the engine-coupled adapter
module itself — delegation to the real orchestrator over a fake
factory/lease, 429/504 mapping through ``create_transcription``,
client cancellation -> the orchestrator's abort path, and the
translation route answering not-implemented on a ``None`` handler.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
import types
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ENTRYPOINTS = _ROOT / "vllm_omni/entrypoints"
_OPENAI = _ENTRYPOINTS / "openai"
_RULES_PATH = _OPENAI / "nemotron_transcription_rules.py"
_ADAPTER_PATH = _OPENAI / "serving_nemotron_transcription.py"
_API_SERVER_PATH = _OPENAI / "api_server.py"
_MODEL_PATH = (
    _ROOT / "vllm_omni/model_executor/models/nemotron_asr/nemotron_asr.py"
)

_HAS_VLLM = importlib.util.find_spec("vllm") is not None
pod = pytest.mark.skipif(
    not _HAS_VLLM, reason="engine-coupled adapter module: pod tier (vllm)"
)

pytestmark = [pytest.mark.cpu]


def _load_chain() -> dict[str, Any]:
    """Load the engine-free modules by file path under dotted names.

    Parents are stubbed so no ``vllm_omni`` package ``__init__`` (which
    imports vllm) executes; ``ephemeral_session`` loads first because
    the rules module imports its exception types.
    """
    for name in (
        "vllm_omni",
        "vllm_omni.entrypoints",
        "vllm_omni.entrypoints.openai",
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    loaded: dict[str, Any] = {}
    modules = [
        (
            "vllm_omni.entrypoints.ephemeral_session",
            _ENTRYPOINTS / "ephemeral_session.py",
            "ephemeral_session",
        ),
        (
            "vllm_omni.entrypoints.openai.nemotron_transcription_rules",
            _RULES_PATH,
            "rules",
        ),
    ]
    for dotted, path, short in modules:
        if dotted in sys.modules and short == "ephemeral_session":
            loaded[short] = sys.modules[dotted]
            continue
        spec = importlib.util.spec_from_file_location(dotted, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = module
        spec.loader.exec_module(module)
        loaded[short] = module
    return loaded


_M = _load_chain()
_RULES = _M["rules"]
_EPH = _M["ephemeral_session"]
first_rejection = _RULES.first_rejection
normalize_language = _RULES.normalize_language
map_orchestrator_error = _RULES.map_orchestrator_error
AdmissionBusyError = _EPH.AdmissionBusyError
FinalizationTimeoutError = _EPH.FinalizationTimeoutError

WAIT_S = 2.0


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


# ---- rejection table: the honest subset (brief §C adapter checklist) -----------


def test_json_and_text_formats_accepted() -> None:
    assert first_rejection(response_format="json") is None
    assert first_rejection(response_format="text") is None


@pytest.mark.parametrize("fmt", ["verbose_json", "srt", "vtt"])
def test_other_response_formats_draw_named_rejection(fmt: str) -> None:
    rejection = first_rejection(response_format=fmt)
    assert rejection is not None
    assert rejection.param == "response_format"
    assert fmt in rejection.message
    assert rejection.status_code == 400


def test_stream_true_draws_named_rejection() -> None:
    rejection = first_rejection(stream=True)
    assert rejection is not None
    assert rejection.param == "stream"
    assert "stream" in rejection.message.lower()


def test_timestamp_granularities_draw_named_rejection() -> None:
    rejection = first_rejection(timestamp_granularities=["segment"])
    assert rejection is not None
    assert rejection.param == "timestamp_granularities"


def test_beam_search_controls_draw_named_rejection() -> None:
    rejection = first_rejection(use_beam_search=True)
    assert rejection is not None
    assert rejection.param == "use_beam_search"
    rejection = first_rejection(n=2)
    assert rejection is not None
    assert rejection.param == "n"


def test_hotwords_draw_named_rejection() -> None:
    rejection = first_rejection(hotwords="nemotron")
    assert rejection is not None
    assert rejection.param == "hotwords"


def test_nonempty_prompt_draws_named_rejection_empty_is_fine() -> None:
    rejection = first_rejection(prompt="style hint")
    assert rejection is not None
    assert rejection.param == "prompt"
    assert first_rejection(prompt="") is None


def test_to_language_draws_named_rejection() -> None:
    rejection = first_rejection(to_language="de")
    assert rejection is not None
    assert rejection.param == "to_language"


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 5),
        ("min_p", 0.1),
        ("seed", 7),
        ("frequency_penalty", 0.5),
        ("repetition_penalty", 1.1),
        ("presence_penalty", 0.5),
        ("max_completion_tokens", 16),
    ],
)
def test_non_default_sampling_knobs_draw_named_rejection(
    knob: str, value: Any
) -> None:
    # The decode is pinned greedy in-model (PORT-DEC-005/007): a knob
    # that cannot take effect is rejected by name, never ignored.
    rejection = first_rejection(**{knob: value})
    assert rejection is not None
    assert rejection.param == knob


def test_request_defaults_pass_the_whole_table() -> None:
    assert first_rejection() is None


# ---- language normalization (brief §C: None -> "auto") -------------------------


def test_normalize_language_maps_missing_to_auto() -> None:
    assert normalize_language(None) == "auto"
    assert normalize_language("") == "auto"
    assert normalize_language("  ") == "auto"


def test_normalize_language_passes_locales_through() -> None:
    assert normalize_language("en-US") == "en-US"
    assert normalize_language("auto") == "auto"


# ---- orchestrator error mapping (PORT-STATE-004 shared capacity error) ---------


def test_admission_busy_maps_to_named_429() -> None:
    mapped = map_orchestrator_error(AdmissionBusyError("pool full"))
    assert mapped is not None
    assert mapped.status_code == 429
    assert mapped.err_type == "TooManyRequestsError"
    assert "capacity" in mapped.message.lower()
    assert "pool full" in mapped.message


def test_finalization_timeout_maps_to_named_504() -> None:
    mapped = map_orchestrator_error(FinalizationTimeoutError(5.0))
    assert mapped is not None
    assert mapped.status_code == 504
    assert mapped.err_type == "GatewayTimeoutError"


def test_other_errors_are_not_mapped() -> None:
    assert map_orchestrator_error(ValueError("boom")) is None
    assert map_orchestrator_error(TimeoutError()) is None


# ---- source-scans: adapter module (GPU-free; the module itself is pod-gated) ---


def test_adapter_source_has_no_cadence_arithmetic() -> None:
    source = _ADAPTER_PATH.read_text()
    banned = (
        "chunk_samples",
        "chunk_ms",
        "_fed",
        "RAW_SAMPLES_PER_CHUNK",
        "// ",
        "16000",
        "16_000",
        "8960",
        "17920",
        "1120",
        "560ms",
    )
    present = [token for token in banned if token in source]
    assert present == [], (
        f"the HTTP adapter contains cadence/chunk arithmetic: {present}; "
        "PORT owns cadence segmentation (ING-FE-006)"
    )


def test_adapter_never_imports_the_ingress_package() -> None:
    # Tenet 3: the OpenAI surface never imports nemotron_asr_ingress
    # (the docstring may NAME the boundary; import lines may not cross
    # it).
    for line in _ADAPTER_PATH.read_text().splitlines():
        if "import" in line:
            assert "nemotron_asr_ingress" not in line, line


def test_adapter_passes_no_transport_deadline() -> None:
    # HTTP carries no transport deadline of its own; the configured
    # finalization limit governs (brief §C, round-5 note).
    assert "deadline=None" in _ADAPTER_PATH.read_text()


def test_adapter_consumes_the_rules_sibling() -> None:
    assert "nemotron_transcription_rules" in _ADAPTER_PATH.read_text()


# ---- source-scans: model classmethods (pod-gated file, ruff/mypy only) ---------


def test_model_gains_supports_transcription_classmethods() -> None:
    source = _MODEL_PATH.read_text()
    for method in (
        "def get_speech_to_text_config",
        "def get_generation_prompt",
        "def validate_language",
    ):
        assert method in source, method
    # The eager base __init__ reads this classvar (base/serving.py:122).
    assert "supports_segment_timestamp" in source


def test_model_does_not_set_the_supports_transcription_classvar() -> None:
    # DECIDED (round-5, brief §D): the task stays off by default; the
    # adapter is constructed by the flag-guarded api_server wiring, not
    # by the capability classvar.
    source = _MODEL_PATH.read_text()
    assert not re.search(r"^\s*supports_transcription\s*[:=]", source, re.M)


# ---- source-scans: guarded opt-in wiring (api_server is vllm-coupled) ----------


def test_wiring_is_flag_guarded_and_traps_translation() -> None:
    source = _API_SERVER_PATH.read_text()
    assert "VLLM_OMNI_EXPERIMENTAL_NEMOTRON_TRANSCRIPTION" in source
    # The translation trap (brief §D): under the opt-in the translation
    # handler is an EXPLICIT None so the route answers not-implemented,
    # never the stock (wrong-for-this-model) path.
    assert re.search(
        r"state\.openai_serving_translation\s*=\s*None", source
    )
    assert "serving_nemotron_transcription" in source
    assert "ServingConcurrencyLimiter" in source


def test_wiring_imports_nemotron_only_behind_the_flag() -> None:
    # Flag OFF must execute zero Nemotron imports: every nemotron import
    # in api_server.py is function-local (lazy), never module-level.
    source = _API_SERVER_PATH.read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line[:1].isspace():
            assert "nemotron" not in stripped, stripped


# ---- pod tier: the engine-coupled adapter module -------------------------------


def _adapter_module() -> Any:
    """Import the adapter module (requires vllm; pod tier)."""
    import importlib

    return importlib.import_module(
        "vllm_omni.entrypoints.openai.serving_nemotron_transcription"
    )


class _FakeLease:
    """SessionLease by shape: records lifecycle, scripted results."""

    def __init__(
        self,
        *,
        transcript: str = "hello world",
        hang_flush: bool = False,
        hang_feed: bool = False,
    ) -> None:
        self.transcript = transcript
        self.hang_flush = hang_flush
        self.hang_feed = hang_feed
        self.fed: list[Any] = []
        self.calls: list[str] = []

    async def feed(self, samples: Any) -> list[str]:
        self.calls.append("feed")
        self.fed.append(samples)
        if self.hang_feed:
            await asyncio.Event().wait()
        return []

    async def flush(self) -> str:
        self.calls.append("flush")
        if self.hang_flush:
            await asyncio.Event().wait()
        return self.transcript

    async def update_locale(self, locale: str) -> None:
        self.calls.append("update_locale")

    async def abort(self) -> None:
        self.calls.append("abort")

    async def finish(self) -> None:
        self.calls.append("finish")

    async def release(self) -> None:
        self.calls.append("release")


class _FakeFactory:
    """SessionFactory by shape: records locales, scripted admission."""

    def __init__(self, lease: _FakeLease | None = None, busy: bool = False) -> None:
        self.lease = lease or _FakeLease()
        self.busy = busy
        self.opened: list[str] = []

    async def open_ephemeral(self, *, locale: str) -> _FakeLease:
        if self.busy:
            raise AdmissionBusyError("no slot")
        self.opened.append(locale)
        return self.lease

    async def open(self, *, cadence: str, locale: str) -> _FakeLease:
        raise AssertionError("the HTTP adapter never opens realtime sessions")


def _make_adapter(
    factory: _FakeFactory,
    *,
    finalization_timeout_s: float = 1.0,
    prompts: dict[str, int] | None = None,
) -> Any:
    """Build the adapter around fakes, bypassing the eager base __init__.

    The eager ``__init__`` chain (engine registry resolution, thread
    pools) is real-engine integration and belongs to the W4 pod round;
    these tests pin OUR execution leg: rejection, delegation, error
    mapping, and cancellation.
    """
    import numpy as np

    module = _adapter_module()
    adapter = object.__new__(module.NemotronServingTranscription)
    adapter.engine_client = SimpleNamespace(errored=False, dead_error=None)
    adapter.models = SimpleNamespace(
        is_base_model=lambda name: True,
        model_name=lambda: "nemotron",
        lora_requests={},
    )
    prompt_table: dict[str, int] = (
        prompts if prompts is not None else {"auto": 0, "en-US": 2}
    )
    adapter.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            prompt_dictionary=dict(prompt_table), num_prompts=128
        )
    )

    class _FakeModelCls:
        @classmethod
        def validate_language(
            cls, language: Any, model_config: Any = None
        ) -> str:
            if language not in prompt_table:
                raise ValueError(f"{language!r} is not a locale")
            return str(language)

    adapter.__dict__["model_cls"] = _FakeModelCls
    adapter.task_type = "transcribe"
    adapter.max_audio_filesize_mb = 25
    adapter.request_logger = None

    async def decode(audio_data: bytes) -> tuple[list[Any], float]:
        return [np.zeros(64, dtype=np.float32)], 1.25

    adapter._decode_and_chunk_speech_async = decode
    adapter._factory = factory
    adapter._finalization_timeout_s = finalization_timeout_s
    adapter._submit_bound_samples = 4096
    return adapter


def _request(**overrides: Any) -> SimpleNamespace:
    """A TranscriptionRequest-shaped namespace at schema defaults."""
    fields: dict[str, Any] = {
        "model": None,
        "language": None,
        "hotwords": None,
        "prompt": "",
        "response_format": "json",
        "timestamp_granularities": [],
        "stream": False,
        "to_language": None,
        "use_beam_search": False,
        "n": 1,
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "seed": None,
        "frequency_penalty": 0.0,
        "repetition_penalty": None,
        "presence_penalty": 0.0,
        "max_completion_tokens": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pod
def test_adapter_delegates_to_orchestrator_with_auto_locale() -> None:
    factory = _FakeFactory()

    async def scenario() -> Any:
        adapter = _make_adapter(factory)
        return await asyncio.wait_for(
            adapter.create_transcription(b"\0" * 16, _request()), WAIT_S
        )

    response = _run(scenario())
    assert response.text == "hello world"
    # Base usage shape: duration seconds rounded UP per the OpenAI spec.
    assert response.usage.seconds == 2
    assert factory.opened == ["auto"]
    # Success path: flush + finish + release, never abort (PORT-EPH-002).
    assert "finish" in factory.lease.calls
    assert "abort" not in factory.lease.calls
    assert factory.lease.calls[-1] == "release"


@pod
def test_adapter_maps_admission_busy_to_shared_429() -> None:
    async def scenario() -> Any:
        adapter = _make_adapter(_FakeFactory(busy=True))
        return await asyncio.wait_for(
            adapter.create_transcription(b"\0" * 16, _request()), WAIT_S
        )

    response = _run(scenario())
    assert response.error.code == 429
    assert response.error.type == "TooManyRequestsError"


@pod
def test_adapter_maps_finalization_timeout_to_504_and_aborts() -> None:
    factory = _FakeFactory(_FakeLease(hang_flush=True))

    async def scenario() -> Any:
        adapter = _make_adapter(factory, finalization_timeout_s=0.05)
        return await asyncio.wait_for(
            adapter.create_transcription(b"\0" * 16, _request()), WAIT_S
        )

    response = _run(scenario())
    assert response.error.code == 504
    assert "abort" in factory.lease.calls
    assert factory.lease.calls[-1] == "release"


@pod
def test_adapter_rejects_unsupported_controls_with_named_400() -> None:
    async def scenario(request: Any) -> Any:
        adapter = _make_adapter(_FakeFactory())
        return await asyncio.wait_for(
            adapter.create_transcription(b"\0" * 16, request), WAIT_S
        )

    for request, param in (
        (_request(stream=True), "stream"),
        (_request(response_format="verbose_json"), "response_format"),
        (_request(hotwords="x"), "hotwords"),
        (_request(prompt="x"), "prompt"),
        (_request(use_beam_search=True), "use_beam_search"),
    ):
        response = _run(scenario(request))
        assert response.error.code == 400, param
        assert response.error.param == param


@pod
def test_adapter_rejects_unknown_language_with_named_400() -> None:
    factory = _FakeFactory()

    async def scenario() -> Any:
        adapter = _make_adapter(factory)
        return await asyncio.wait_for(
            adapter.create_transcription(
                b"\0" * 16, _request(language="xx-XX")
            ),
            WAIT_S,
        )

    response = _run(scenario())
    assert response.error.code == 400
    assert response.error.param == "language"
    assert factory.opened == []


@pod
def test_client_cancellation_propagates_and_aborts() -> None:
    factory = _FakeFactory(_FakeLease(hang_feed=True))

    async def scenario() -> None:
        adapter = _make_adapter(factory)
        task = asyncio.ensure_future(
            adapter.create_transcription(b"\0" * 16, _request())
        )
        await asyncio.sleep(0.05)  # let the feed park
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, WAIT_S)

    _run(scenario())
    # Disconnect -> the orchestrator's abort path, then release.
    assert "abort" in factory.lease.calls
    assert factory.lease.calls[-1] == "release"


@pod
def test_translation_route_answers_not_implemented_on_none_handler() -> None:
    # The translation trap (brief §D): with the handler pinned to None,
    # the mounted route raises NotImplementedError, which the app's
    # generic exception handler maps to a 501 body.
    from vllm.entrypoints.speech_to_text.translation.api_router import (
        create_translations,
    )

    raw_request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(openai_serving_translation=None)
        ),
        headers={},
    )

    # Unwrap the router decorators (with_cancellation/load_aware_call)
    # to call the handler body directly against the None handler.
    handler: Any = create_translations
    while hasattr(handler, "__wrapped__"):
        handler = handler.__wrapped__

    async def scenario() -> None:
        with pytest.raises(NotImplementedError):
            await handler(request=SimpleNamespace(), raw_request=raw_request)

    _run(scenario())
