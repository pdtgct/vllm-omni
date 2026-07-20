"""Riva gRPC servicer: three RPCs, dispositions, front-end (ING-GRPC-001..006).

Direct-call tests drive the servicer methods with a fake grpc context
(sans transport), matching the house sans-IO style; one registration
round-trip uses a real in-process grpc server to pin that the
generated ``add_RivaSpeechRecognitionServicer_to_server`` accepts the
servicer unchanged. The python-clients conformance run against a live
server is the ING-2 pod exit gate, not a local test.
"""

from collections.abc import Callable, Mapping
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any

import grpc
import numpy as np
import numpy.typing as npt
import pytest
from ingress_helpers import (
    PROVENANCE,
    FakeTranscriber,
    make_gate,
    make_values,
    pcm16_wav,
    riff_wav,
)
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_asr_pb2_grpc as rasr_grpc
from riva.client.proto import riva_audio_pb2 as raud

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.core import IngressValues, InProcessGate
from nemotron_asr_ingress.frontend import RESAMPLER_ID
from nemotron_asr_ingress.riva_grpc import (
    CAPABILITY_CLASS_FIELDS,
    DISPOSITIONS,
    ENCODING_NAMES,
    RivaAsrServicer,
    build_config_response,
    disposition,
)

MODEL = "nemotron-asr"
CHUNK_MS = 80
CHUNK_SAMPLES = CHUNK_MS * 16  # 16 kHz samples per admitted chunk
CHUNK_BYTES = CHUNK_SAMPLES * 2  # pcm16 @ 16 kHz


class AbortError(Exception):
    """Raised by the fake context's ``abort`` (as real grpc does)."""


class FakeContext:
    """The slice of ``grpc.ServicerContext`` the servicer touches."""

    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        raise AbortError(details)


class FakeOffline:
    """Records the full-context single-shot seam's calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[npt.NDArray[np.float32], str]] = []

    def transcribe(self, audio: Any, target_lang: str) -> str:
        self.calls.append((np.asarray(audio, dtype=np.float32), target_lang))
        return "offline transcript"


class FakeTime:
    """Injected clock + sleep so no test ever waits."""

    def __init__(self) -> None:
        self.t = 0.0
        self.on_sleep: Callable[[], None] | None = None

    def clock(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt
        if self.on_sleep is not None:
            self.on_sleep()


@dataclass
class Harness:
    servicer: RivaAsrServicer
    gate: InProcessGate
    values: IngressValues
    transcribers: list[FakeTranscriber]
    offline: FakeOffline
    recorded: list[Mapping[str, Any]]
    time: FakeTime
    context: FakeContext = field(default_factory=FakeContext)


def make_servicer(**value_overrides: Any) -> Harness:
    values = make_values(**value_overrides)
    gate = make_gate(values)
    transcribers: list[FakeTranscriber] = []

    def factory() -> FakeTranscriber:
        transcriber = FakeTranscriber()
        transcribers.append(transcriber)
        return transcriber

    offline = FakeOffline()
    recorded: list[Mapping[str, Any]] = []
    time = FakeTime()
    servicer = RivaAsrServicer(
        gate=gate,
        values=values,
        provenance=PROVENANCE,
        make_transcriber=factory,
        offline=offline,
        model_name=MODEL,
        chunk_ms=CHUNK_MS,
        clock=time.clock,
        sleep=time.sleep,
        record_session=recorded.append,
    )
    return Harness(
        servicer=servicer,
        gate=gate,
        values=values,
        transcribers=transcribers,
        offline=offline,
        recorded=recorded,
        time=time,
    )


def recognition_config(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "encoding": raud.LINEAR_PCM,
        "sample_rate_hertz": 16000,
        "language_code": "en-US",
    }
    fields.update(overrides)
    return rasr.RecognitionConfig(**fields)


def config_request(config: Any = None, interim: bool = True) -> Any:
    return rasr.StreamingRecognizeRequest(
        streaming_config=rasr.StreamingRecognitionConfig(
            config=config if config is not None else recognition_config(),
            interim_results=interim,
        )
    )


def audio_request(data: bytes) -> Any:
    return rasr.StreamingRecognizeRequest(audio_content=data)


def pcm16(n_samples: int) -> bytes:
    ramp = np.arange(n_samples, dtype=np.int16)
    return ramp.astype("<i2").tobytes()


def run_stream(harness: Harness, requests: list[Any]) -> list[Any]:
    return list(
        harness.servicer.StreamingRecognize(iter(requests), harness.context)
    )


def transcripts(responses: list[Any]) -> list[tuple[str, bool]]:
    return [
        (r.results[0].alternatives[0].transcript, r.results[0].is_final)
        for r in responses
    ]


# ---- ING-GRPC-001: the three RPCs, generated-stub conformance --------------


# @spec ING-GRPC-001
def test_servicer_has_the_three_generated_rpc_names() -> None:
    generated = {
        name
        for name in vars(rasr_grpc.RivaSpeechRecognitionServicer)
        if not name.startswith("_")
    }
    assert generated == {
        "StreamingRecognize",
        "Recognize",
        "GetRivaSpeechRecognitionConfig",
    }
    for name in generated:
        assert callable(getattr(RivaAsrServicer, name))


# @spec ING-GRPC-001
def test_encoding_names_match_the_generated_enum() -> None:
    assert ENCODING_NAMES == {
        raud.LINEAR_PCM: "LINEAR_PCM",
        raud.MULAW: "MULAW",
        raud.ALAW: "ALAW",
    }


# @spec ING-GRPC-001, ING-GRPC-004
def test_generated_registration_serves_the_config_rpc() -> None:
    harness = make_servicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    rasr_grpc.add_RivaSpeechRecognitionServicer_to_server(
        harness.servicer, server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = rasr_grpc.RivaSpeechRecognitionStub(channel)
            response = stub.GetRivaSpeechRecognitionConfig(
                rasr.RivaSpeechRecognitionConfigRequest(), timeout=5.0
            )
    finally:
        server.stop(0)
    assert response.model_config[0].model_name == MODEL


# ---- ING-GRPC-005/006: the disposition matrix -------------------------------


# @spec ING-GRPC-005
def test_dispositions_cover_every_recognition_config_field() -> None:
    # A wheel bump that adds a proto field must break here, loudly,
    # instead of being silently ignored.
    proto_fields = {f.name for f in rasr.RecognitionConfig.DESCRIPTOR.fields}
    assert set(DISPOSITIONS) == proto_fields
    allowed = {"honored", "model-intrinsic", "rejected-non-default"}
    assert set(DISPOSITIONS.values()) <= allowed
    assert {
        name
        for name, kind in DISPOSITIONS.items()
        if kind == "rejected-non-default"
    } >= CAPABILITY_CLASS_FIELDS


# @spec ING-GRPC-005, ING-GRPC-006
def test_dispositions_match_the_design_matrix() -> None:
    honored = {
        "encoding",
        "sample_rate_hertz",
        "language_code",
        "max_alternatives",
        "audio_channel_count",
        "model",
    }
    assert {n for n, k in DISPOSITIONS.items() if k == "honored"} == honored
    intrinsic = {n for n, k in DISPOSITIONS.items() if k == "model-intrinsic"}
    # verbatim_transcripts joined punctuation after the round-1 pod
    # gate: the canonical python-clients default is true (ING-GRPC-006).
    assert intrinsic == {
        "enable_automatic_punctuation",
        "verbatim_transcripts",
    }


# @spec ING-GRPC-005
def test_default_config_is_admissible() -> None:
    values = make_values()
    assert disposition(recognition_config(), values, MODEL) == []


# @spec ING-GRPC-005
def test_empty_language_code_means_auto_never_fuzzy() -> None:
    values = make_values()
    config = recognition_config(language_code="")
    assert disposition(config, values, MODEL) == []
    unknown = recognition_config(language_code="xx-XX")
    (rejection,) = disposition(unknown, values, MODEL)
    assert rejection.code == errors.UNKNOWN_LOCALE
    assert "language_code" in rejection.fields


# @spec ING-GRPC-005
@pytest.mark.parametrize(
    ("overrides", "expected_code", "expected_field"),
    [
        (
            {"profanity_filter": True},
            errors.INVALID_CONFIG_FIELD,
            "profanity_filter",
        ),
        (
            {"max_alternatives": 2},
            errors.INVALID_CONFIG_FIELD,
            "max_alternatives",
        ),
        (
            {"audio_channel_count": 2},
            errors.INVALID_CONFIG_FIELD,
            "audio_channel_count",
        ),
        (
            {"enable_separate_recognition_per_channel": True},
            errors.INVALID_CONFIG_FIELD,
            "enable_separate_recognition_per_channel",
        ),
        ({"model": "somebody-else"}, errors.INVALID_CONFIG_FIELD, "model"),
        (
            {"enable_word_time_offsets": True},
            errors.UNSUPPORTED_CAPABILITY,
            "enable_word_time_offsets",
        ),
        (
            {"speech_contexts": [rasr.SpeechContext(phrases=["boost me"])]},
            errors.UNSUPPORTED_CAPABILITY,
            "speech_contexts",
        ),
        (
            {
                "diarization_config": rasr.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True
                )
            },
            errors.UNSUPPORTED_CAPABILITY,
            "diarization_config",
        ),
        (
            {"endpointing_config": rasr.EndpointingConfig(stop_threshold=0.8)},
            errors.UNSUPPORTED_CAPABILITY,
            "endpointing_config",
        ),
    ],
)
def test_non_default_rejected_fields_name_themselves(
    overrides: dict[str, Any], expected_code: str, expected_field: str
) -> None:
    values = make_values()
    (rejection,) = disposition(recognition_config(**overrides), values, MODEL)
    assert rejection.code == expected_code
    assert expected_field in rejection.fields


# @spec ING-GRPC-005
def test_custom_configuration_is_rejected_non_empty() -> None:
    values = make_values()
    config = recognition_config()
    config.custom_configuration["anything"] = "at all"
    (rejection,) = disposition(config, values, MODEL)
    assert rejection.code == errors.INVALID_CONFIG_FIELD
    assert "custom_configuration" in rejection.fields


# @spec ING-GRPC-005
def test_default_values_of_rejected_fields_stay_admissible() -> None:
    values = make_values()
    config = recognition_config(
        profanity_filter=False,
        verbatim_transcripts=False,
        max_alternatives=1,
        audio_channel_count=1,
        model=MODEL,
    )
    assert disposition(config, values, MODEL) == []


# @spec ING-FE-001, ING-GRPC-005
@pytest.mark.parametrize(
    ("encoding", "rate"),
    [
        (raud.FLAC, 16000),
        (raud.OGGOPUS, 16000),
        (raud.MULAW, 16000),  # G.711 declared at 16 kHz: a client error
        (raud.ALAW, 16000),
        (raud.LINEAR_PCM, 44100),
    ],
)
def test_off_matrix_formats_name_both_fields(encoding: int, rate: int) -> None:
    values = make_values()
    config = recognition_config(encoding=encoding, sample_rate_hertz=rate)
    (rejection,) = disposition(config, values, MODEL)
    assert rejection.code == errors.UNSUPPORTED_FORMAT
    assert "encoding" in rejection.fields
    assert any("sample_rate" in name for name in rejection.fields)


# @spec ING-GRPC-006
def test_punctuation_is_model_intrinsic_under_either_value() -> None:
    values = make_values()
    for value in (True, False):
        config = recognition_config(enable_automatic_punctuation=value)
        assert disposition(config, values, MODEL) == []


# @spec ING-GRPC-006
def test_verbatim_transcripts_is_model_intrinsic_under_either_value() -> None:
    # The canonical python-clients tools send verbatim_transcripts=true
    # by default (transcribe_file.py @ 5a443b5): both values admissible.
    values = make_values()
    for value in (True, False):
        config = recognition_config(verbatim_transcripts=value)
        assert disposition(config, values, MODEL) == []


# @spec ING-GRPC-007
def test_unspecified_encoding_defers_instead_of_rejecting() -> None:
    values = make_values()
    config = recognition_config(
        encoding=raud.ENCODING_UNSPECIFIED, sample_rate_hertz=0
    )
    assert disposition(config, values, MODEL) == []
    # A declared rate rides along as a constraint, not a rejection.
    constrained = recognition_config(
        encoding=raud.ENCODING_UNSPECIFIED, sample_rate_hertz=16000
    )
    assert disposition(constrained, values, MODEL) == []


# @spec ING-GRPC-005, ING-ERR-003
def test_a_config_with_many_violations_names_every_field() -> None:
    values = make_values()
    config = recognition_config(
        profanity_filter=True,
        enable_word_time_offsets=True,
        max_alternatives=3,
    )
    rejections = disposition(config, values, MODEL)
    named = {name for r in rejections for name in r.fields}
    assert {
        "profanity_filter",
        "enable_word_time_offsets",
        "max_alternatives",
    } <= named


# ---- ING-GRPC-002: StreamingRecognize ---------------------------------------


# @spec ING-GRPC-002, ING-LIFE-001
def test_first_message_must_be_the_streaming_config() -> None:
    harness = make_servicer()
    with pytest.raises(AbortError):
        run_stream(harness, [audio_request(pcm16(CHUNK_SAMPLES))])
    assert harness.context.code == grpc.StatusCode.FAILED_PRECONDITION
    assert harness.context.details is not None
    assert errors.PROTOCOL_ORDER in harness.context.details


# @spec ING-GRPC-002, ING-CORE-005
def test_interim_results_carry_the_cumulative_hypothesis() -> None:
    harness = make_servicer()
    requests = [
        config_request(interim=True),
        audio_request(pcm16(2 * CHUNK_SAMPLES)),  # two exact chunks
        audio_request(pcm16(CHUNK_SAMPLES // 2)),  # a sub-chunk residual
    ]
    responses = run_stream(harness, requests)
    got = transcripts(responses)
    # Riva interim results are cumulative (never deltas): the dialect
    # projection of ING-CORE-005's cumulative hypothesis is identity.
    assert got[0] == ("hey", False)
    assert got[1] == ("hey there", False)
    assert got[-1] == ("hey there [flushed]", True)
    assert [is_final for _, is_final in got].count(True) == 1
    (transcriber,) = harness.transcribers
    assert all(len(chunk) == CHUNK_SAMPLES for chunk in transcriber.steps)
    assert transcriber.flush_called


# @spec ING-GRPC-002
def test_interim_results_false_suppresses_partials() -> None:
    harness = make_servicer()
    responses = run_stream(
        harness,
        [
            config_request(interim=False),
            audio_request(pcm16(2 * CHUNK_SAMPLES)),
        ],
    )
    got = transcripts(responses)
    assert len(got) == 1
    assert got[0][1] is True


# @spec ING-GRPC-002, ING-LIFE-003
def test_half_close_with_zero_audio_yields_one_empty_final() -> None:
    harness = make_servicer()
    responses = run_stream(harness, [config_request()])
    got = transcripts(responses)
    assert got == [("", True)]
    (transcriber,) = harness.transcribers
    assert not transcriber.flush_called  # PORT's tail never runs on nothing


# @spec ING-ADM-001, ING-ERR-001
def test_busy_aborts_resource_exhausted() -> None:
    harness = make_servicer(watermark=1, admission_queue=0)
    assert harness.gate.request("other-session", now=0.0) is not None
    with pytest.raises(AbortError):
        run_stream(harness, [config_request()])
    assert harness.context.code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert harness.context.details is not None
    assert errors.BUSY in harness.context.details


# @spec ING-GRPC-005, ING-ERR-003
def test_rejected_config_aborts_naming_the_field_taking_no_slot() -> None:
    harness = make_servicer()
    config = recognition_config(profanity_filter=True)
    with pytest.raises(AbortError):
        run_stream(harness, [config_request(config=config)])
    assert harness.context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert harness.context.details is not None
    assert "profanity_filter" in harness.context.details
    assert harness.gate.active == 0


# @spec ING-GRPC-005
def test_capability_class_rejections_abort_unimplemented() -> None:
    harness = make_servicer()
    config = recognition_config(
        diarization_config=rasr.SpeakerDiarizationConfig(
            enable_speaker_diarization=True
        )
    )
    with pytest.raises(AbortError):
        run_stream(harness, [config_request(config=config)])
    assert harness.context.code == grpc.StatusCode.UNIMPLEMENTED
    assert harness.context.details is not None
    assert "diarization_config" in harness.context.details


# @spec ING-FE-001
def test_off_matrix_format_aborts_naming_both_fields() -> None:
    harness = make_servicer()
    config = recognition_config(encoding=raud.MULAW, sample_rate_hertz=16000)
    with pytest.raises(AbortError):
        run_stream(harness, [config_request(config=config)])
    assert harness.context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert harness.context.details is not None
    assert "encoding" in harness.context.details
    assert "sample_rate" in harness.context.details


# @spec ING-GRPC-002, ING-FE-002, ING-FE-004
def test_mulaw_8k_session_feeds_exact_16k_chunks() -> None:
    harness = make_servicer()
    config = recognition_config(encoding=raud.MULAW, sample_rate_hertz=8000)
    requests = [
        config_request(config=config),
        # 640 8 kHz samples -> exactly one 1280-sample 16 kHz chunk.
        audio_request(bytes(range(256)) * 2 + bytes(128)),
    ]
    responses = run_stream(harness, requests)
    (transcriber,) = harness.transcribers
    assert all(len(chunk) == CHUNK_SAMPLES for chunk in transcriber.steps)
    assert len(transcriber.steps) >= 1
    assert transcripts(responses)[-1][1] is True


# @spec ING-FE-004
def test_resampled_session_records_the_resampler_identifier() -> None:
    harness = make_servicer()
    config = recognition_config(encoding=raud.MULAW, sample_rate_hertz=8000)
    run_stream(harness, [config_request(config=config)])
    (provenance,) = harness.recorded
    assert provenance["resampler_identifier"] == RESAMPLER_ID
    assert provenance["precision_policy_id"]  # base provenance rides along


# @spec ING-FE-004
def test_native_16k_session_records_no_resampler() -> None:
    harness = make_servicer()
    run_stream(harness, [config_request()])
    (provenance,) = harness.recorded
    assert "resampler_identifier" not in provenance


# @spec ING-GRPC-005, ING-ERR-001
def test_runtime_config_is_never_silently_ignored() -> None:
    # Not a RecognitionConfig field, but the same honest-subset rule:
    # a request rider this v1 does not honor answers, never drops.
    request = rasr.StreamingRecognizeRequest(audio_content=b"")
    request.runtime_config["hotword"] = "boost"
    harness = make_servicer()
    with pytest.raises(AbortError):
        run_stream(harness, [config_request(), request])
    assert harness.context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert harness.context.details is not None
    assert "runtime_config" in harness.context.details


# @spec ING-ADM-002
def test_queued_admission_admits_when_a_slot_frees() -> None:
    harness = make_servicer(watermark=1, admission_queue=1)
    assert harness.gate.request("other-session", now=0.0) is not None

    def free_the_slot() -> None:
        if harness.gate.active:
            harness.gate.release("other-session")

    harness.time.on_sleep = free_the_slot
    responses = run_stream(
        harness,
        [config_request(), audio_request(pcm16(CHUNK_SAMPLES))],
    )
    got = transcripts(responses)
    assert got[-1][1] is True
    assert got[-1][0] == "hey"


# @spec ING-ADM-002, ING-ERR-002
def test_admission_wait_timeout_aborts_deadline_exceeded() -> None:
    harness = make_servicer(
        watermark=1, admission_queue=1, admission_wait_s=5.0
    )
    assert harness.gate.request("other-session", now=0.0) is not None
    with pytest.raises(AbortError):
        run_stream(harness, [config_request()])
    assert harness.context.code == grpc.StatusCode.DEADLINE_EXCEEDED
    assert harness.context.details is not None
    assert errors.ADMISSION_WAIT_TIMEOUT in harness.context.details


# @spec ING-CORE-001
def test_a_core_wanting_transcriber_is_bound_before_first_step() -> None:
    # The per-session assembly the β script does by hand (serve_ws:
    # lazy.core = session_core): a transcriber exposing `core` reads
    # the admitted config from its session core, so the servicer must
    # wire the backref before any audio steps.
    class CoreWantingTranscriber(FakeTranscriber):
        def __init__(self) -> None:
            super().__init__()
            self.core: Any = None
            self.core_at_first_step: Any = None

        def step(self, chunk: Any) -> str:
            if self.core_at_first_step is None:
                self.core_at_first_step = self.core
            return super().step(chunk)

    values = make_values()
    gate = make_gate(values)
    bound = CoreWantingTranscriber()
    servicer = RivaAsrServicer(
        gate=gate,
        values=values,
        provenance=PROVENANCE,
        make_transcriber=lambda: bound,
        offline=FakeOffline(),
        model_name=MODEL,
        chunk_ms=CHUNK_MS,
    )
    context = FakeContext()
    responses = list(
        servicer.StreamingRecognize(
            iter([config_request(), audio_request(pcm16(CHUNK_SAMPLES))]),
            context,
        )
    )
    assert transcripts(responses)[-1][1] is True
    core = bound.core_at_first_step
    assert core is not None
    assert core.config is not None
    assert core.config.chunk_ms == CHUNK_MS


# ---- ING-GRPC-007: the canonical no-declared-format shape -------------------


def canonical_config_request(interim: bool = True, **overrides: Any) -> Any:
    """The python-clients shape: no encoding, no rate declared."""
    fields: dict[str, Any] = {
        "encoding": raud.ENCODING_UNSPECIFIED,
        "sample_rate_hertz": 0,
    }
    fields.update(overrides)
    return config_request(config=recognition_config(**fields), interim=interim)


# @spec ING-GRPC-007
def test_canonical_shape_streams_a_pcm16_wav() -> None:
    n_samples = 2 * CHUNK_SAMPLES + CHUNK_SAMPLES // 2
    harness = make_servicer()
    responses = run_stream(
        harness,
        [canonical_config_request(), audio_request(pcm16_wav(n_samples))],
    )
    got = transcripts(responses)
    assert got[-1] == ("hey there [flushed]", True)
    (transcriber,) = harness.transcribers
    assert all(len(chunk) == CHUNK_SAMPLES for chunk in transcriber.steps)
    # The header was stripped, never decoded as audio: every data
    # sample (and nothing else) reached the seam.
    assert transcriber.flushed is not None
    total = sum(len(c) for c in transcriber.steps) + len(transcriber.flushed)
    assert total == n_samples


# @spec ING-GRPC-007
def test_a_header_split_across_messages_still_resolves() -> None:
    wav = pcm16_wav(CHUNK_SAMPLES)
    harness = make_servicer()
    responses = run_stream(
        harness,
        [
            canonical_config_request(),
            audio_request(wav[:6]),
            audio_request(wav[6:20]),
            audio_request(wav[20:]),
        ],
    )
    got = transcripts(responses)
    assert got[-1] == ("hey", True)
    (transcriber,) = harness.transcribers
    assert len(transcriber.steps) == 1
    assert len(transcriber.steps[0]) == CHUNK_SAMPLES


# @spec ING-GRPC-007, ING-FE-004
def test_canonical_mulaw_wav_resamples_and_records_provenance() -> None:
    wav = riff_wav(7, bytes(range(256)) * 2 + bytes(128), rate=8000, bits=8)
    harness = make_servicer()
    responses = run_stream(
        harness, [canonical_config_request(), audio_request(wav)]
    )
    assert transcripts(responses)[-1][1] is True
    (transcriber,) = harness.transcribers
    assert all(len(chunk) == CHUNK_SAMPLES for chunk in transcriber.steps)
    (provenance,) = harness.recorded
    assert provenance["resampler_identifier"] == RESAMPLER_ID


# @spec ING-GRPC-007
def test_declared_rate_is_a_constraint_the_header_must_match() -> None:
    harness = make_servicer()
    with pytest.raises(AbortError):
        run_stream(
            harness,
            [
                canonical_config_request(sample_rate_hertz=8000),
                audio_request(pcm16_wav(64, rate=16000)),
            ],
        )
    assert harness.context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert harness.context.details is not None
    assert "sample_rate_hertz" in harness.context.details


# @spec ING-GRPC-007
def test_a_headerless_stream_with_no_declared_format_aborts() -> None:
    harness = make_servicer()
    with pytest.raises(AbortError):
        run_stream(
            harness,
            [
                canonical_config_request(),
                audio_request(b"\xde\xad\xbe\xef" * 8),
            ],
        )
    assert harness.context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert harness.context.details is not None
    assert errors.UNSUPPORTED_FORMAT in harness.context.details


# @spec ING-GRPC-007, ING-LIFE-003
def test_deferred_zero_audio_session_finalizes_empty() -> None:
    harness = make_servicer()
    responses = run_stream(harness, [canonical_config_request()])
    assert transcripts(responses) == [("", True)]
    # Provenance still records exactly once — resampler-less, since no
    # audio ever resolved a format.
    (provenance,) = harness.recorded
    assert "resampler_identifier" not in provenance


# @spec ING-GRPC-003, ING-GRPC-007
def test_recognize_accepts_the_canonical_wav_shape() -> None:
    n_samples = 3 * CHUNK_SAMPLES + 21
    harness = make_servicer()
    request = rasr.RecognizeRequest(
        config=recognition_config(
            encoding=raud.ENCODING_UNSPECIFIED, sample_rate_hertz=0
        ),
        audio=pcm16_wav(n_samples),
    )
    response = harness.servicer.Recognize(request, harness.context)
    assert response.results[0].alternatives[0].transcript
    ((clip, _),) = harness.offline.calls
    assert len(clip) == n_samples  # header stripped, data intact


# @spec ING-GRPC-003, ING-GRPC-007
def test_recognize_canonical_alaw_wav_reaches_the_seam_at_16k() -> None:
    payload = bytes(range(256)) * 4
    harness = make_servicer()
    request = rasr.RecognizeRequest(
        config=recognition_config(
            encoding=raud.ENCODING_UNSPECIFIED, sample_rate_hertz=0
        ),
        audio=riff_wav(6, payload, rate=8000, bits=8, with_fact=True),
    )
    harness.servicer.Recognize(request, harness.context)
    ((clip, _),) = harness.offline.calls
    assert len(clip) == 2 * len(payload)


# @spec ING-GRPC-006
def test_config_response_states_the_verbatim_deviation() -> None:
    values = make_values()
    response = build_config_response(values, MODEL)
    parameters = dict(response.model_config[0].parameters)
    assert "intrinsic" in parameters["verbatim_transcripts"]


# ---- ING-GRPC-003: Recognize ------------------------------------------------


# @spec ING-GRPC-003
def test_recognize_is_a_full_context_single_shot() -> None:
    harness = make_servicer()
    n_samples = 5 * CHUNK_SAMPLES + 17  # deliberately not chunk-shaped
    request = rasr.RecognizeRequest(
        config=recognition_config(), audio=pcm16(n_samples)
    )
    response = harness.servicer.Recognize(request, harness.context)
    transcript = response.results[0].alternatives[0].transcript
    assert transcript == "offline transcript"
    # One whole-clip call on the offline seam; never a chunked session.
    ((clip, target_lang),) = harness.offline.calls
    assert len(clip) == n_samples
    assert target_lang == "en-US"
    assert harness.transcribers == []


# @spec ING-GRPC-003, ING-FE-004
def test_recognize_telephony_input_reaches_the_seam_at_16k() -> None:
    harness = make_servicer()
    request = rasr.RecognizeRequest(
        config=recognition_config(encoding=raud.ALAW, sample_rate_hertz=8000),
        audio=bytes(range(256)) * 4,
    )
    harness.servicer.Recognize(request, harness.context)
    ((clip, _),) = harness.offline.calls
    assert len(clip) == 2 * 256 * 4  # resampled to 16 kHz, whole clip


# @spec ING-GRPC-005
def test_recognize_dispositions_the_config_too() -> None:
    harness = make_servicer()
    request = rasr.RecognizeRequest(
        config=recognition_config(profanity_filter=True), audio=pcm16(64)
    )
    with pytest.raises(AbortError):
        harness.servicer.Recognize(request, harness.context)
    assert harness.context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert harness.context.details is not None
    assert "profanity_filter" in harness.context.details
    assert harness.offline.calls == []


# @spec ING-GRPC-005
def test_recognize_unknown_locale_aborts_naming_the_field() -> None:
    harness = make_servicer()
    request = rasr.RecognizeRequest(
        config=recognition_config(language_code="xx-XX"), audio=pcm16(64)
    )
    with pytest.raises(AbortError):
        harness.servicer.Recognize(request, harness.context)
    assert harness.context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert harness.context.details is not None
    assert "language_code" in harness.context.details


# ---- ING-GRPC-004/006: GetRivaSpeechRecognitionConfig -----------------------


def joined_parameters(response: Any) -> str:
    entry = response.model_config[0]
    return " ".join(
        f"{key}={value}" for key, value in sorted(entry.parameters.items())
    )


# @spec ING-GRPC-004
def test_config_rpc_reports_the_static_surface() -> None:
    harness = make_servicer()
    response = harness.servicer.GetRivaSpeechRecognitionConfig(
        rasr.RivaSpeechRecognitionConfigRequest(), harness.context
    )
    entry = response.model_config[0]
    assert entry.model_name == MODEL
    text = joined_parameters(response)
    for token in ("8000", "16000", "LINEAR_PCM", "MULAW", "ALAW"):
        assert token in text
    for locale in harness.values.locales:
        assert locale in text
    assert "streaming=true" in text.lower()
    assert "offline=true" in text.lower()


# @spec ING-GRPC-004, ING-GRPC-006
def test_config_response_builder_states_the_punctuation_deviation() -> None:
    values = make_values()
    response = build_config_response(values, MODEL)
    parameters = dict(response.model_config[0].parameters)
    (punctuation_key,) = [key for key in parameters if "punctuation" in key]
    assert "intrinsic" in parameters[punctuation_key]
