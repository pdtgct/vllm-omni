"""NIM-realtime WS adapter: dialect codec over the session core (ING-NIMWS).

Sans-IO direct-call tests: JSON-shaped dict events in and out, a fake
transcriber behind the core, injected time — matching the house style.
Wire shapes are pinned to the public NIM Realtime API reference and to
what the canonical ``python-clients`` ``realtime_asr_client.py`` @
``5a443b5`` actually sends (the ING-3 conformance gate); the
``realtime.py`` run against a live server is the pod exit gate, not a
local test.

Tests annotated ING-NIMWS-009 bind the decided spec (Pete,
2026-07-13): the canonical client's file mode declares
``input_audio_format: "none"`` and streams raw WAV bytes
header-included, so ``"none"`` defers format resolution to the RIFF
header (mirroring ING-GRPC-007) — a documented NIM-compatibility
deviation, never part of the honored format enum.
"""

import base64
import inspect
import json
from typing import Any

import numpy as np
import pytest
from ingress_helpers import (
    FakeTranscriber,
    make_core,
    make_gate,
    make_values,
    pcm16_wav,
)

from nemotron_asr_ingress import errors, nim_ws
from nemotron_asr_ingress.core import InProcessGate, SessionCore
from nemotron_asr_ingress.events import AdmissionOutcome
from nemotron_asr_ingress.frontend import ACCEPT_MATRIX, RESAMPLER_ID
from nemotron_asr_ingress.nim_ws import (
    CAPABILITY_CLASS_FIELDS,
    CLOSE_CODES,
    DEFERRED_FORMAT,
    ECHO_KEYS,
    FAILED_CODES,
    GRPC_FIELD_NAMES,
    NIM_FORMAT_NAMES,
    RECOGNITION_DISPOSITIONS,
    SECTION_DISPOSITIONS,
    NimRealtimeAdapter,
    default_session_object,
    disposition_session,
    intent_close_code,
    validate_nim_format,
)
from nemotron_asr_ingress.riva_grpc import (
    CAPABILITY_CLASS_FIELDS as GRPC_CAPABILITY_FIELDS,
)
from nemotron_asr_ingress.riva_grpc import (
    DISPOSITIONS as GRPC_DISPOSITIONS,
)

MODEL = "nemotron-asr"
CHUNK_MS = 80
CHUNK_SAMPLES = CHUNK_MS * 16  # 16 kHz samples per admitted chunk
CHUNK_BYTES = CHUNK_SAMPLES * 2  # pcm16

UPDATE = "transcription_session.update"
UPDATED = "transcription_session.updated"
APPEND = "input_audio_buffer.append"
COMMIT = "input_audio_buffer.commit"
COMMITTED = "input_audio_buffer.committed"
CLEAR = "input_audio_buffer.clear"
CLEARED = "input_audio_buffer.cleared"
DONE = "input_audio_buffer.done"
DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"


# ---- helpers ---------------------------------------------------------------


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def pcm16(n_samples: int, start: int = 0) -> bytes:
    ramp = np.arange(start, start + n_samples, dtype=np.int16)
    return ramp.astype("<i2").tobytes()


def make_adapter(
    values: Any = None,
    gate: InProcessGate | None = None,
    chunk_ms: int = CHUNK_MS,
) -> tuple[NimRealtimeAdapter, SessionCore, FakeTranscriber, list[Any]]:
    core, fake, _the_gate = make_core(gate=gate, values=values)
    recorded: list[Any] = []
    adapter = NimRealtimeAdapter(
        core, MODEL, chunk_ms=chunk_ms, record_session=recorded.append
    )
    return adapter, core, fake, recorded


def session_dict(**overrides: Any) -> dict[str, Any]:
    """A minimal admissible session object; overrides merge shallow."""
    session: dict[str, Any] = {
        "input_audio_format": "pcm16",
        "input_audio_params": {"sample_rate_hz": 16000, "num_channels": 1},
        "input_audio_transcription": {"language": "en-US"},
    }
    session.update(overrides)
    return session


def update_event(**overrides: Any) -> dict[str, Any]:
    return {"type": UPDATE, "session": session_dict(**overrides)}


def append_event(data: bytes) -> dict[str, Any]:
    return {"type": APPEND, "audio": b64(data)}


async def configure(
    adapter: NimRealtimeAdapter, now: float = 0.0, **overrides: Any
) -> list[dict[str, Any]]:
    return await adapter.on_event(update_event(**overrides), now)


async def admitted_adapter(
    **value_overrides: Any,
) -> tuple[NimRealtimeAdapter, SessionCore, FakeTranscriber, list[Any]]:
    values = make_values(**value_overrides)
    adapter, core, fake, recorded = make_adapter(values=values)
    replies = await configure(adapter)
    assert [e["type"] for e in replies] == [UPDATED]
    return adapter, core, fake, recorded


def types(events: list[dict[str, Any]]) -> list[str]:
    return [e["type"] for e in events]


def only_error(events: list[dict[str, Any]]) -> dict[str, Any]:
    (event,) = events
    assert event["type"] == "error"
    return event


# ---- contract greens: constants and catalog parity -------------------------


# @spec ING-NIMWS-007
def test_format_names_cover_the_accept_matrix_exactly() -> None:
    dialect_cells = {
        ("pcm16", 16000),
        ("pcm16", 8000),
        ("g711_ulaw", 8000),
        ("g711_alaw", 8000),
    }
    mapped = {(NIM_FORMAT_NAMES[name], rate) for name, rate in dialect_cells}
    assert mapped == ACCEPT_MATRIX
    assert DEFERRED_FORMAT not in NIM_FORMAT_NAMES


# @spec ING-NIMWS-008
def test_recognition_dispositions_mirror_the_grpc_matrix() -> None:
    # Field-for-field parity with the gRPC matrix, joined through the
    # dialect-name map; a class change on either side breaks here.
    assert set(GRPC_FIELD_NAMES) == set(RECOGNITION_DISPOSITIONS)
    for nim_name, grpc_name in GRPC_FIELD_NAMES.items():
        assert (
            RECOGNITION_DISPOSITIONS[nim_name] == GRPC_DISPOSITIONS[grpc_name]
        ), nim_name


# @spec ING-NIMWS-008
def test_section_dispositions_mirror_the_grpc_capability_classes() -> None:
    assert set(SECTION_DISPOSITIONS) == {
        "speaker_diarization",
        "word_boosting",
        "endpointing_config",
        "custom_configuration",
    }
    assert set(SECTION_DISPOSITIONS.values()) == {"rejected-non-default"}
    # The dialect's sections map onto gRPC capability classes:
    # speaker_diarization <-> diarization_config, word_boosting <->
    # speech_contexts, endpointing_config <-> itself.
    assert {
        "diarization_config",
        "speech_contexts",
        "endpointing_config",
    } <= GRPC_CAPABILITY_FIELDS
    assert {
        "speaker_diarization",
        "word_boosting",
        "endpointing_config",
        "prompt",
        "enable_word_time_offsets",
    } == CAPABILITY_CLASS_FIELDS
    # custom_configuration is a scalar-class rejection on both dialects.
    assert "custom_configuration" not in CAPABILITY_CLASS_FIELDS
    assert "custom_configuration" not in GRPC_CAPABILITY_FIELDS


# @spec ING-ERR-001
def test_close_and_failed_sets_follow_the_catalog_nim_column() -> None:
    catalog = errors.catalog()
    assert {
        code for code, p in catalog.items() if p.nim == "error+close"
    } == CLOSE_CODES
    assert {
        code for code, p in catalog.items() if p.nim == "transcription.failed"
    } == FAILED_CODES
    assert {errors.INVALID_AUDIO, errors.INTERNAL} == FAILED_CODES
    assert not CLOSE_CODES & FAILED_CODES


# ---- ING-NIMWS-002: intent gate ---------------------------------------------


# @spec ING-NIMWS-002
def test_transcription_intent_is_accepted() -> None:
    assert intent_close_code("intent=transcription") is None
    assert intent_close_code("intent=transcription&x=1") is None


# @spec ING-NIMWS-002
@pytest.mark.parametrize(
    "query",
    ["", "foo=bar", "intent=translation", "intent=", "intent"],
)
def test_missing_or_unsupported_intent_closes_1008(query: str) -> None:
    assert intent_close_code(query) == 1008


# ---- ING-NIMWS-001: the default session object ------------------------------


# @spec ING-NIMWS-001
def test_default_session_object_shape() -> None:
    values = make_values()
    obj = default_session_object(MODEL, values)
    assert obj["id"].startswith("sess_")
    assert obj["object"] == "realtime.transcription_session"
    assert obj["modalities"] == ["text"]
    assert obj["input_audio_format"] == "pcm16"
    assert obj["input_audio_params"] == {
        "sample_rate_hz": 16000,
        "num_channels": 1,
    }
    transcription = obj["input_audio_transcription"]
    assert transcription["model"] == MODEL
    assert transcription["language"] in values.locales
    assert transcription["prompt"] == ""
    assert set(obj["recognition_config"]) == set(RECOGNITION_DISPOSITIONS)
    assert obj["speaker_diarization"]["enable_speaker_diarization"] is False
    assert obj["word_boosting"]["enable_word_boosting"] is False
    assert obj["word_boosting"]["word_boosting_list"] == []
    assert set(obj["endpointing_config"].values()) == {0}
    assert obj["client_secret"] is None


# @spec ING-NIMWS-001
def test_default_object_ids_are_fresh_per_call() -> None:
    values = make_values()
    first = default_session_object(MODEL, values)
    second = default_session_object(MODEL, values)
    assert first["id"] != second["id"]


# @spec ING-NIMWS-001, ING-NIMWS-008
def test_default_session_object_round_trips_unrejected() -> None:
    # The canonical client copies the POST response and echoes it back
    # as its first transcription_session.update: our own default
    # object must disposition clean through our own matrix.
    values = make_values()
    obj = default_session_object(MODEL, values)
    assert disposition_session(obj, values, MODEL) == []


# @spec ING-NIMWS-001, ING-NIMWS-004
async def test_echoed_default_object_is_admitted() -> None:
    values = make_values()
    adapter, _core, _fake, _recorded = make_adapter(values=values)
    echoed = default_session_object(MODEL, values)
    replies = await adapter.on_event({"type": UPDATE, "session": echoed}, 0.0)
    assert types(replies) == [UPDATED]
    assert not adapter.should_close


# ---- ING-NIMWS-003: conversation.created ------------------------------------


# @spec ING-NIMWS-003
def test_conversation_created_on_connect() -> None:
    adapter, _core, _fake, _recorded = make_adapter()
    (event,) = adapter.on_connect()
    assert event["type"] == "conversation.created"
    assert event["conversation"]["object"] == "realtime.conversation"
    assert event["conversation"]["id"].startswith("conv_")
    assert event["event_id"]


# ---- configure and admission -------------------------------------------------


# @spec ING-NIMWS-004, ING-CORE-004
async def test_first_update_is_configure_acked_updated_with_provenance() -> None:
    adapter, core, _fake, recorded = make_adapter()
    (reply,) = await configure(adapter)
    assert reply["type"] == UPDATED
    session = reply["session"]
    assert session["input_audio_format"] == "pcm16"
    # The admitted ack carries provenance on this dialect: the updated
    # event answers the configure, and there is no separate admitted
    # event to ride (ING-CORE-004).
    assert session["provenance"]["precision_policy_id"]
    assert recorded and recorded[0]["precision_policy_id"]
    assert not core.terminal


# @spec ING-LIFE-001
async def test_first_message_not_update_is_protocol_order_and_close() -> None:
    adapter, _core, _fake, _recorded = make_adapter()
    error = only_error(await adapter.on_event(append_event(pcm16(16)), 0.0))
    assert error["error"]["code"] == errors.PROTOCOL_ORDER
    assert adapter.should_close


# @spec ING-ADM-001
async def test_busy_admission_is_error_and_close() -> None:
    values = make_values(watermark=1)
    gate = make_gate(values)
    first, _core1, _fake1, _rec1 = make_adapter(values=values, gate=gate)
    assert types(await configure(first)) == [UPDATED]
    second, core2, _fake2, recorded2 = make_adapter(values=values, gate=gate)
    error = only_error(await configure(second))
    assert error["error"]["code"] == errors.BUSY
    assert second.should_close
    assert recorded2 == []  # a never-admitted session records nothing
    assert core2.terminal


# @spec ING-ADM-002, ING-ADM-005
async def test_queued_admission_defers_ack_and_replays_pre_roll_after_it() -> None:
    values = make_values(watermark=1, admission_queue=1)
    gate = make_gate(values)
    first, first_core, _fake1, _rec1 = make_adapter(values=values, gate=gate)
    assert types(await configure(first)) == [UPDATED]
    second, _core2, fake2, _rec2 = make_adapter(values=values, gate=gate)
    assert await configure(second, now=0.0) == []  # queued: the ack is deferred
    # Pre-roll: audio sent ahead of the outcome is held, not processed.
    assert await second.on_event(append_event(pcm16(CHUNK_SAMPLES)), 0.1) == []
    assert fake2.steps == []
    assert await second.poll(0.2) == []  # still waiting
    await first.on_event({"type": DONE}, 0.3)  # frees the slot
    events = await second.poll(0.4)
    assert types(events) == [UPDATED, DELTA]
    assert len(fake2.steps) == 1  # the held chunk replayed post-admission


# @spec ING-ADM-002, ING-ERR-002
async def test_admission_wait_timeout_is_its_own_code_and_closes() -> None:
    values = make_values(watermark=1, admission_queue=1, admission_wait_s=5.0)
    gate = make_gate(values)
    first, _core1, _fake1, _rec1 = make_adapter(values=values, gate=gate)
    assert types(await configure(first)) == [UPDATED]
    second, _core2, _fake2, _rec2 = make_adapter(values=values, gate=gate)
    assert await configure(second, now=0.0) == []
    error = only_error(await second.poll(6.0))
    assert error["error"]["code"] == errors.ADMISSION_WAIT_TIMEOUT
    assert second.should_close


# @spec ING-ADM-005
async def test_pre_roll_overflow_is_fatal_buffer_overflow() -> None:
    values = make_values(watermark=1, admission_queue=1, pre_roll_bytes=64)
    gate = make_gate(values)
    first, _core1, _fake1, _rec1 = make_adapter(values=values, gate=gate)
    assert types(await configure(first)) == [UPDATED]
    second, core2, fake2, recorded2 = make_adapter(values=values, gate=gate)
    assert await configure(second, now=0.0) == []
    error = only_error(await second.on_event(append_event(pcm16(256)), 0.1))
    assert error["error"]["code"] == errors.BUFFER_OVERFLOW
    assert second.should_close
    assert fake2.steps == []  # nothing half-processed
    assert recorded2 == []
    assert core2.terminal


# ---- ING-NIMWS-008: the disposition matrix ----------------------------------


# @spec ING-NIMWS-008
@pytest.mark.parametrize(
    ("overrides", "expected_code", "expected_field"),
    [
        (
            {
                "speaker_diarization": {
                    "enable_speaker_diarization": True,
                    "max_speaker_count": 2,
                }
            },
            errors.UNSUPPORTED_CAPABILITY,
            "speaker_diarization",
        ),
        (
            {
                "word_boosting": {
                    "enable_word_boosting": True,
                    "word_boosting_list": [{"phrases": ["hey"], "boost": 4.0}],
                }
            },
            errors.UNSUPPORTED_CAPABILITY,
            "word_boosting",
        ),
        (
            {"endpointing_config": {"start_history": 500}},
            errors.UNSUPPORTED_CAPABILITY,
            "endpointing_config",
        ),
        (
            {"recognition_config": {"enable_word_time_offsets": True}},
            errors.UNSUPPORTED_CAPABILITY,
            "enable_word_time_offsets",
        ),
        (
            {"recognition_config": {"enable_profanity_filter": True}},
            errors.INVALID_CONFIG_FIELD,
            "enable_profanity_filter",
        ),
        (
            {"recognition_config": {"max_alternatives": 2}},
            errors.INVALID_CONFIG_FIELD,
            "max_alternatives",
        ),
        (
            {"input_audio_transcription": {"prompt": "boost this"}},
            errors.UNSUPPORTED_CAPABILITY,
            "prompt",
        ),
        (
            {"modalities": ["text", "audio"]},
            errors.INVALID_CONFIG_FIELD,
            "modalities",
        ),
        (
            {"input_audio_transcription": {"model": "someone-elses-model"}},
            errors.INVALID_CONFIG_FIELD,
            "model",
        ),
        (
            {"custom_configuration": {"k": "v"}},
            errors.INVALID_CONFIG_FIELD,
            "custom_configuration",
        ),
    ],
)
def test_non_default_fields_are_rejected_naming_the_field(
    overrides: dict[str, Any], expected_code: str, expected_field: str
) -> None:
    values = make_values()
    session = session_dict(**overrides)
    (rejection,) = disposition_session(session, values, MODEL)
    assert rejection.code == expected_code
    assert expected_field in rejection.fields


# @spec ING-NIMWS-008
def test_punctuation_and_verbatim_are_model_intrinsic() -> None:
    values = make_values()
    for flag in (True, False):
        session = session_dict(
            recognition_config={
                "enable_automatic_punctuation": flag,
                "enable_verbatim_transcripts": flag,
            }
        )
        assert disposition_session(session, values, MODEL) == []


# @spec ING-NIMWS-008, ING-ERR-003
async def test_unknown_locale_rejection_and_pre_audio_retry() -> None:
    adapter, core, _fake, _recorded = make_adapter()
    error = only_error(
        await configure(adapter, input_audio_transcription={"language": "xx-XX"})
    )
    assert error["error"]["code"] == errors.UNKNOWN_LOCALE
    assert "language" in (error["error"]["param"] or "")
    # The catalog's NIM column: error, session continues pre-audio —
    # the corrected update then configures normally.
    assert not adapter.should_close
    assert not core.terminal
    assert types(await configure(adapter)) == [UPDATED]


# @spec ING-NIMWS-008
def test_missing_language_means_auto_never_fuzzy() -> None:
    values = make_values()
    session = session_dict()
    del session["input_audio_transcription"]
    assert disposition_session(session, values, MODEL) == []


# @spec ING-NIMWS-008
def test_serving_model_name_and_empty_model_are_honored() -> None:
    values = make_values()
    for model_value in (MODEL, ""):
        session = session_dict(
            input_audio_transcription={
                "language": "en-US",
                "model": model_value,
            }
        )
        assert disposition_session(session, values, MODEL) == []


# @spec ING-NIMWS-008, ING-ERR-001
def test_unknown_top_level_key_is_never_silently_ignored() -> None:
    values = make_values()
    (rejection,) = disposition_session(
        session_dict(frobnicate=1), values, MODEL
    )
    assert rejection.code == errors.INVALID_CONFIG_FIELD
    assert "frobnicate" in rejection.fields


# @spec ING-NIMWS-008
def test_echo_keys_pass_undispositioned() -> None:
    values = make_values()
    session = session_dict(
        id="sess_0000",
        object="realtime.transcription_session",
        client_secret=None,
    )
    assert disposition_session(session, values, MODEL) == []
    assert {"id", "object", "client_secret"} <= ECHO_KEYS


# @spec ING-NIMWS-008
def test_one_update_reports_every_rejection() -> None:
    values = make_values()
    session = session_dict(
        recognition_config={"max_alternatives": 3},
        endpointing_config={"stop_history": 800},
    )
    rejections = disposition_session(session, values, MODEL)
    assert {r.code for r in rejections} == {
        errors.INVALID_CONFIG_FIELD,
        errors.UNSUPPORTED_CAPABILITY,
    }
    named = {field for r in rejections for field in r.fields}
    assert {"max_alternatives", "endpointing_config"} <= named


# ---- ING-NIMWS-007: format validation ----------------------------------------


# @spec ING-NIMWS-007
@pytest.mark.parametrize(
    ("audio_format", "rate"),
    [
        ("pcm16", 16000),
        ("pcm16", 8000),
        ("g711_ulaw", 8000),
        ("g711_alaw", 8000),
    ],
)
async def test_accept_matrix_cells_admit(audio_format: str, rate: int) -> None:
    assert validate_nim_format(audio_format, rate, 1) is None
    adapter, _core, _fake, _recorded = make_adapter()
    replies = await configure(
        adapter,
        input_audio_format=audio_format,
        input_audio_params={"sample_rate_hz": rate, "num_channels": 1},
    )
    assert types(replies) == [UPDATED]


# @spec ING-NIMWS-007
@pytest.mark.parametrize(
    ("audio_format", "rate"),
    [
        ("g711_ulaw", 16000),
        ("g711_alaw", 16000),
        ("pcm16", 44100),
        ("mp3", 16000),
    ],
)
async def test_off_matrix_formats_reject_naming_both_fields(
    audio_format: str, rate: int
) -> None:
    rejection = validate_nim_format(audio_format, rate, 1)
    assert rejection is not None
    assert rejection.code == errors.UNSUPPORTED_FORMAT
    assert {"input_audio_format", "sample_rate_hz"} <= set(rejection.fields)
    adapter, _core, _fake, _recorded = make_adapter()
    error = only_error(
        await configure(
            adapter,
            input_audio_format=audio_format,
            input_audio_params={"sample_rate_hz": rate, "num_channels": 1},
        )
    )
    assert error["error"]["code"] == errors.UNSUPPORTED_FORMAT
    assert adapter.should_close


# @spec ING-NIMWS-007
def test_multichannel_is_rejected() -> None:
    rejection = validate_nim_format("pcm16", 16000, 2)
    assert rejection is not None
    assert rejection.code == errors.UNSUPPORTED_FORMAT
    assert "num_channels" in rejection.fields


# ---- audio flow and the delta projection -------------------------------------


# @spec ING-NIMWS-004, ING-CORE-005
async def test_appends_step_chunks_and_emit_deltas() -> None:
    adapter, _core, fake, _recorded = make_adapter()
    await configure(adapter)
    # Two chunks arrive split off sample alignment: framing never
    # shifts the steps the model sees.
    data = pcm16(2 * CHUNK_SAMPLES)
    first = await adapter.on_event(append_event(data[: CHUNK_BYTES + 3]), 1.0)
    second = await adapter.on_event(append_event(data[CHUNK_BYTES + 3 :]), 2.0)
    deltas = [e for e in first + second if e["type"] == DELTA]
    assert [d["delta"] for d in deltas] == ["hey", " there"]
    assert [len(step) for step in fake.steps] == [CHUNK_SAMPLES] * 2


# @spec ING-CORE-005
async def test_a_chunk_that_adds_nothing_emits_no_delta() -> None:
    adapter, _core, fake, _recorded = make_adapter()

    async def silent_step(chunk: Any) -> str:
        fake.steps.append(chunk)
        return "hey"  # cumulative never grows after the first chunk

    fake.step = silent_step  # type: ignore[method-assign]
    await configure(adapter)
    first = await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES)), 1.0)
    second = await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES)), 2.0)
    assert [e["delta"] for e in first if e["type"] == DELTA] == ["hey"]
    assert [e for e in second if e["type"] == DELTA] == []


# @spec ING-NIMWS-004
async def test_invalid_base64_rides_transcription_failed() -> None:
    adapter, _core, _fake, _recorded = make_adapter()
    await configure(adapter)
    # Ride-along fix (masked by NotImplementedError pre-implementation):
    # the original "not//valid==" payload was, in fact, valid base64 —
    # every character is in the alphabet and the padding is correct.
    (event,) = await adapter.on_event(
        {"type": APPEND, "audio": "!!!not-base64!!!"}, 1.0
    )
    assert event["type"] == FAILED
    assert event["error"]["code"] == errors.INVALID_AUDIO
    assert event["error"]["type"] == "transcription_error"


# @spec ING-NIMWS-004
async def test_unknown_event_type_is_an_error_never_silent() -> None:
    adapter, _core, _fake, _recorded = make_adapter()
    await configure(adapter)
    error = only_error(await adapter.on_event({"type": "response.create"}, 1.0))
    assert error["error"]["code"] == "invalid_event"
    assert error["error"]["type"] == "invalid_request_error"
    assert not adapter.should_close


# ---- ING-NIMWS-005: advisory commit ------------------------------------------


# @spec ING-NIMWS-005
async def test_commit_is_acked_and_never_forces_a_sub_chunk_step() -> None:
    adapter, _core, fake, _recorded = make_adapter()
    await configure(adapter)
    await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES // 2)), 1.0)
    (ack,) = await adapter.on_event({"type": COMMIT}, 1.1)
    assert ack["type"] == COMMITTED
    assert "item_id" in ack and "previous_item_id" in ack
    assert fake.steps == []  # the fixed-chunk invariant outranks the hint
    events = await adapter.on_event(
        append_event(pcm16(CHUNK_SAMPLES // 2, start=CHUNK_SAMPLES // 2)), 1.2
    )
    assert [e["type"] for e in events] == [DELTA]


# @spec ING-NIMWS-005
async def test_only_done_triggers_the_tail_flush() -> None:
    adapter, _core, fake, _recorded = make_adapter()
    await configure(adapter)
    await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES // 2)), 1.0)
    await adapter.on_event({"type": COMMIT}, 1.1)
    assert fake.flush_called is False
    events = await adapter.on_event({"type": DONE}, 1.2)
    assert fake.flush_called is True
    assert fake.flushed is not None and len(fake.flushed) == CHUNK_SAMPLES // 2
    (completed,) = events
    assert completed["type"] == COMPLETED
    assert completed["is_last_result"] is True


# ---- ING-NIMWS-006: clear ----------------------------------------------------


# @spec ING-NIMWS-006
async def test_clear_acks_cleared() -> None:
    adapter, _core, _fake, _recorded = make_adapter()
    await configure(adapter)
    (ack,) = await adapter.on_event({"type": CLEAR}, 1.0)
    assert ack["type"] == CLEARED
    assert ack["event_id"]


# @spec ING-NIMWS-006
async def test_clear_drops_only_the_unreleased_tail() -> None:
    adapter, _core, fake, _recorded = make_adapter()
    await configure(adapter)
    data = pcm16(CHUNK_SAMPLES + CHUNK_SAMPLES // 2)
    events = await adapter.on_event(append_event(data), 1.0)
    assert [e["type"] for e in events] == [DELTA]  # chunk 1 stepped
    await adapter.on_event({"type": CLEAR}, 1.1)
    (completed,) = await adapter.on_event({"type": DONE}, 1.2)
    # The un-released half-chunk tail was dropped: nothing to flush,
    # and the final is the stepped cumulative.
    assert fake.flush_called is False
    assert completed["type"] == COMPLETED
    assert completed["transcript"] == "hey"


# @spec ING-NIMWS-006
async def test_clear_never_rewinds_session_core_or_model_state() -> None:
    adapter, _core, fake, _recorded = make_adapter()
    await configure(adapter)
    await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES)), 1.0)
    await adapter.on_event({"type": CLEAR}, 1.1)
    assert fake.aborted is False
    assert len(fake.steps) == 1  # the stepped chunk stands
    events = await adapter.on_event(
        append_event(pcm16(CHUNK_SAMPLES, start=CHUNK_SAMPLES)), 1.2
    )
    assert [e["delta"] for e in events if e["type"] == DELTA] == [" there"]


# ---- finalize and lifecycle --------------------------------------------------


# @spec ING-NIMWS-004, ING-LIFE-002
async def test_done_finalizes_with_exactly_one_completed() -> None:
    adapter, _core, fake, _recorded = make_adapter()
    await configure(adapter)
    await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES + 160)), 1.0)
    (completed,) = await adapter.on_event({"type": DONE}, 2.0)
    assert completed["type"] == COMPLETED
    assert completed["is_last_result"] is True
    assert completed["transcript"] == "hey [flushed]"
    # Ride-along fix: `fake.flushed or ()` is ambiguous on a numpy
    # array (masked by NotImplementedError pre-implementation).
    assert fake.flushed is not None
    assert len(fake.flushed) == 160  # raw, un-padded residual


# @spec ING-LIFE-003
async def test_zero_audio_done_completes_empty_not_error() -> None:
    adapter, _core, fake, _recorded = make_adapter()
    await configure(adapter)
    (completed,) = await adapter.on_event({"type": DONE}, 1.0)
    assert completed["type"] == COMPLETED
    assert completed["transcript"] == ""
    assert completed["is_last_result"] is True
    assert fake.flush_called is False  # skip-flush on nothing


# @spec ING-ERR-004, ING-LIFE-002
async def test_events_after_done_answer_session_terminal() -> None:
    adapter, _core, _fake, _recorded = make_adapter()
    await configure(adapter)
    await adapter.on_event({"type": DONE}, 1.0)
    for event in (append_event(pcm16(16)), {"type": DONE}, {"type": COMMIT}):
        error = only_error(await adapter.on_event(event, 2.0))
        assert error["error"]["code"] == errors.SESSION_TERMINAL


# @spec ING-LIFE-004
async def test_detected_disconnect_aborts_and_frees_immediately() -> None:
    values = make_values(watermark=1)
    gate = make_gate(values)
    adapter, core, fake, _recorded = make_adapter(values=values, gate=gate)
    await configure(adapter)
    await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES // 2)), 1.0)
    await adapter.on_disconnect(2.0)
    assert fake.aborted is True
    assert core.terminal
    assert gate.active == 0  # the slot freed now, not at the idle TTL


# @spec ING-LIFE-006, ING-ERR-002
async def test_idle_timeout_surfaces_before_close() -> None:
    adapter, _core, fake, _recorded = make_adapter(
        values=make_values(idle_ttl_s=60.0)
    )
    await configure(adapter, now=0.0)
    error = only_error(await adapter.poll(61.0))
    assert error["error"]["code"] == errors.IDLE_TIMEOUT
    assert adapter.should_close  # the error reaches the wire, then close
    assert fake.aborted is True


# ---- mid-session updates -----------------------------------------------------


# @spec ING-LIFE-007
async def test_mid_session_language_update_is_honored() -> None:
    adapter, _core, fake, _recorded = await admitted_adapter()
    (ack,) = await adapter.on_event(
        update_event(input_audio_transcription={"language": "es-US"}), 1.0
    )
    assert ack["type"] == UPDATED
    assert fake.locales == ["es-US"]


# @spec ING-LIFE-007
async def test_mid_session_unknown_locale_continues_at_prior() -> None:
    adapter, _core, fake, _recorded = await admitted_adapter()
    error = only_error(
        await adapter.on_event(
            update_event(input_audio_transcription={"language": "xx-XX"}), 1.0
        )
    )
    assert error["error"]["code"] == errors.UNKNOWN_LOCALE
    assert fake.locales == []
    assert not adapter.should_close
    events = await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES)), 2.0)
    assert [e["type"] for e in events] == [DELTA]  # the session continues


# @spec ING-LIFE-008
async def test_mid_session_format_change_is_rejected_with_continuation() -> None:
    adapter, _core, _fake, _recorded = await admitted_adapter()
    error = only_error(
        await adapter.on_event(
            update_event(
                input_audio_format="g711_ulaw",
                input_audio_params={"sample_rate_hz": 8000, "num_channels": 1},
            ),
            1.0,
        )
    )
    assert error["error"]["code"] == errors.CONFIG_CHANGE_REJECTED
    assert not adapter.should_close
    events = await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES)), 2.0)
    assert [e["type"] for e in events] == [DELTA]  # admitted config stands


# @spec ING-LIFE-009
async def test_mixed_update_rejections_precede_the_truthful_ack() -> None:
    adapter, _core, fake, _recorded = await admitted_adapter()
    events = await adapter.on_event(
        update_event(
            input_audio_format="g711_ulaw",
            input_audio_params={"sample_rate_hz": 8000, "num_channels": 1},
            input_audio_transcription={"language": "es-US"},
        ),
        1.0,
    )
    assert types(events)[-1] == UPDATED  # the ack is last
    assert "error" in types(events)[:-1]
    assert fake.locales == ["es-US"]  # exactly the honored subset applied


# @spec ING-LIFE-009
async def test_noop_update_still_gets_a_truthful_ack() -> None:
    adapter, _core, _fake, _recorded = await admitted_adapter()
    (ack,) = await adapter.on_event(update_event(), 1.0)
    assert ack["type"] == UPDATED


# ---- ING-FE-004: resampler provenance ----------------------------------------


# @spec ING-FE-004
async def test_8k_session_records_the_resampler_identifier() -> None:
    adapter, _core, _fake, recorded = make_adapter()
    await configure(
        adapter,
        input_audio_params={"sample_rate_hz": 8000, "num_channels": 1},
    )
    assert recorded[0]["resampler_identifier"] == RESAMPLER_ID


# @spec ING-FE-004
async def test_16k_session_records_no_resampler_identifier() -> None:
    adapter, _core, _fake, recorded = make_adapter()
    await configure(adapter)
    assert "resampler_identifier" not in recorded[0]


# ---- ING-NIMWS-009: the canonical client's "none" format ---------------------


# @spec ING-NIMWS-009
async def test_none_format_defers_to_the_riff_header() -> None:
    adapter, _core, fake, _recorded = make_adapter()
    replies = await configure(adapter, input_audio_format=DEFERRED_FORMAT)
    assert types(replies) == [UPDATED]  # admission does not wait for audio
    wav = pcm16_wav(2 * CHUNK_SAMPLES)
    # The header straddles the first two appends: no error while the
    # sniff needs more bytes, and header bytes never decode as samples.
    assert await adapter.on_event(append_event(wav[:10]), 1.0) == []
    events = await adapter.on_event(append_event(wav[10:]), 1.1)
    assert [e["type"] for e in events] == [DELTA, DELTA]
    assert [len(step) for step in fake.steps] == [CHUNK_SAMPLES] * 2
    expected = np.arange(CHUNK_SAMPLES, dtype=np.float32) / 32768.0
    np.testing.assert_allclose(fake.steps[0], expected)


# @spec ING-NIMWS-009
async def test_none_format_declared_params_are_constraints() -> None:
    adapter, _core, _fake, _recorded = make_adapter()
    await configure(adapter, input_audio_format=DEFERRED_FORMAT)
    # 16 kHz was declared (the dialect always carries params); an 8 kHz
    # header contradicts it.
    error = only_error(
        await adapter.on_event(append_event(pcm16_wav(CHUNK_SAMPLES, rate=8000)), 1.0)
    )
    assert error["error"]["code"] == errors.UNSUPPORTED_FORMAT
    assert adapter.should_close


# @spec ING-NIMWS-009
async def test_none_format_without_a_riff_header_is_rejected() -> None:
    adapter, _core, _fake, _recorded = make_adapter()
    await configure(adapter, input_audio_format=DEFERRED_FORMAT)
    error = only_error(
        await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES)), 1.0)
    )
    assert error["error"]["code"] == errors.UNSUPPORTED_FORMAT
    assert adapter.should_close


# @spec ING-NIMWS-009
async def test_declared_format_is_never_overridden_by_a_header() -> None:
    adapter, _core, fake, _recorded = make_adapter()
    await configure(adapter)  # pcm16 declared explicitly
    wav = pcm16_wav(CHUNK_SAMPLES - 22)  # header (44 B) + payload = 1 chunk
    events = await adapter.on_event(append_event(wav), 1.0)
    # The declared format wins: the WAV header decodes as audio bytes.
    assert [e["type"] for e in events] == [DELTA]
    assert len(fake.steps[0]) == CHUNK_SAMPLES


# ---- wire hygiene and the canonical rehearsal --------------------------------


# @spec ING-NIMWS-003, ING-NIMWS-004
async def test_every_server_event_carries_a_unique_event_id() -> None:
    adapter, _core, _fake, _recorded = make_adapter()
    events = adapter.on_connect()
    events += await configure(adapter)
    events += await adapter.on_event(append_event(pcm16(CHUNK_SAMPLES)), 1.0)
    events += await adapter.on_event({"type": COMMIT}, 1.1)
    events += await adapter.on_event({"type": DONE}, 2.0)
    ids = [e["event_id"] for e in events]
    assert all(i.startswith("event_") for i in ids)
    assert len(ids) == len(set(ids))
    assert json.dumps(events)  # every event is JSON-serializable


# @spec ING-NIMWS-001, ING-NIMWS-004, ING-NIMWS-005, ING-NIMWS-009
async def test_canonical_file_mode_conformance_flow() -> None:
    # The exact realtime_asr_client.py file-mode exchange with default
    # flags (@ 5a443b5): POST object echoed back with the client's
    # overrides (input_audio_format "none", header-derived params,
    # default recognition flags), raw WAV bytes appended
    # header-included with a commit per chunk, then done.
    values = make_values()
    adapter, _core, fake, _recorded = make_adapter(values=values)
    (created,) = adapter.on_connect()
    assert created["type"] == "conversation.created"

    session = default_session_object(MODEL, values)
    session["input_audio_format"] = "none"
    session["input_audio_params"] = {"sample_rate_hz": 16000, "num_channels": 1}
    session["input_audio_transcription"]["language"] = "en-US"
    session["recognition_config"] = {
        "max_alternatives": 1,
        "enable_automatic_punctuation": False,
        "enable_word_time_offsets": False,
        "enable_profanity_filter": False,
        "enable_verbatim_transcripts": False,
    }
    (ack,) = await adapter.on_event({"type": UPDATE, "session": session}, 0.0)
    assert ack["type"] == UPDATED
    assert ack["session"]  # the client reads response["session"]

    wav = pcm16_wav(3 * CHUNK_SAMPLES + 160)
    received: list[dict[str, Any]] = []
    step = 1600 * 2  # the client's --file-streaming-chunk in pcm16 bytes
    for start in range(0, len(wav), step):
        received += await adapter.on_event(
            append_event(wav[start : start + step]), 1.0
        )
        received += await adapter.on_event({"type": COMMIT}, 1.0)
    received += await adapter.on_event({"type": DONE}, 2.0)

    deltas = [e["delta"] for e in received if e["type"] == DELTA]
    assert "".join(deltas) == fake.cumulative()
    finals = [e for e in received if e["type"] == COMPLETED]
    assert len(finals) == 1
    assert finals[0]["is_last_result"] is True
    assert finals[0]["transcript"] == "hey there friend [flushed]"
    committed = [e for e in received if e["type"] == COMMITTED]
    assert len(committed) == -(-len(wav) // step)  # one ack per commit
    assert not adapter.should_close


# @spec ING-CORE-001
def test_adapter_is_sans_io() -> None:
    # A dialect codec, not a server: no sockets, no HTTP, no event
    # loop — the serving shell owns transport.
    source = inspect.getsource(nim_ws)
    for forbidden in ("websockets", "asyncio", "socket", "http"):
        assert f"import {forbidden}" not in source


# @spec ING-CORE-003
def test_unknown_admission_outcome_is_rejected_not_defaulted() -> None:
    # QUEUED is reserved for the offload phase: a v1 adapter must
    # refuse to project it as success.
    adapter, _core, _fake, _recorded = make_adapter()
    with pytest.raises(ValueError):
        adapter.project_admission(AdmissionOutcome.QUEUED, None)
