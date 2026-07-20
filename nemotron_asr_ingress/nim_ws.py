"""The NIM-realtime WebSocket adapter: the NIM dialect over the session core.

A sans-IO dialect codec (ING-CORE-001): NIM Realtime JSON events in,
session-core events through, NIM Realtime JSON events out — no session
behavior of its own. Wire shapes are those of the public NIM Realtime
API reference (docs.nvidia.com/nim/speech — Realtime API Reference)
as driven by the canonical client, ``python-clients``
``riva/client/realtime.py`` @ ``5a443b5`` — the ING-3 conformance
gate. Every server event carries an ``event_id``; errors are the
nested ``{"type", "code", "message", "param"}`` object of the
reference's error format.

Dialect mechanics honored from the reference and the canonical client:

- ``POST /v1/realtime/transcription_sessions`` returns the default
  transcription-session object (:func:`default_session_object`,
  ING-NIMWS-001); the canonical client copies that object and echoes
  it back — overrides applied — as its first
  ``transcription_session.update``, so the default object must
  round-trip through this module's own disposition unrejected.
- The WebSocket endpoint requires ``intent=transcription``; a missing
  or unsupported intent closes with WebSocket code 1008
  (:func:`intent_close_code`, ING-NIMWS-002).
- ``conversation.created`` is sent on connect, before any client
  event is processed (ING-NIMWS-003).
- The first ``transcription_session.update`` is the ``configure``:
  the session object is dispositioned field-for-field like the gRPC
  matrix (ING-NIMWS-008) *before* the session core sees it, so a
  pre-audio rejection leaves the session unconfigured and the client
  may retry (the catalog's NIM column: ``error``, session continues).
  Admission answers that update: ``admitted`` projects as the
  ``transcription_session.updated`` ack whose session object
  additively carries ``provenance`` (ING-CORE-004 on a dialect with
  no separate admitted event); ``busy`` is ``error`` + close.
- ``input_audio_buffer.commit`` is advisory: acked
  ``input_audio_buffer.committed``, never forcing a sub-chunk step
  (ING-NIMWS-005). ``input_audio_buffer.done`` maps to ``finalize``
  and is the only tail-flush trigger. ``input_audio_buffer.clear``
  drops only the adapter-held un-released buffer tail — the dialect's
  own ``input_audio_buffer`` object, audio not yet released to the
  session core — and never rewinds session-core or model state
  (ING-NIMWS-006, a documented deviation: mid-stream model state is
  not reversible).
- ``partial`` projects as ``conversation.item.input_audio_
  transcription.delta`` — the delta arithmetic from consecutive
  cumulative hypotheses is this adapter's work (ING-CORE-005), and a
  chunk that adds nothing emits no delta. ``final`` projects as
  ``…transcription.completed`` with ``is_last_result: true``.
  Transcription failures ride ``…transcription.failed``; protocol and
  session errors ride ``error`` (NIM's own split, per the catalog's
  NIM column; ``error+close`` codes flip :attr:`NimRealtimeAdapter.
  should_close`).
- ``input_audio_format`` and ``input_audio_params`` are validated
  jointly against the accept-matrix through :data:`NIM_FORMAT_NAMES`
  (ING-NIMWS-007; the g711 members are the OpenAI Realtime enum, a
  documented superset of NIM's pcm16-only). The canonical client in
  file mode declares ``input_audio_format: "none"`` and streams raw
  audio-file bytes header-included (``realtime.py`` non-mic path @
  ``5a443b5``), so ``"none"`` defers format resolution to a RIFF/WAVE
  header sniff at the head of the audio stream — header stripped,
  declared ``input_audio_params`` honored as constraints — mirroring
  ING-GRPC-007 (ING-NIMWS-009; decided 2026-07-13, a documented
  NIM-compatibility deviation never added to the honored enum).
"""

import base64
import binascii
import uuid
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import parse_qs

import numpy as np

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.core import (
    SAMPLE_RATE,
    FloatAudio,
    IngressValues,
    PreRollBuffer,
    SessionCore,
)
from nemotron_asr_ingress.events import (
    AdmissionOutcome,
    Admitted,
    Busy,
    Configure,
    Event,
    Final,
    Partial,
    SessionError,
    UpdateAck,
)
from nemotron_asr_ingress.frontend import (
    AudioFrontEnd,
    sniff_riff,
    validate_format,
)

#: NIM/OpenAI-Realtime audio-format names -> accept-matrix encodings
#: (ING-NIMWS-007). The four dialect cells {(pcm16, 16000),
#: (pcm16, 8000), (g711_ulaw, 8000), (g711_alaw, 8000)} map exactly
#: onto the front-end ACCEPT_MATRIX; G.711 is intrinsically 8 kHz, so
#: a 16 kHz g711 declaration is a client error (ING-FE-001).
NIM_FORMAT_NAMES: dict[str, str] = {
    "pcm16": "LINEAR_PCM",
    "g711_ulaw": "MULAW",
    "g711_alaw": "ALAW",
}

#: The canonical client's file-mode format declaration (realtime.py
#: non-mic path @ 5a443b5): no wire format, raw audio-file bytes
#: streamed header-included — format resolution defers to the RIFF
#: header sniff (ING-NIMWS-009, mirroring ING-GRPC-007).
DEFERRED_FORMAT = "none"

#: ``recognition_config`` field dispositions (ING-NIMWS-008): the
#: gRPC matrix rows under this dialect's JSON names — the parity is
#: contract-tested against ``riva_grpc.DISPOSITIONS`` through
#: :data:`GRPC_FIELD_NAMES`.
RECOGNITION_DISPOSITIONS: dict[str, str] = {
    "max_alternatives": "honored",
    "enable_automatic_punctuation": "model-intrinsic",
    "enable_word_time_offsets": "rejected-non-default",
    "enable_profanity_filter": "rejected-non-default",
    "enable_verbatim_transcripts": "model-intrinsic",
}

#: This dialect's ``recognition_config`` JSON names -> the gRPC
#: proto field names they mirror (the parity contract's join key).
GRPC_FIELD_NAMES: dict[str, str] = {
    "max_alternatives": "max_alternatives",
    "enable_automatic_punctuation": "enable_automatic_punctuation",
    "enable_word_time_offsets": "enable_word_time_offsets",
    "enable_profanity_filter": "profanity_filter",
    "enable_verbatim_transcripts": "verbatim_transcripts",
}

#: Session-object section dispositions (ING-NIMWS-008). A section is
#: non-default when its enable toggle is true (``speaker_diarization``,
#: ``word_boosting``), any window value is non-zero
#: (``endpointing_config``), or it is non-empty
#: (``custom_configuration``, ``prompt``) — the JSON analog of the
#: gRPC matrix's ``ByteSize() > 0``, chosen so the echoed default
#: session object always dispositions clean.
SECTION_DISPOSITIONS: dict[str, str] = {
    "speaker_diarization": "rejected-non-default",
    "word_boosting": "rejected-non-default",
    "endpointing_config": "rejected-non-default",
    "custom_configuration": "rejected-non-default",
}

#: Rejected capability *classes* answer ``unsupported_capability``;
#: rejected scalar toggles and out-of-range values answer
#: ``invalid_config_field`` — the ING-GRPC-005 two-status split under
#: this dialect's names. ``prompt`` is a proto-less NIM-dialect field
#: the LLD matrix predates (Whisper/Canary prompting): a capability
#: class, rejected when non-empty.
CAPABILITY_CLASS_FIELDS: frozenset[str] = frozenset(
    {
        "speaker_diarization",
        "word_boosting",
        "endpointing_config",
        "enable_word_time_offsets",
        "prompt",
    }
)

#: Structural session-object keys that are never dispositioned as
#: client config: server-assigned identity and dialect framing the
#: canonical client echoes back verbatim.
ECHO_KEYS: frozenset[str] = frozenset(
    {"id", "object", "client_secret", "provenance"}
)

#: Catalog codes whose NIM projection is ``error`` + connection close
#: (the catalog's NIM column, ING-ERR-001).
CLOSE_CODES: frozenset[str] = frozenset(
    code
    for code, projection in errors.catalog().items()
    if projection.nim == "error+close"
)

#: Catalog codes that ride ``conversation.item.input_audio_
#: transcription.failed`` rather than ``error`` (NIM's split between
#: transcription failures and protocol/session errors).
FAILED_CODES: frozenset[str] = frozenset(
    code
    for code, projection in errors.catalog().items()
    if projection.nim == "transcription.failed"
)

#: The dialect's wire event-type strings (the public NIM Realtime
#: reference's event names).
_UPDATE = "transcription_session.update"
_UPDATED = "transcription_session.updated"
_APPEND = "input_audio_buffer.append"
_COMMIT = "input_audio_buffer.commit"
_COMMITTED = "input_audio_buffer.committed"
_CLEAR = "input_audio_buffer.clear"
_CLEARED = "input_audio_buffer.cleared"
_DONE = "input_audio_buffer.done"
_DELTA = "conversation.item.input_audio_transcription.delta"
_COMPLETED = "conversation.item.input_audio_transcription.completed"
_FAILED = "conversation.item.input_audio_transcription.failed"

#: Accept-matrix field names -> this dialect's field names: rejection
#: payloads name dialect syntax (ING-ERR-003 on the dialect's terms).
_FIELD_RENAMES: dict[str, str] = {
    "encoding": "input_audio_format",
    "sample_rate_hz": "sample_rate_hz",
    "sample_rate_hertz": "sample_rate_hz",
    "channels": "num_channels",
}

#: How many head-of-stream bytes may accumulate while a deferred
#: format hunts for its RIFF data chunk before the hunt is called off
#: (ING-NIMWS-009; the same bound as the gRPC sniff).
_SNIFF_LIMIT_BYTES = 65536

#: The six endpointing window fields (public proto vocabulary).
_ENDPOINTING_KEYS = (
    "start_history",
    "start_threshold",
    "stop_history",
    "stop_threshold",
    "stop_history_eou",
    "stop_threshold_eou",
)


def _event_id() -> str:
    """One unique wire event id (every server event carries one)."""
    return f"event_{uuid.uuid4().hex}"


def _rejection(code: str, field_name: str, detail: str) -> SessionError:
    """One catalog-coded rejection naming its field (ING-ERR-003)."""
    return SessionError(code=code, fields=(field_name,), detail=detail)


# @spec ING-NIMWS-002
def intent_close_code(query: str) -> int | None:
    """Check a WebSocket connect query string's ``intent``.

    Returns ``None`` when the query carries ``intent=transcription``
    (the one supported intent), else WebSocket close code 1008
    (policy violation) — missing parameter included (ING-NIMWS-002).
    """
    params = parse_qs(query, keep_blank_values=True)
    if params.get("intent") == ["transcription"]:
        return None
    return 1008


# @spec ING-NIMWS-001
def default_session_object(
    model_name: str, values: IngressValues
) -> dict[str, Any]:
    """The default transcription-session object (ING-NIMWS-001).

    Shaped per the public NIM Realtime reference — ``id`` with the
    ``sess_`` prefix, ``object: "realtime.transcription_session"``,
    ``modalities: ["text"]``, and the six config sections — with this
    deployment's own identity: ``input_audio_transcription.model`` is
    the served model name, never a NIM model's. Every field carries a
    value this module's own disposition accepts, because the
    canonical client echoes the whole object back as its first
    ``transcription_session.update``.
    """
    language = "en-US" if "en-US" in values.locales else values.locales[0]
    return {
        "id": f"sess_{uuid.uuid4().hex}",
        "object": "realtime.transcription_session",
        "modalities": ["text"],
        "input_audio_format": "pcm16",
        "input_audio_transcription": {
            "language": language,
            "model": model_name,
            "prompt": "",
        },
        "input_audio_params": {"sample_rate_hz": 16000, "num_channels": 1},
        "recognition_config": {
            "max_alternatives": 1,
            "enable_automatic_punctuation": False,
            "enable_word_time_offsets": False,
            "enable_profanity_filter": False,
            "enable_verbatim_transcripts": False,
        },
        "speaker_diarization": {
            "enable_speaker_diarization": False,
            "max_speaker_count": 8,
        },
        "word_boosting": {
            "enable_word_boosting": False,
            "word_boosting_list": [],
        },
        "endpointing_config": {key: 0 for key in _ENDPOINTING_KEYS},
        "client_secret": None,
    }


# @spec ING-NIMWS-007
def validate_nim_format(
    input_audio_format: str, sample_rate_hz: int, num_channels: int
) -> SessionError | None:
    """Validate one declared (format, rate, channels) jointly.

    Maps the dialect format name through :data:`NIM_FORMAT_NAMES`
    onto the accept-matrix and defers to
    :func:`~nemotron_asr_ingress.frontend.validate_format`;
    an unknown format name is ``unsupported_format`` naming both
    fields (ING-NIMWS-007) — under this dialect's own field names
    (``input_audio_format``, ``sample_rate_hz``, ``num_channels``:
    field names are dialect syntax, so the codec translates them).
    ``"none"`` never reaches here — the deferred path resolves from
    the RIFF header instead (ING-NIMWS-009).
    """
    mapped = NIM_FORMAT_NAMES.get(input_audio_format, input_audio_format)
    rejection = validate_format(mapped, sample_rate_hz, num_channels)
    if rejection is None:
        return None
    return SessionError(
        code=rejection.code,
        fields=tuple(_FIELD_RENAMES[field] for field in rejection.fields),
        detail=(
            f"({input_audio_format}, {sample_rate_hz} Hz, {num_channels} ch)"
            " is outside the accepted set {(pcm16, 16000), (pcm16, 8000), "
            "(g711_ulaw, 8000), (g711_alaw, 8000)}, mono"
        ),
    )


def _section_non_default(name: str, section: Mapping[str, Any]) -> bool:
    """Whether one session section departs from its documented default.

    The JSON analog of the gRPC matrix's ``ByteSize() > 0``: the
    enable toggle for ``speaker_diarization``/``word_boosting``, any
    non-zero window for ``endpointing_config`` — chosen so the echoed
    default object (toggles false, windows zero) is always default.
    """
    if name == "speaker_diarization":
        return bool(section.get("enable_speaker_diarization"))
    if name == "word_boosting":
        return bool(section.get("enable_word_boosting")) or bool(
            section.get("word_boosting_list")
        )
    if name == "endpointing_config":
        return any(bool(value) for value in section.values())
    return bool(section)  # custom_configuration: non-empty


def _section_rejection(name: str) -> SessionError:
    """The two-status split for one non-default section."""
    if name in CAPABILITY_CLASS_FIELDS:
        return _rejection(
            errors.UNSUPPORTED_CAPABILITY,
            name,
            f"{name} is not a v1 capability",
        )
    return _rejection(
        errors.INVALID_CONFIG_FIELD,
        name,
        f"{name} is rejected when non-default",
    )


def _dispose_transcription(
    section: Mapping[str, Any], values: IngressValues, model_name: str
) -> list[SessionError]:
    """Disposition the ``input_audio_transcription`` section."""
    rejections: list[SessionError] = []
    for key, value in section.items():
        if key == "language":
            locale = value or "auto"
            if locale not in values.locales:
                rejections.append(
                    _rejection(
                        errors.UNKNOWN_LOCALE,
                        "language",
                        f"unknown locale {value!r}",
                    )
                )
        elif key == "model":
            if value and value != model_name:
                rejections.append(
                    _rejection(
                        errors.INVALID_CONFIG_FIELD,
                        "model",
                        f"model {value!r} is not served (serving {model_name})",
                    )
                )
        elif key == "prompt":
            if value:
                rejections.append(
                    _rejection(
                        errors.UNSUPPORTED_CAPABILITY,
                        "prompt",
                        "prompting is not a v1 capability",
                    )
                )
        else:
            rejections.append(
                _rejection(
                    errors.INVALID_CONFIG_FIELD,
                    key,
                    f"unknown input_audio_transcription field {key!r}",
                )
            )
    return rejections


def _dispose_recognition(
    section: Mapping[str, Any],
) -> list[SessionError]:
    """Disposition the ``recognition_config`` section per the matrix."""
    rejections: list[SessionError] = []
    for key, value in section.items():
        kind = RECOGNITION_DISPOSITIONS.get(key)
        if kind is None:
            rejections.append(
                _rejection(
                    errors.INVALID_CONFIG_FIELD,
                    key,
                    f"unknown recognition_config field {key!r}",
                )
            )
        elif kind == "model-intrinsic":
            continue  # accepted under either value, alters nothing
        elif key == "max_alternatives":
            if value > 1:
                rejections.append(
                    _rejection(
                        errors.INVALID_CONFIG_FIELD,
                        "max_alternatives",
                        "greedy decode has one hypothesis; "
                        "max_alternatives <= 1",
                    )
                )
        elif bool(value):
            if key in CAPABILITY_CLASS_FIELDS:
                rejections.append(
                    _rejection(
                        errors.UNSUPPORTED_CAPABILITY,
                        key,
                        f"{key} is not a v1 capability",
                    )
                )
            else:
                rejections.append(
                    _rejection(
                        errors.INVALID_CONFIG_FIELD,
                        key,
                        f"{key} is rejected when non-default",
                    )
                )
    return rejections


# @spec ING-NIMWS-008
def disposition_session(
    session: Mapping[str, Any],
    values: IngressValues,
    model_name: str,
) -> list[SessionError]:
    """Disposition one session object against the matrix.

    Returns the ordered rejection list (empty means admissible), one
    catalog-coded error naming its field(s) per violation
    (ING-ERR-003), field-for-field identical to the gRPC matrix
    (ING-NIMWS-008). :data:`ECHO_KEYS` pass undispositioned; unknown
    keys are ``invalid_config_field`` naming the key — never silently
    ignored (ING-ERR-001). Format validation is separate
    (:func:`validate_nim_format`) — the deferred ``"none"`` path has
    nothing to validate yet.
    """
    rejections: list[SessionError] = []
    for key, value in session.items():
        if key in ECHO_KEYS or key in (
            "input_audio_format",
            "input_audio_params",
        ):
            continue
        if key == "modalities":
            if list(value) != ["text"]:
                rejections.append(
                    _rejection(
                        errors.INVALID_CONFIG_FIELD,
                        "modalities",
                        'the one supported modality set is ["text"]',
                    )
                )
        elif key == "input_audio_transcription":
            rejections.extend(_dispose_transcription(value, values, model_name))
        elif key == "recognition_config":
            rejections.extend(_dispose_recognition(value))
        elif key in SECTION_DISPOSITIONS:
            if _section_non_default(key, value):
                rejections.append(_section_rejection(key))
        else:
            rejections.append(
                _rejection(
                    errors.INVALID_CONFIG_FIELD,
                    key,
                    f"unknown session field {key!r}",
                )
            )
    return rejections


# @spec ING-CORE-001, ING-NIMWS-004
class NimRealtimeAdapter:
    """One connection's NIM-dialect codec over one session core.

    Sans-IO like every adapter: JSON-shaped dicts in and out; the
    serving shell owns sockets, the session core owns behavior. The
    adapter owns dialect syntax only: the event mapping
    (ING-NIMWS-004), the delta projection from cumulative hypotheses
    (ING-CORE-005), the dialect's ``input_audio_buffer`` object —
    audio decodes through the front-end as it arrives, complete
    admitted-size chunks release to the core immediately, and the
    sub-chunk tail stays adapter-held so ``clear`` has something
    honest to drop (ING-NIMWS-006) — and the RIFF sniff for the
    canonical client's ``"none"`` format (ING-NIMWS-009).

    ``record_session`` receives each admitted session's effective
    provenance — including the conditional ``resampler_identifier``
    for 8 kHz sessions (ING-FE-004) — for EVAL-ART recording; a
    deferred-format session records once its format resolves.
    ``chunk_ms`` is the server-side admitted chunk config (the NIM
    dialect has no client chunk field; the value rides ENV like every
    other tunable).
    """

    def __init__(
        self,
        core: SessionCore,
        model_name: str,
        chunk_ms: int = 560,
        record_session: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        """Bind a session core, the served model's name, and seams."""
        self._core = core
        self._model_name = model_name
        self._chunk_ms = chunk_ms
        self._record_session = record_session
        self._pre_roll = PreRollBuffer(core.values.pre_roll_bytes)
        self._chunk_samples = chunk_ms * (SAMPLE_RATE // 1000)
        self._configured = False
        self._queued = False
        self._close = False
        self._recorded = False
        self._front: AudioFrontEnd | None = None
        self._deferred = False
        self._sniff_buf = b""
        self._declared_rate = 0
        self._declared_channels = 0
        self._pending_provenance: dict[str, Any] | None = None
        self._session_obj: dict[str, Any] = {}
        self._tail: FloatAudio = np.zeros(0, dtype=np.float32)
        self._last_cumulative = ""
        self._item_seq = 0
        self._item_id = "msg_0000"

    @property
    def terminal(self) -> bool:
        """Whether the session takes no further input (core-owned)."""
        return self._core.terminal

    @property
    def should_close(self) -> bool:
        """Whether an ``error+close`` projection ended the connection.

        Flips when a :data:`CLOSE_CODES` error was emitted; the shell
        sends every pending event first, then closes — the error
        always reaches the wire before the transport drops
        (ING-LIFE-006).
        """
        return self._close

    # @spec ING-NIMWS-003
    def on_connect(self) -> list[dict[str, Any]]:
        """Wire events sent on connect: ``conversation.created`` only.

        Sent before any client event is processed (ING-NIMWS-003).
        """
        return [
            {
                "event_id": _event_id(),
                "type": "conversation.created",
                "conversation": {
                    "id": f"conv_{uuid.uuid4().hex}",
                    "object": "realtime.conversation",
                },
            }
        ]

    # @spec ING-LIFE-004
    def on_disconnect(self, now: float) -> None:
        """A detected transport drop: immediate ``close``, no reply.

        Engine abort and slot free happen now, never left to the
        idle-TTL backstop; a queued attempt leaves the queue; every
        adapter-held buffer drops whole.
        """
        self._queued = False
        self._pre_roll.discard()
        self._tail = np.zeros(0, dtype=np.float32)
        self._sniff_buf = b""
        self._core.close(now)

    # @spec ING-NIMWS-004, ING-NIMWS-005, ING-NIMWS-006
    def on_event(
        self, event: Mapping[str, Any], now: float
    ) -> list[dict[str, Any]]:
        """Translate one client wire event; return wire events out.

        The mapping (ING-NIMWS-004): ``transcription_session.update``
        -> ``configure`` first / ``update`` later, acked
        ``transcription_session.updated``; ``input_audio_buffer.
        append`` -> front-end decode into the dialect buffer;
        ``…commit`` -> advisory ack only (ING-NIMWS-005); ``…clear``
        -> drop the adapter-held tail, ack ``…cleared``
        (ING-NIMWS-006); ``…done`` -> release the tail, ``finalize``.
        Any other first message is ``protocol_order`` (ING-LIFE-001);
        an unknown event type is an ``error`` event, never silent.
        """
        event_type = event.get("type")
        if event_type == _UPDATE:
            return self._on_update(event, now)
        if event_type == _APPEND:
            return self._on_append(event, now)
        if event_type == _COMMIT:
            return self._on_commit()
        if event_type == _CLEAR:
            return self._on_clear()
        if event_type == _DONE:
            return self._on_done(now)
        return [
            self._dialect_error(
                "invalid_event", f"unknown event type {event_type!r}"
            )
        ]

    def poll(self, now: float) -> list[dict[str, Any]]:
        """Surface timer-driven core events.

        Resolves a queued admission (the ``transcription_session.
        updated`` ack precedes every replayed pre-roll event, and a
        negative outcome discards the pre-roll whole, ING-ADM-005)
        and surfaces the wait/idle timeouts as their catalog errors.
        """
        core_events = self._core.poll(now)
        wire: list[dict[str, Any]] = []
        admitted = False
        for core_event in core_events:
            wire.extend(self.project(core_event))
            if isinstance(core_event, Admitted):
                admitted = True
        if self._queued and core_events:
            self._queued = False
            if admitted:
                wire.extend(self._drain_pre_roll(now))
            else:
                # Negative outcome: the whole pre-roll drops, nothing
                # half-processed (ING-ADM-005).
                self._pre_roll.discard()
        return wire

    # @spec ING-CORE-005, ING-ERR-001
    def project(self, event: Event) -> list[dict[str, Any]]:
        """Project one session-core stream event to wire events.

        ``partial`` -> ``…transcription.delta`` carrying the delta
        between consecutive cumulative hypotheses — an empty delta
        emits nothing (ING-CORE-005); ``final`` ->
        ``…transcription.completed`` with ``is_last_result: true``;
        ``update-ack`` -> ``transcription_session.updated``; errors
        ride ``…transcription.failed`` for :data:`FAILED_CODES`, the
        ``error`` event otherwise, with the catalog code in
        ``error.code``, the detail in ``error.message``, and the
        named fields in ``error.param`` (ING-ERR-001/003).
        """
        if isinstance(event, Admitted):
            provenance = dict(event.provenance)
            if self._deferred and self._front is None:
                # Record once the format resolves (ING-FE-004's
                # conditional resampler identity is unknowable yet).
                self._pending_provenance = provenance
            else:
                provenance = self._stamp_and_record(provenance)
            self._session_obj["provenance"] = provenance
            return [self._updated_event()]
        if isinstance(event, Busy):
            return [
                self._catalog_error(
                    SessionError(
                        code=errors.BUSY,
                        detail=event.detail or "admission watermark full",
                    )
                )
            ]
        if isinstance(event, Partial):
            delta = self._delta_from(event.cumulative)
            if not delta:
                # A chunk that added nothing (silence) sends nothing
                # (ING-CORE-005): an empty delta is wire noise.
                return []
            return [
                {
                    "event_id": _event_id(),
                    "type": _DELTA,
                    "item_id": self._item_id,
                    "content_index": 0,
                    "delta": delta,
                }
            ]
        if isinstance(event, Final):
            return [
                {
                    "event_id": _event_id(),
                    "type": _COMPLETED,
                    "item_id": self._item_id,
                    "content_index": 0,
                    "transcript": event.transcript,
                    "is_last_result": True,
                }
            ]
        if isinstance(event, UpdateAck):
            honored = dict(event.honored)
            if "target_lang" in honored:
                transcription = dict(
                    self._session_obj.get("input_audio_transcription") or {}
                )
                transcription["language"] = honored["target_lang"]
                self._session_obj["input_audio_transcription"] = transcription
            return [self._updated_event()]
        return [self._catalog_error(event)]

    # @spec ING-CORE-003
    def project_admission(
        self,
        outcome: AdmissionOutcome,
        provenance: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Project the admission outcome onto the wire.

        ``ADMITTED`` becomes the ``transcription_session.updated``
        ack (provenance riding the session object, ING-CORE-004);
        ``BUSY`` becomes the ``busy`` error + close.

        Raises:
            ValueError: For any outcome this v1 adapter does not know
                (``QUEUED`` included) — never defaulted to success
                (ING-CORE-003).
        """
        if outcome is AdmissionOutcome.ADMITTED:
            if provenance:
                self._session_obj["provenance"] = dict(provenance)
            return [self._updated_event()]
        if outcome is AdmissionOutcome.BUSY:
            return [
                self._catalog_error(
                    SessionError(
                        code=errors.BUSY, detail="admission watermark full"
                    )
                )
            ]
        raise ValueError(
            f"admission outcome {outcome!r} is not projectable by the "
            "v1 adapter"
        )

    # ---- client events -----------------------------------------------------

    def _on_update(
        self, event: Mapping[str, Any], now: float
    ) -> list[dict[str, Any]]:
        """First update is the configure; later ones are mid-session."""
        session = event.get("session") or {}
        if not self._configured:
            return self._on_configure(session, now)
        if self._queued:
            self._pre_roll.hold_update(dict(session))
            return []
        return self._on_mid_update(session, now)

    # @spec ING-ADM-001, ING-NIMWS-007, ING-NIMWS-008
    def _on_configure(
        self, session: Mapping[str, Any], now: float
    ) -> list[dict[str, Any]]:
        """Disposition, validate the format, then run admission.

        Rejections precede the gate — a rejected session object never
        takes a slot, stays unconfigured, and the client may retry
        (the catalog's NIM column: session continues pre-audio).
        """
        rejections = disposition_session(
            session, self._core.values, self._model_name
        )
        if rejections:
            return [self._catalog_error(r) for r in rejections]
        audio_format = str(session.get("input_audio_format", "pcm16"))
        params = session.get("input_audio_params") or {}
        rate = int(params.get("sample_rate_hz", 16000))
        channels = int(params.get("num_channels", 1))
        if audio_format == DEFERRED_FORMAT:
            # The canonical client's file mode (ING-NIMWS-009): the
            # RIFF header at the stream head resolves the format;
            # declared params become constraints it must match.
            self._deferred = True
            self._declared_rate = rate
            self._declared_channels = channels
        else:
            rejection = validate_nim_format(audio_format, rate, channels)
            if rejection is not None:
                return [self._catalog_error(rejection)]
            self._front = AudioFrontEnd(NIM_FORMAT_NAMES[audio_format], rate)
        self._session_obj = self._effective_session(
            session, audio_format, rate, channels
        )
        transcription = session.get("input_audio_transcription") or {}
        target_lang = str(transcription.get("language") or "auto")
        self._configured = True
        answers = self._core.configure(
            Configure(chunk_ms=self._chunk_ms, target_lang=target_lang), now
        )
        if not answers:
            self._queued = True
            return []
        return self._project_all(answers)

    # @spec ING-LIFE-007, ING-LIFE-008, ING-LIFE-009
    def _on_mid_update(
        self, session: Mapping[str, Any], now: float
    ) -> list[dict[str, Any]]:
        """A mid-session update: the adapter forwards only *changes*.

        The wire event carries a whole session object (the dialect's
        shape), so unchanged echoes are no-ops, a changed
        admission-fixed field answers one ``config_change_rejected``
        naming every changed field (ING-LIFE-008 — the dialect's
        format/params travel together as one client action), and a
        changed ``language`` forwards to the core (ING-LIFE-007).
        Rejections precede the truthful ack (ING-LIFE-009).
        """
        wire: list[dict[str, Any]] = []
        changed_fixed: list[str] = []
        audio_format = session.get("input_audio_format")
        if audio_format and audio_format != self._session_obj.get(
            "input_audio_format"
        ):
            changed_fixed.append("input_audio_format")
        params = session.get("input_audio_params") or {}
        current_params = self._session_obj.get("input_audio_params") or {}
        for wire_name in ("sample_rate_hz", "num_channels"):
            value = params.get(wire_name)
            if value and value != current_params.get(wire_name):
                changed_fixed.append(wire_name)
        if changed_fixed:
            wire.append(
                self._catalog_error(
                    SessionError(
                        code=errors.CONFIG_CHANGE_REJECTED,
                        fields=tuple(changed_fixed),
                        detail="the audio format is fixed at admission "
                        "(PORT-SESS-002); the session continues at the "
                        "admitted config",
                    )
                )
            )
        rejected_locale = False
        fields: dict[str, Any] = {}
        transcription = session.get("input_audio_transcription") or {}
        language = transcription.get("language")
        current_language = (
            self._session_obj.get("input_audio_transcription") or {}
        ).get("language")
        if language and language != current_language:
            # Validated exactly as at admission (ING-LIFE-007), named
            # in dialect vocabulary.
            if language not in self._core.values.locales:
                rejected_locale = True
                wire.append(
                    self._catalog_error(
                        _rejection(
                            errors.UNKNOWN_LOCALE,
                            "language",
                            f"unknown locale {language!r}; the session "
                            "continues at the prior locale",
                        )
                    )
                )
            else:
                fields["target_lang"] = language
        if fields or not (changed_fixed or rejected_locale):
            wire.extend(self._project_all(self._core.update(fields, now)))
        return wire

    # @spec ING-ADM-005, ING-NIMWS-009
    def _on_append(
        self, event: Mapping[str, Any], now: float
    ) -> list[dict[str, Any]]:
        """Decode one append through the front-end into the core."""
        if not self._configured:
            return [self._protocol_order("audio before the session config")]
        if self._core.terminal:
            return [self._session_terminal()]
        payload = event.get("audio")
        if not isinstance(payload, str):
            return [
                self._catalog_error(
                    SessionError(
                        code=errors.INVALID_AUDIO,
                        detail="audio payload must be a base64 string",
                    )
                )
            ]
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return [
                self._catalog_error(
                    SessionError(
                        code=errors.INVALID_AUDIO,
                        detail="audio payload is not base64",
                    )
                )
            ]
        if self._queued:
            if self._pre_roll.hold_audio(raw):
                return []
            # Pre-admission overflow: fatal to the attempt, nothing
            # processed, queue slot freed, retry-with-backoff safe.
            self._queued = False
            self._pre_roll.discard()
            self._core.close(now)
            return [
                self._catalog_error(
                    SessionError(
                        code=errors.BUFFER_OVERFLOW,
                        detail="pre-admission buffer bound exceeded; the "
                        "attempt is over (retry after backoff, or wait "
                        "for the admission answer before streaming)",
                    )
                )
            ]
        if self._deferred and self._front is None:
            return self._sniff(raw, now)
        return self._feed(raw, now)

    # @spec ING-NIMWS-005
    def _on_commit(self) -> list[dict[str, Any]]:
        """Advisory ack only — never a sub-chunk step (ING-NIMWS-005)."""
        if not self._configured:
            return [self._protocol_order("commit before the session config")]
        if self._core.terminal:
            return [self._session_terminal()]
        previous = self._item_id
        self._item_seq += 1
        self._item_id = f"msg_{self._item_seq:04d}"
        return [
            {
                "event_id": _event_id(),
                "type": _COMMITTED,
                "previous_item_id": previous,
                "item_id": self._item_id,
            }
        ]

    # @spec ING-NIMWS-006
    def _on_clear(self) -> list[dict[str, Any]]:
        """Drop the adapter-held un-released tail; never rewind.

        Only the dialect's own buffer object drops — the sub-chunk
        float tail and any unresolved sniff bytes. Stepped chunks,
        session-core state, and model state stand (the documented
        deviation: mid-stream model state is not reversible).
        """
        if not self._configured:
            return [self._protocol_order("clear before the session config")]
        if self._core.terminal:
            return [self._session_terminal()]
        self._tail = np.zeros(0, dtype=np.float32)
        self._sniff_buf = b""
        return [{"event_id": _event_id(), "type": _CLEARED}]

    # @spec ING-LIFE-002, ING-LIFE-003
    def _on_done(self, now: float) -> list[dict[str, Any]]:
        """Release the tail, flush, finalize: exactly one completed."""
        if not self._configured:
            return [self._protocol_order("done before the session config")]
        if self._core.terminal:
            return [self._session_terminal()]
        if self._queued:
            self._pre_roll.hold_finalize()
            return []
        return self._project_all(self._finalize_core(now))

    # ---- internals -----------------------------------------------------------

    def _finalize_core(self, now: float) -> list[Event]:
        """Release everything held, drain the front-end, finalize."""
        parts = [self._tail]
        self._tail = np.zeros(0, dtype=np.float32)
        if self._front is not None:
            parts.append(self._front.flush())
        tail_audio = np.concatenate(parts)
        events: list[Event] = []
        if len(tail_audio):
            events.extend(self._core.receive_audio(tail_audio, now))
        if self._pending_provenance is not None and not self._recorded:
            # A deferred session that never resolved a format records
            # resampler-less at finalize (gRPC parity, ING-FE-004).
            self._stamp_and_record(self._pending_provenance)
        events.extend(self._core.finalize(now))
        return events

    def _feed(self, raw: bytes, now: float) -> list[dict[str, Any]]:
        """Front-end decode; release complete chunks, hold the tail."""
        front = self._front
        if front is None:  # pragma: no cover — guarded by callers
            raise RuntimeError("audio fed before a format resolved")
        samples = front.feed(raw)
        if len(samples):
            self._tail = np.concatenate([self._tail, samples])
        n_ready = len(self._tail) // self._chunk_samples
        release = self._tail[: n_ready * self._chunk_samples]
        self._tail = self._tail[n_ready * self._chunk_samples :]
        # Always called — an all-tail append still touches the idle
        # clock (the wire receive happened, ING-LIFE-005).
        return self._project_all(self._core.receive_audio(release, now))

    # @spec ING-NIMWS-009
    def _sniff(self, raw: bytes, now: float) -> list[dict[str, Any]]:
        """Resolve a deferred format from the RIFF header, then feed."""
        self._sniff_buf += raw
        sniffed = sniff_riff(self._sniff_buf)
        if sniffed is None:
            if len(self._sniff_buf) > _SNIFF_LIMIT_BYTES:
                return [
                    self._catalog_error(
                        SessionError(
                            code=errors.UNSUPPORTED_FORMAT,
                            fields=("input_audio_format", "sample_rate_hz"),
                            detail="no RIFF/WAVE data chunk within the "
                            f"first {_SNIFF_LIMIT_BYTES} bytes of audio",
                        )
                    )
                ]
            return []  # the header needs more bytes
        if isinstance(sniffed, SessionError):
            return [self._catalog_error(self._rename_fields(sniffed))]
        if (
            self._declared_rate
            and self._declared_rate != sniffed.sample_rate_hz
        ):
            return [
                self._catalog_error(
                    SessionError(
                        code=errors.UNSUPPORTED_FORMAT,
                        fields=("sample_rate_hz",),
                        detail=f"declared sample_rate_hz {self._declared_rate}"
                        f" does not match the header's "
                        f"{sniffed.sample_rate_hz}",
                    )
                )
            ]
        if self._declared_channels and (
            self._declared_channels != sniffed.channels
        ):
            return [
                self._catalog_error(
                    SessionError(
                        code=errors.UNSUPPORTED_FORMAT,
                        fields=("num_channels",),
                        detail=f"declared num_channels "
                        f"{self._declared_channels} does not match the "
                        f"header's {sniffed.channels}",
                    )
                )
            ]
        rejection = validate_format(
            sniffed.encoding, sniffed.sample_rate_hz, sniffed.channels
        )
        if rejection is not None:
            return [self._catalog_error(self._rename_fields(rejection))]
        self._front = AudioFrontEnd(sniffed.encoding, sniffed.sample_rate_hz)
        if self._pending_provenance is not None:
            provenance = self._stamp_and_record(self._pending_provenance)
            self._session_obj["provenance"] = provenance
            self._pending_provenance = None
        data = self._sniff_buf[sniffed.data_offset :]
        self._sniff_buf = b""
        return self._feed(data, now)

    def _drain_pre_roll(self, now: float) -> list[dict[str, Any]]:
        """Replay everything held, in arrival order, post-admission."""
        wire: list[dict[str, Any]] = []
        for kind, payload in self._pre_roll.release():
            if kind == "audio" and isinstance(payload, bytes):
                if self._deferred and self._front is None:
                    wire.extend(self._sniff(payload, now))
                else:
                    wire.extend(self._feed(payload, now))
            elif kind == "update" and isinstance(payload, Mapping):
                wire.extend(self._on_mid_update(payload, now))
            elif kind == "finalize":
                wire.extend(self._project_all(self._finalize_core(now)))
        return wire

    def _effective_session(
        self,
        session: Mapping[str, Any],
        audio_format: str,
        rate: int,
        channels: int,
    ) -> dict[str, Any]:
        """The effective session object the ``updated`` ack echoes."""
        effective = default_session_object(self._model_name, self._core.values)
        effective["input_audio_format"] = audio_format
        effective["input_audio_params"] = {
            "sample_rate_hz": rate,
            "num_channels": channels,
        }
        transcription = dict(effective["input_audio_transcription"])
        requested = session.get("input_audio_transcription") or {}
        if requested.get("language"):
            transcription["language"] = requested["language"]
        if requested.get("model"):
            transcription["model"] = requested["model"]
        effective["input_audio_transcription"] = transcription
        if "recognition_config" in session:
            recognition = dict(effective["recognition_config"])
            recognition.update(session["recognition_config"])
            effective["recognition_config"] = recognition
        return effective

    def _stamp_and_record(self, provenance: dict[str, Any]) -> dict[str, Any]:
        """Record one session's effective provenance (EVAL-ART seam).

        The resampler identity joins the mapping for any resampled
        session (ING-FE-004); recording happens exactly once per
        admitted session.
        """
        if self._front is not None and self._front.resampler is not None:
            provenance["resampler_identifier"] = self._front.resampler
        if self._record_session is not None and not self._recorded:
            self._record_session(provenance)
            self._recorded = True
        return provenance

    def _updated_event(self) -> dict[str, Any]:
        """The ``transcription_session.updated`` ack (session rides it)."""
        return {
            "event_id": _event_id(),
            "type": _UPDATED,
            "session": dict(self._session_obj),
        }

    def _project_all(self, events: list[Event]) -> list[dict[str, Any]]:
        """Project a core event list in order."""
        return [wire for event in events for wire in self.project(event)]

    def _delta_from(self, cumulative: str) -> str:
        """Incremental delta between consecutive cumulative hypotheses."""
        if cumulative.startswith(self._last_cumulative):
            delta = cumulative[len(self._last_cumulative) :]
        else:
            delta = cumulative
        self._last_cumulative = cumulative
        return delta

    def _rename_fields(self, error: SessionError) -> SessionError:
        """Accept-matrix field names -> dialect names (ING-ERR-003)."""
        return SessionError(
            code=error.code,
            fields=tuple(
                _FIELD_RENAMES.get(field, field) for field in error.fields
            ),
            detail=error.detail,
        )

    def _catalog_error(self, error: SessionError) -> dict[str, Any]:
        """Project one catalog error per its NIM column (ING-ERR-001).

        :data:`FAILED_CODES` ride ``…transcription.failed`` with
        ``error.type: "transcription_error"``; everything else rides
        the ``error`` event; :data:`CLOSE_CODES` flip
        :attr:`should_close`.
        """
        if error.code in CLOSE_CODES:
            self._close = True
        failed = error.code in FAILED_CODES
        body = {
            "type": "transcription_error"
            if failed
            else "invalid_request_error",
            "code": error.code,
            "message": error.detail or error.code,
            "param": ", ".join(error.fields) if error.fields else None,
        }
        event: dict[str, Any] = {
            "event_id": _event_id(),
            "type": _FAILED if failed else "error",
            "error": body,
        }
        if failed:
            event["item_id"] = self._item_id
            event["content_index"] = 0
        return event

    def _dialect_error(self, code: str, message: str) -> dict[str, Any]:
        """A dialect-native protocol error (never a catalog code)."""
        return {
            "event_id": _event_id(),
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": code,
                "message": message,
                "param": None,
            },
        }

    def _protocol_order(self, detail: str) -> dict[str, Any]:
        """The ING-LIFE-001 answer (error+close per the catalog)."""
        return self._catalog_error(
            SessionError(code=errors.PROTOCOL_ORDER, detail=detail)
        )

    def _session_terminal(self) -> dict[str, Any]:
        """The one post-finalize answer (ING-ERR-004)."""
        return self._catalog_error(
            SessionError(
                code=errors.SESSION_TERMINAL,
                detail="the session is terminal; no further input is taken",
            )
        )
