"""The Riva gRPC servicer: `RivaSpeechRecognition` over the session core.

A dialect codec (ING-CORE-001): riva proto messages in, session-core
events through, proto responses out — no session behavior of its own.
Message types come from the public ``nvidia-riva-client`` package's
generated stubs (the canonical client and this servicer share one
proto artifact by construction), and registration uses the generated
``add_RivaSpeechRecognitionServicer_to_server`` — the servicer
duck-types the generated interface rather than subclassing it, so the
module stays fully typed under mypy strict.

The three RPCs (ING-GRPC-001):

- ``StreamingRecognize`` — first message must be ``streaming_config``
  (else ``protocol_order``/FAILED_PRECONDITION, ING-LIFE-001); audio
  bytes run through the front-end into the session core; each
  ``partial`` projects to an ``is_final=false`` response when
  ``interim_results`` is set (Riva interim results carry the
  cumulative transcript, so ING-CORE-005's cumulative hypothesis
  projects as-is — no delta arithmetic on this dialect); client
  half-close maps to ``finalize``, producing exactly one
  ``is_final=true`` response before the server closes.
- ``Recognize`` — a full-context single-shot (PORT-REGIME-001..003):
  the front-end decodes the whole payload, the offline seam
  transcribes once, one ``RecognizeResponse`` returns. Never a
  chunked streaming session (ING-GRPC-003).
- ``GetRivaSpeechRecognitionConfig`` — a static description derived
  from the checkpoint and the accept-matrix (ING-GRPC-004), carrying
  the ``enable_automatic_punctuation`` model-intrinsic deviation
  statement (ING-GRPC-006).

Every ``RecognitionConfig`` field is dispositioned per the LLD's
honest-subset matrix — honored, model-intrinsic, or
rejected-when-non-default naming the field; nothing is silently
ignored (ING-GRPC-005). The canonical ``python-clients`` tools declare
no format at all — encoding unset, raw audio-file bytes streamed
header-included — so an ``UNSPECIFIED`` encoding defers format
resolution to a RIFF/WAVE header sniff at the head of the audio
stream, header stripped and declared rate/channels honored as
constraints (ING-GRPC-007). Errors surface as catalog codes projected
through the catalog's gRPC status column (ING-ERR-001); the code
string rides the status ``details`` — the one text channel the
dialect has for the catalog's stable codes.
"""

import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal, NoReturn, Protocol, cast

import grpc
import numpy as np
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_audio_pb2 as raud

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.core import (
    FloatAudio,
    IngressValues,
    InProcessGate,
    SessionCore,
    Transcriber,
)
from nemotron_asr_ingress.events import (
    Admitted,
    Busy,
    Configure,
    Event,
    Final,
    Partial,
    SessionError,
)
from nemotron_asr_ingress.frontend import (
    ACCEPT_MATRIX,
    AudioFrontEnd,
    RiffFormat,
    sniff_riff,
    validate_format,
)

#: How many head-of-stream bytes may accumulate while a deferred
#: format hunts for its RIFF data chunk before the hunt is called off
#: (ING-GRPC-007) — generous against real WAV headers (tens of bytes,
#: occasionally a LIST/fact chunk), tiny against the chunk buffer.
_SNIFF_LIMIT_BYTES = 65536


def _sniff_limit_error() -> SessionError:
    """The ING-GRPC-007 rejection when no data chunk ever appears."""
    return SessionError(
        code=errors.UNSUPPORTED_FORMAT,
        fields=("encoding", "sample_rate_hertz"),
        detail=(
            "no declared format and no RIFF/WAVE data chunk within the "
            f"first {_SNIFF_LIMIT_BYTES} bytes of audio"
        ),
    )


#: Proto ``AudioEncoding`` numbers -> accept-matrix encoding names.
#: Absent numbers (FLAC, OGGOPUS, ENCODING_UNSPECIFIED) are outside
#: the matrix and answer ``unsupported_format`` (ING-FE-001).
ENCODING_NAMES: dict[int, str] = {
    raud.LINEAR_PCM: "LINEAR_PCM",
    raud.MULAW: "MULAW",
    raud.ALAW: "ALAW",
}

DispositionKind = Literal["honored", "model-intrinsic", "rejected-non-default"]

#: The honest-subset matrix (LLD §Surface 3), one row per
#: ``RecognitionConfig`` field in the wheel's generated proto — the
#: completeness contract test walks the descriptor, so a wheel bump
#: that adds a field breaks loudly instead of being silently ignored
#: (ING-GRPC-005). ``model`` and
#: ``enable_separate_recognition_per_channel`` are proto fields the
#: design-doc matrix predates: ``model`` is honored (empty or the one
#: served model name), per-channel separation is rejected non-default
#: (the matrix is mono, audio_channel_count == 1).
DISPOSITIONS: dict[str, DispositionKind] = {
    "encoding": "honored",
    "sample_rate_hertz": "honored",
    "language_code": "honored",
    "max_alternatives": "honored",
    "profanity_filter": "rejected-non-default",
    "speech_contexts": "rejected-non-default",
    "audio_channel_count": "honored",
    "enable_word_time_offsets": "rejected-non-default",
    "enable_automatic_punctuation": "model-intrinsic",
    "enable_separate_recognition_per_channel": "rejected-non-default",
    "model": "honored",
    # Reclassified from rejected-non-default after the round-1 pod
    # gate: the canonical python-clients tools send
    # verbatim_transcripts=true by default (transcribe_file.py @
    # 5a443b5), and rejecting a canonical default breaks exactly the
    # connectivity the honest-subset rule protects (ING-GRPC-006).
    "verbatim_transcripts": "model-intrinsic",
    "diarization_config": "rejected-non-default",
    "custom_configuration": "rejected-non-default",
    "endpointing_config": "rejected-non-default",
}

#: Rejected capability *classes* answer ``unsupported_capability``
#: (UNIMPLEMENTED); rejected scalar toggles and out-of-range values
#: answer ``invalid_config_field`` (INVALID_ARGUMENT). ING-GRPC-005's
#: two-status split, made explicit per field.
CAPABILITY_CLASS_FIELDS: frozenset[str] = frozenset(
    {
        "speech_contexts",
        "enable_word_time_offsets",
        "diarization_config",
        "endpointing_config",
    }
)


def _rejection(code: str, field_name: str, detail: str) -> SessionError:
    """One catalog-coded rejection naming its field (ING-ERR-003)."""
    return SessionError(code=code, fields=(field_name,), detail=detail)


# @spec ING-GRPC-005, ING-GRPC-006
def disposition(
    config: Any,
    values: IngressValues,
    model_name: str,
) -> list[SessionError]:
    """Disposition one ``RecognitionConfig`` against the matrix.

    Returns the ordered rejection list (empty means admissible), one
    catalog-coded :class:`SessionError` naming the offending field(s)
    per violation (ING-ERR-003):

    - (encoding, sample_rate_hertz) outside the accept-matrix ->
      ``unsupported_format`` naming both fields (ING-FE-001);
    - unknown ``language_code`` -> ``unknown_locale`` (empty string
      means ``auto`` — proto3's unset default, never fuzzy-matched);
    - honored-range violations (``max_alternatives`` > 1,
      ``audio_channel_count`` > 1, ``model`` naming anything but the
      one served model) -> ``invalid_config_field``;
    - non-default capability-class fields
      (:data:`CAPABILITY_CLASS_FIELDS`) -> ``unsupported_capability``;
    - other non-default rejected fields -> ``invalid_config_field``;
    - ``enable_automatic_punctuation`` -> never a rejection under
      either value (model-intrinsic, ING-GRPC-006).
    """
    rejections: list[SessionError] = []

    # An UNSPECIFIED encoding defers format resolution to the RIFF
    # header sniff at the head of the audio stream (ING-GRPC-007) —
    # the canonical python-clients shape; declared rate/channels act
    # as constraints the header must match at resolution time.
    if config.encoding != raud.ENCODING_UNSPECIFIED:
        encoding_name = ENCODING_NAMES.get(config.encoding)
        if (
            encoding_name is None
            or (encoding_name, config.sample_rate_hertz) not in ACCEPT_MATRIX
        ):
            label = encoding_name or raud.AudioEncoding.Name(config.encoding)
            rejections.append(
                SessionError(
                    code=errors.UNSUPPORTED_FORMAT,
                    fields=("encoding", "sample_rate_hertz"),
                    detail=(
                        f"({label}, {config.sample_rate_hertz} Hz) is "
                        "outside the accept-matrix {(LINEAR_PCM, 16000), "
                        "(LINEAR_PCM, 8000), (MULAW, 8000), (ALAW, 8000)}"
                    ),
                )
            )

    locale = config.language_code or "auto"
    if locale not in values.locales:
        rejections.append(
            _rejection(
                errors.UNKNOWN_LOCALE,
                "language_code",
                f"unknown locale {config.language_code!r}",
            )
        )

    if config.max_alternatives > 1:
        rejections.append(
            _rejection(
                errors.INVALID_CONFIG_FIELD,
                "max_alternatives",
                "greedy decode has one hypothesis; max_alternatives <= 1",
            )
        )
    if config.audio_channel_count > 1:
        rejections.append(
            _rejection(
                errors.INVALID_CONFIG_FIELD,
                "audio_channel_count",
                "the accept-matrix is mono; audio_channel_count <= 1",
            )
        )
    if config.model and config.model != model_name:
        rejections.append(
            _rejection(
                errors.INVALID_CONFIG_FIELD,
                "model",
                f"model {config.model!r} is not served (serving {model_name})",
            )
        )

    non_defaults: list[tuple[str, bool]] = [
        ("profanity_filter", config.profanity_filter),
        (
            "enable_separate_recognition_per_channel",
            config.enable_separate_recognition_per_channel,
        ),
        ("enable_word_time_offsets", config.enable_word_time_offsets),
        ("speech_contexts", len(config.speech_contexts) > 0),
        ("diarization_config", config.diarization_config.ByteSize() > 0),
        ("endpointing_config", config.endpointing_config.ByteSize() > 0),
        ("custom_configuration", len(config.custom_configuration) > 0),
    ]
    for field_name, is_non_default in non_defaults:
        if not is_non_default:
            continue
        if field_name in CAPABILITY_CLASS_FIELDS:
            code = errors.UNSUPPORTED_CAPABILITY
            detail = f"{field_name} is not a v1 capability"
        else:
            code = errors.INVALID_CONFIG_FIELD
            detail = f"{field_name} is rejected when non-default"
        rejections.append(_rejection(code, field_name, detail))

    return rejections


# @spec ING-GRPC-004, ING-GRPC-006
def build_config_response(values: IngressValues, model_name: str) -> Any:
    """The static ``GetRivaSpeechRecognitionConfig`` description.

    One ``model_config`` entry derived from the checkpoint and the
    accept-matrix: supported locales from the ``prompt_dictionary``
    (``values.locales``), sample rates {8000, 16000}, encodings
    {LINEAR_PCM, MULAW, ALAW}, ``streaming`` and ``offline`` both
    true, and an ``enable_automatic_punctuation`` parameter stating
    the model-intrinsic deviation (ING-GRPC-006).
    """
    response = rasr.RivaSpeechRecognitionConfigResponse()
    entry = response.model_config.add()
    entry.model_name = model_name
    entry.parameters["streaming"] = "true"
    entry.parameters["offline"] = "true"
    entry.parameters["supported_sample_rates"] = "8000,16000"
    entry.parameters["supported_encodings"] = "LINEAR_PCM,MULAW,ALAW"
    entry.parameters["language_code"] = ",".join(values.locales)
    entry.parameters["enable_automatic_punctuation"] = (
        "model-intrinsic: accepted under either value and alters nothing — "
        "punctuation and capitalization follow the checkpoint's own "
        "per-locale behavior"
    )
    entry.parameters["verbatim_transcripts"] = (
        "model-intrinsic: accepted under either value and alters nothing — "
        "normalization follows the checkpoint's own behavior"
    )
    return response


class OfflineTranscriber(Protocol):
    """The full-context single-shot seam ``Recognize`` drives.

    PORT owns the regime (PORT-REGIME-001..003); the servicer hands it
    the whole decoded 16 kHz clip exactly once — never a chunked
    session (ING-GRPC-003).
    """

    def transcribe(self, audio: FloatAudio, target_lang: str) -> str:
        """Transcribe one whole clip in the full-context regime."""
        ...


# @spec ING-GRPC-001, ING-CORE-001
class RivaAsrServicer:
    """The `RivaSpeechRecognition` servicer over the session core.

    Duck-types the generated servicer interface (the three RPC method
    names and signatures), so
    ``add_RivaSpeechRecognitionServicer_to_server`` registers it
    unchanged. Behavior lives in the session core: this class holds
    the shared gate, the values, and the compute seams, and translates
    dialect only. Time and sleep are injected values so the GPU-free
    tier drives every branch without waiting; ``record_session``
    receives each admitted session's effective provenance — including
    the conditional ``resampler_identifier`` for 8 kHz sessions
    (ING-FE-004) — for EVAL-ART recording.
    """

    def __init__(
        self,
        gate: InProcessGate,
        values: IngressValues,
        provenance: Mapping[str, Any],
        make_transcriber: Callable[[], Transcriber],
        offline: OfflineTranscriber,
        model_name: str,
        chunk_ms: int = 560,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_s: float = 0.05,
        record_session: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        """Bind the shared gate, the values, and the compute seams.

        ``chunk_ms`` is the server-side admitted chunk config for
        streaming sessions (the Riva dialect has no client chunk-size
        field; the value rides ENV like every other tunable);
        ``poll_s`` is the queued-admission poll cadence.
        """
        if poll_s <= 0:
            raise ValueError(f"poll_s must be positive, got {poll_s}")
        self._gate = gate
        self._values = values
        self._provenance = provenance
        self._make_transcriber = make_transcriber
        self._offline = offline
        self._model_name = model_name
        self._chunk_ms = chunk_ms
        self._clock = clock
        self._sleep = sleep
        self._poll_s = poll_s
        self._record_session = record_session

    # @spec ING-GRPC-002, ING-LIFE-001
    def StreamingRecognize(  # noqa: N802 — grpc method names are fixed
        self, request_iterator: Iterator[Any], context: Any
    ) -> Iterator[Any]:
        """One bidi streaming session over the session core.

        First message ``streaming_config`` or the stream aborts
        FAILED_PRECONDITION (``protocol_order``, ING-LIFE-001);
        admission answers it (``busy`` -> RESOURCE_EXHAUSTED, a
        queued wait past the bound -> DEADLINE_EXCEEDED); audio bytes
        decode through :class:`~nemotron_asr_ingress.frontend.
        AudioFrontEnd`; ``partial`` -> ``is_final=false`` responses
        gated on ``interim_results``; iterator exhaustion (client
        half-close) -> ``finalize`` -> exactly one ``is_final=true``
        response, then the server closes the stream (ING-GRPC-002).
        """
        first = next(request_iterator, None)
        if first is None or (
            first.WhichOneof("streaming_request") != "streaming_config"
        ):
            self._abort(
                context,
                errors.PROTOCOL_ORDER,
                "the first message must be streaming_config",
            )
        self._reject_runtime_config(first, context)
        streaming_config = first.streaming_config
        recognition = streaming_config.config
        self._dispose_or_abort(recognition, context)

        # An UNSPECIFIED encoding defers format resolution to the RIFF
        # header at the head of the audio stream (ING-GRPC-007, the
        # canonical python-clients shape); a declared one resolves now.
        front: AudioFrontEnd | None = None
        if recognition.encoding != raud.ENCODING_UNSPECIFIED:
            front = AudioFrontEnd(
                ENCODING_NAMES[recognition.encoding],
                recognition.sample_rate_hertz,
            )
        provenance = dict(self._provenance)
        recorded = False
        transcriber = self._make_transcriber()
        core = SessionCore(
            gate=self._gate,
            transcriber=transcriber,
            values=self._values,
            provenance=provenance,
        )
        if hasattr(transcriber, "core"):
            # The per-session assembly the β script does by hand
            # (serve_ws: ``lazy.core = session_core``): a lazy kernel
            # binding reads the admitted config from its core — the
            # SessionCore.config seam — and here the servicer is the
            # assembler.
            cast(Any, transcriber).core = core
        self._admit_or_abort(core, recognition, context)
        if front is not None:
            recorded = self._record(provenance, front)

        interim = streaming_config.interim_results
        sniff_buf = b""
        try:
            for request in request_iterator:
                self._reject_runtime_config(request, context)
                if request.WhichOneof("streaming_request") != "audio_content":
                    self._abort(
                        context,
                        errors.PROTOCOL_ORDER,
                        "the session is already configured",
                    )
                data = request.audio_content
                if front is None:
                    sniff_buf += data
                    sniffed = sniff_riff(sniff_buf)
                    if sniffed is None:
                        if len(sniff_buf) > _SNIFF_LIMIT_BYTES:
                            self._abort_error(context, _sniff_limit_error())
                        continue  # the header needs more bytes
                    if isinstance(sniffed, SessionError):
                        self._abort_error(context, sniffed)
                    front = self._resolve_deferred(
                        sniffed, recognition, context
                    )
                    recorded = self._record(provenance, front)
                    data = sniff_buf[sniffed.data_offset :]
                samples = front.feed(data)
                yield from self._project(
                    core.receive_audio(samples, self._clock()),
                    interim,
                    context,
                )
            # Client half-close: drain the front-end lookahead, then
            # finalize — exactly one is_final=true response. A deferred
            # session that never resolved a format (zero or too little
            # audio) finalizes empty (ING-LIFE-003).
            if front is not None:
                tail = front.flush()
                if len(tail):
                    yield from self._project(
                        core.receive_audio(tail, self._clock()),
                        interim,
                        context,
                    )
            if not recorded:
                self._record(provenance, front)
            yield from self._project(
                core.finalize(self._clock()), interim, context
            )
        finally:
            if not core.terminal:
                # An abort or a cancelled RPC is a detected end: free
                # the slot now, never leave it to the idle TTL
                # (ING-LIFE-004).
                core.close(self._clock())

    # @spec ING-GRPC-003, ING-GRPC-007
    def Recognize(  # noqa: N802 — grpc method names are fixed
        self, request: Any, context: Any
    ) -> Any:
        """One full-context single-shot; one ``RecognizeResponse``."""
        self._dispose_or_abort(request.config, context)
        payload = request.audio
        if request.config.encoding == raud.ENCODING_UNSPECIFIED:
            # The whole payload is at hand: an unresolvable head is
            # final here, never need-more-bytes (ING-GRPC-007).
            sniffed = sniff_riff(payload)
            if sniffed is None:
                self._abort_error(context, _sniff_limit_error())
            if isinstance(sniffed, SessionError):
                self._abort_error(context, sniffed)
            front = self._resolve_deferred(sniffed, request.config, context)
            payload = payload[sniffed.data_offset :]
        else:
            front = AudioFrontEnd(
                ENCODING_NAMES[request.config.encoding],
                request.config.sample_rate_hertz,
            )
        clip = np.concatenate([front.feed(payload), front.flush()])
        target_lang = request.config.language_code or "auto"
        transcript = self._offline.transcribe(clip, target_lang)
        response = rasr.RecognizeResponse()
        alternative = response.results.add().alternatives.add()
        alternative.transcript = transcript
        return response

    def _resolve_deferred(
        self, sniffed: RiffFormat, recognition: Any, context: Any
    ) -> AudioFrontEnd:
        """Turn a header resolution into a validated front-end.

        Declared ``sample_rate_hertz`` / ``audio_channel_count`` values
        are constraints the header must match (ING-GRPC-007); the
        resolved cell then validates against the accept-matrix exactly
        as a declared one would (ING-FE-001).
        """
        declared_rate = recognition.sample_rate_hertz
        if declared_rate and declared_rate != sniffed.sample_rate_hz:
            self._abort_error(
                context,
                SessionError(
                    code=errors.UNSUPPORTED_FORMAT,
                    fields=("sample_rate_hertz",),
                    detail=(
                        f"declared sample_rate_hertz {declared_rate} does "
                        f"not match the header's {sniffed.sample_rate_hz}"
                    ),
                ),
            )
        declared_channels = recognition.audio_channel_count
        if declared_channels and declared_channels != sniffed.channels:
            self._abort_error(
                context,
                SessionError(
                    code=errors.UNSUPPORTED_FORMAT,
                    fields=("audio_channel_count",),
                    detail=(
                        f"declared audio_channel_count {declared_channels} "
                        f"does not match the header's {sniffed.channels}"
                    ),
                ),
            )
        rejection = validate_format(
            sniffed.encoding, sniffed.sample_rate_hz, sniffed.channels
        )
        if rejection is not None:
            self._abort_error(context, rejection)
        return AudioFrontEnd(sniffed.encoding, sniffed.sample_rate_hz)

    def _record(
        self, provenance: dict[str, Any], front: AudioFrontEnd | None
    ) -> bool:
        """Record one session's effective provenance (EVAL-ART seam).

        The resampler identity joins the mapping for any resampled
        session (ING-FE-004); a deferred session records once its
        format resolves — or at finalize, resampler-less, when no
        audio ever resolved one.
        """
        if front is not None and front.resampler is not None:
            provenance["resampler_identifier"] = front.resampler
        if self._record_session is not None:
            self._record_session(provenance)
        return True

    # @spec ING-GRPC-004
    def GetRivaSpeechRecognitionConfig(  # noqa: N802 — grpc-fixed name
        self, request: Any, context: Any
    ) -> Any:
        """The static checkpoint/accept-matrix description."""
        return build_config_response(self._values, self._model_name)

    def _admit_or_abort(
        self, core: SessionCore, recognition: Any, context: Any
    ) -> None:
        """Run the admission handshake; abort on any negative outcome.

        A queued configure polls at ``poll_s`` through the injected
        sleep until the gate answers (ING-ADM-002) — the wait timeout
        surfaces as DEADLINE_EXCEEDED, ``busy`` as RESOURCE_EXHAUSTED.
        """
        config = Configure(
            chunk_ms=self._chunk_ms,
            target_lang=recognition.language_code or "auto",
        )
        answers = core.configure(config, self._clock())
        while not answers:
            self._sleep(self._poll_s)
            answers = core.poll(self._clock())
        for event in answers:
            if isinstance(event, Admitted):
                return
            if isinstance(event, Busy):
                self._abort(
                    context,
                    errors.BUSY,
                    event.detail or "admission watermark full",
                )
            if isinstance(event, SessionError):
                self._abort_error(context, event)
        raise AssertionError(f"unanswerable admission events: {answers}")

    def _project(
        self, events: list[Event], interim: bool, context: Any
    ) -> Iterator[Any]:
        """Project session-core stream events onto the wire."""
        for event in events:
            if isinstance(event, Partial):
                if interim:
                    yield self._response(event.cumulative, is_final=False)
            elif isinstance(event, Final):
                yield self._response(event.transcript, is_final=True)
            elif isinstance(event, SessionError):
                self._abort_error(context, event)

    @staticmethod
    def _response(transcript: str, is_final: bool) -> Any:
        """One ``StreamingRecognizeResponse`` (cumulative transcript)."""
        response = rasr.StreamingRecognizeResponse()
        result = response.results.add()
        result.is_final = is_final
        result.alternatives.add().transcript = transcript
        return response

    def _reject_runtime_config(self, request: Any, context: Any) -> None:
        """Reject a non-empty ``runtime_config`` rider, never ignore it.

        The request-level analog of the matrix's
        ``custom_configuration`` row (ING-ERR-001).
        """
        if request.runtime_config:
            self._abort_error(
                context,
                _rejection(
                    errors.INVALID_CONFIG_FIELD,
                    "runtime_config",
                    "runtime_config entries are rejected when non-empty",
                ),
            )

    def _dispose_or_abort(self, recognition: Any, context: Any) -> None:
        """Disposition the config; one abort names every rejection."""
        rejections = disposition(recognition, self._values, self._model_name)
        if not rejections:
            return
        first = rejections[0]
        projection = errors.catalog()[first.code]
        detail = "; ".join(self._detail(rejection) for rejection in rejections)
        context.abort(getattr(grpc.StatusCode, projection.grpc_status), detail)
        raise AssertionError("context.abort() must raise")

    def _abort(self, context: Any, code: str, detail: str) -> NoReturn:
        """Abort with a catalog code; the code string rides details."""
        self._abort_error(context, SessionError(code=code, detail=detail))

    def _abort_error(self, context: Any, error: SessionError) -> NoReturn:
        """Project one catalog error onto the RPC status (ING-ERR-001)."""
        projection = errors.catalog()[error.code]
        context.abort(
            getattr(grpc.StatusCode, projection.grpc_status),
            self._detail(error),
        )
        raise AssertionError("context.abort() must raise")

    @staticmethod
    def _detail(error: SessionError) -> str:
        """``code: detail (fields: ...)`` — stable code first."""
        detail = f"{error.code}: {error.detail or error.code}"
        if error.fields:
            detail += f" (fields: {', '.join(error.fields)})"
        return detail
