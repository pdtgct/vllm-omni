"""The session-core error catalog (ingress-design.md §Error model).

One catalog of stable string codes, projected mechanically into each
dialect — no adapter invents an error, and no failure is silent. EARS
bind to these code strings; the projections mirror the LLD's dialect
columns (gRPC status name, NIM-dialect behavior, vLLM-dialect wire
code — three codes are inherited from the pin's realtime handler by
name).
"""

from collections.abc import Mapping
from dataclasses import dataclass

BUSY = "busy"
ADMISSION_WAIT_TIMEOUT = "admission_wait_timeout"
IDLE_TIMEOUT = "idle_timeout"
PROTOCOL_ORDER = "protocol_order"
INVALID_CONFIG_FIELD = "invalid_config_field"
UNSUPPORTED_CAPABILITY = "unsupported_capability"
UNKNOWN_LOCALE = "unknown_locale"
CONFIG_CHANGE_REJECTED = "config_change_rejected"
UNSUPPORTED_FORMAT = "unsupported_format"
INVALID_AUDIO = "invalid_audio"
BUFFER_OVERFLOW = "buffer_overflow"
SESSION_TERMINAL = "session_terminal"
INTERNAL = "internal"


@dataclass(frozen=True)
class DialectProjection:
    """How one catalog code surfaces on each dialect.

    ``grpc_status`` is the canonical gRPC status name; ``nim`` is the
    NIM-dialect behavior (``error``, ``error+close``, or
    ``transcription.failed``); ``vllm`` is the wire code on the vLLM
    realtime dialect (``None`` where the dialect cannot express the
    condition, e.g. ``unsupported_format`` on a fixed-pcm16 dialect).
    """

    grpc_status: str
    nim: str
    vllm: str | None


# Two codes are in-band on gRPC rather than stream-terminating:
# config_change_rejected continues the stream (its eventual status is
# OK), and session_terminal arrives while the stream is already
# closing (the finalize-acceptance precondition failed). Their status
# column carries the status the stream ends with.
_CATALOG: dict[str, DialectProjection] = {
    BUSY: DialectProjection("RESOURCE_EXHAUSTED", "error+close", "busy"),
    ADMISSION_WAIT_TIMEOUT: DialectProjection(
        "DEADLINE_EXCEEDED", "error+close", "admission_wait_timeout"
    ),
    IDLE_TIMEOUT: DialectProjection("ABORTED", "error+close", "idle_timeout"),
    PROTOCOL_ORDER: DialectProjection(
        "FAILED_PRECONDITION", "error+close", "model_not_validated"
    ),
    INVALID_CONFIG_FIELD: DialectProjection(
        "INVALID_ARGUMENT", "error", "invalid_config_field"
    ),
    UNSUPPORTED_CAPABILITY: DialectProjection(
        "UNIMPLEMENTED", "error", "unsupported_capability"
    ),
    UNKNOWN_LOCALE: DialectProjection(
        "INVALID_ARGUMENT", "error", "unknown_locale"
    ),
    CONFIG_CHANGE_REJECTED: DialectProjection(
        "OK", "error", "config_change_rejected"
    ),
    UNSUPPORTED_FORMAT: DialectProjection(
        "INVALID_ARGUMENT", "error+close", None
    ),
    INVALID_AUDIO: DialectProjection(
        "INVALID_ARGUMENT", "transcription.failed", "invalid_audio"
    ),
    BUFFER_OVERFLOW: DialectProjection(
        "RESOURCE_EXHAUSTED", "error+close", "buffer_overflow"
    ),
    SESSION_TERMINAL: DialectProjection(
        "FAILED_PRECONDITION", "error", "session_terminal"
    ),
    INTERNAL: DialectProjection(
        "INTERNAL", "transcription.failed", "processing_error"
    ),
}


# @spec ING-ERR-001, ING-ERR-002
def catalog() -> Mapping[str, DialectProjection]:
    """The full code -> projection table (ING-ERR-001/002)."""
    return _CATALOG
