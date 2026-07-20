"""Session-core event vocabulary (sans-IO, transport-agnostic).

The event set of ingress-design.md §The session-core seam: adapters
translate their dialect to and from exactly these events; behavior
lives in :mod:`nemotron_asr_ingress.core`, never in an adapter.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AdmissionOutcome(Enum):
    """Admission gate answers (ING-CORE-003).

    ``QUEUED`` is reserved for the offload phase (ledger A11): the
    enum carries it so the vocabulary is stable, but no v1 gate ever
    returns it and v1 adapters reject it rather than defaulting it to
    success.
    """

    ADMITTED = "admitted"
    BUSY = "busy"
    QUEUED = "queued"


# ---- inbound (client -> session core) ------------------------------------


@dataclass(frozen=True)
class Configure:
    """The session config; first event, answered by the admission outcome."""

    chunk_ms: int = 560
    target_lang: str = "auto"


@dataclass(frozen=True)
class Finalize:
    """Client-driven end of audio (vLLM ``commit {final:true}`` analog)."""


@dataclass(frozen=True)
class Close:
    """Disconnect/cleanup; also the mapping of a detected transport drop."""


# ---- outbound (session core -> client) -----------------------------------


@dataclass(frozen=True)
class Admitted:
    """Admission success; carries provenance for EVAL-ART recording."""

    session_id: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class Busy:
    """Admission watermark full (observable, never silent)."""

    detail: str = ""


@dataclass(frozen=True)
class Partial:
    """Cumulative hypothesis after one chunk step (ING-CORE-005)."""

    cumulative: str
    chunk_index: int


@dataclass(frozen=True)
class Final:
    """The finalized transcript (v1: exactly one per session)."""

    transcript: str


@dataclass(frozen=True)
class SessionError:
    """A typed error with a stable catalog code (ING-ERR-001/003)."""

    code: str
    fields: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class UpdateAck:
    """Mid-session update acknowledgment: exactly the honored subset."""

    honored: Mapping[str, Any] = field(default_factory=dict)


Event = Admitted | Busy | Partial | Final | SessionError | UpdateAck
