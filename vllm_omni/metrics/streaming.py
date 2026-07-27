# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OmniStreamingMetrics — production streaming Prometheus families.

PORT-OBS-001..009 (port-specs.md §Capture and observability) and
port-design.md §Production metric export. Families follow the existing
metrics-module conventions (``prometheus.py``/``modality.py``): module-level
singletons registered once on the process default registry, imported only
from the streaming serving path so a deployment serving no streaming model
exposes no streaming families, and no unregister helper.

Phase-5 tests-first stub: the ten families below are REAL — they register on
import exactly as they will in production — but every ``OmniStreamingMetrics``
observe method unconditionally raises ``NotImplementedError`` (no ``log_stats``
gating logic yet; PORT-OBS-002's disabled-collection early-return and every
value-recording behavior land in a later phase). This keeps family-shape
tests (names, label sets, bucket ladders) green today while behavioral tests
(gating, values, model_name binding) stay red until implementation.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.streaming_transport import ChunkReadyHandle, StreamingObserver

_cadence_labels = list(defs.STREAMING_CADENCE_LABELS)
_finished_labels = list(defs.STREAMING_FINISHED_LABELS)
_chunk_latency_labels = list(defs.STREAMING_CHUNK_LATENCY_LABELS)
_chunks_labels = list(defs.STREAMING_CHUNKS_LABELS)
_deadline_miss_labels = list(defs.STREAMING_DEADLINE_MISS_LABELS)
_backlog_labels = list(defs.STREAMING_BACKLOG_LABELS)
_overflow_labels = list(defs.STREAMING_OVERFLOW_LABELS)
_open_rejection_labels = list(defs.STREAMING_OPEN_REJECTION_LABELS)
_input_audio_labels = list(defs.STREAMING_INPUT_AUDIO_LABELS)
_batch_size_labels = list(defs.STREAMING_BATCH_SIZE_LABELS)


# ----------------------------------------------------------------------------
# Family declarations (port-design.md §Production metric export, the
# "Exported families" table). Counters gain "_total" at exposition.
# ----------------------------------------------------------------------------
_sessions_active_family = Gauge(
    defs.STREAMING_SESSIONS_ACTIVE,
    "Active model sessions, native or leased — an upper bound on `resident` until provider admission exists.",
    labelnames=_cadence_labels,
)
_sessions_finished_family = Counter(
    defs.STREAMING_SESSIONS_FINISHED,
    "Model-session terminal outcomes, by reason.",
    labelnames=_finished_labels,
)
_chunk_latency_family = Histogram(
    defs.STREAMING_CHUNK_LATENCY_S,
    "Time from readiness to committed legal park, including queueing and "
    "execution (cadence completion for regular units, finalize acceptance "
    "for final-tail units).",
    labelnames=_chunk_latency_labels,
    buckets=defs.STREAMING_LATENCY_BUCKETS,
)
_chunks_family = Counter(
    defs.STREAMING_CHUNKS,
    "Ready units by terminal disposition; outcome=parked is the miss-rate denominator.",
    labelnames=_chunks_labels,
)
_deadline_misses_family = Counter(
    defs.STREAMING_DEADLINE_MISSES,
    "Chunk latency exceeded the admitted cadence period.",
    labelnames=_deadline_miss_labels,
)
_backlog_family = Gauge(
    defs.STREAMING_BACKLOG_CHUNKS,
    "Ready units without terminal disposition, summed over sessions.",
    labelnames=_backlog_labels,
)
_backlog_overflow_family = Counter(
    defs.STREAMING_BACKLOG_OVERFLOWS,
    "Per-session bound trips — the accepted-audio queue on any path; carrier/receipt ledger bounds on the leased path.",
    labelnames=_overflow_labels,
)
_open_rejections_family = Counter(
    defs.STREAMING_SESSION_OPEN_REJECTIONS,
    "Session opens denied for request/configuration causes — diagnostic only, never a scaling input.",
    labelnames=_open_rejection_labels,
)
_input_audio_seconds_family = Counter(
    defs.STREAMING_INPUT_AUDIO_SECONDS,
    "Accepted input audio duration — accepted samples / sample rate, accrued at the common PORT acceptance event.",
    labelnames=_input_audio_labels,
)
_chunk_batch_size_family = Histogram(
    defs.STREAMING_CHUNK_BATCH_SIZE,
    "Rows in each executed nonempty CHUNK geometry bucket.",
    labelnames=_batch_size_labels,
    buckets=defs.STREAMING_BATCH_SIZE_BUCKETS,
)


class OmniStreamingMetrics:
    """Label-bound observe API for the production streaming families.

    One instance per serving app state (PORT-OBS-001/003). Every observe
    method is unconditionally unimplemented in this tests-first phase —
    see the module docstring.
    """

    def __init__(self, model_name: str, log_stats: bool = True) -> None:
        self._model_name = model_name
        self._log_stats = log_stats

    # ---- Session lifecycle (PORT-OBS-006) ----------------------------------

    def inc_sessions_active(self, cadence_ms: str) -> None:
        """One successful session open (opens - finished == active)."""
        raise NotImplementedError

    def observe_session_finished(self, cadence_ms: str, reason: str) -> None:
        """The session's single idempotent terminal-disposition section."""
        raise NotImplementedError

    def inc_open_rejection(self, reason: str) -> None:
        """A denied session open — diagnostic only (PORT-OBS-007)."""
        raise NotImplementedError

    def inc_input_audio_seconds(self, cadence_ms: str, seconds: float) -> None:
        """Accepted-audio seconds at the common PORT acceptance event."""
        raise NotImplementedError

    # ---- Chunk lifecycle (PORT-OBS-004/005) --------------------------------

    def observe_chunk_latency(self, cadence_ms: str, chunk_type: str, latency_s: float) -> None:
        """Ready-to-park latency for one parked unit only."""
        raise NotImplementedError

    def observe_chunk_outcome(self, cadence_ms: str, chunk_type: str, outcome: str) -> None:
        """One unit's terminal disposition (parked/aborted/error)."""
        raise NotImplementedError

    def inc_deadline_miss(self, cadence_ms: str, chunk_type: str) -> None:
        """Observed latency strictly exceeded the admitted cadence period."""
        raise NotImplementedError

    def inc_backlog(self, cadence_ms: str) -> None:
        """One unit crossed readiness (+1 at ready)."""
        raise NotImplementedError

    def dec_backlog(self, cadence_ms: str, count: int = 1) -> None:
        """One or more units reached terminal disposition (-1 each)."""
        raise NotImplementedError

    def inc_backlog_overflow(self, kind: str) -> None:
        """A bound trip on the accepted-audio queue or armed ledger."""
        raise NotImplementedError

    # ---- Engine sub-stat (PORT-OBS-008/009) --------------------------------

    def observe_chunk_batch_size(
        self,
        stage: str,
        replica: str,
        cadence_ms: str,
        rows: int,
    ) -> None:
        """One executed nonempty CHUNK geometry bucket's row count."""
        raise NotImplementedError


class PrometheusStreamingObserver(StreamingObserver):
    """The production ``StreamingObserver`` implementation.

    ``install_streaming_observer`` installs and returns one of these per
    app state. Adapts :class:`OmniStreamingMetrics` (Prometheus-specific)
    to the neutral, Prometheus-free ``StreamingObserver`` protocol
    (``vllm_omni.metrics.streaming_transport``, PORT-OBS-003's "Observer
    protocol home" decision) — statically typed against it (method
    signatures use ``ChunkReadyHandle`, not ``object``), which importing
    that neutral module allows (metrics -> neutral is a legal direction;
    only neutral -> metrics, or model -> metrics, would invert layering).
    The model package imports the SAME protocol from that neutral module,
    never from here, so this generic metrics package stays reusable by
    any future cache-aware streaming model (port-design.md §Production
    metric export: "the family is streaming-generic").

    Phase-5 stub: every method is unconditionally unimplemented, matching
    ``OmniStreamingMetrics``' own stubs — see the module docstring.
    """

    def __init__(self, metrics: OmniStreamingMetrics) -> None:
        self._metrics = metrics

    def session_opened(self, *, cadence_ms: str) -> None:
        raise NotImplementedError

    def session_finished(self, *, cadence_ms: str, reason: str) -> None:
        raise NotImplementedError

    def session_open_rejected(self, *, reason: str) -> None:
        raise NotImplementedError

    def accepted_audio_seconds(self, *, cadence_ms: str, seconds: float) -> None:
        raise NotImplementedError

    def unit_ready(
        self,
        *,
        session_key: str,
        cadence_ms: str,
        chunk_type: str,
        ready_stamp_s: float,
    ) -> ChunkReadyHandle:
        raise NotImplementedError

    def unit_minted(self, handle: ChunkReadyHandle) -> None:
        raise NotImplementedError

    def complete_inflight(self, session_key: str) -> ChunkReadyHandle | None:
        raise NotImplementedError

    def unit_parked(self, handle: ChunkReadyHandle, *, park_stamp_s: float) -> None:
        raise NotImplementedError

    def unit_cleared(self, handle: ChunkReadyHandle, *, outcome: str) -> None:
        raise NotImplementedError

    def overflow(self, *, kind: str) -> None:
        raise NotImplementedError
