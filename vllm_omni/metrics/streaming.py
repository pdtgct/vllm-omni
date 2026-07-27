# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OmniStreamingMetrics — production streaming Prometheus families.

PORT-OBS-001..009 (port-specs.md §Capture and observability) and
port-design.md §Production metric export. Families follow the existing
metrics-module conventions (``prometheus.py``/``modality.py``): module-level
singletons registered once on the process default registry, imported only
from the streaming serving path so a deployment serving no streaming model
exposes no streaming families, and no unregister helper.

Every ``OmniStreamingMetrics`` observe method early-returns while
``log_stats`` is disabled (PORT-OBS-002), then defensively skips any
value outside its bounded label enumeration (PORT-OBS-001: unknown
reason/kind/outcome/cadence values are dropped, never promoted to a new
label) before touching a family. ``PrometheusStreamingObserver`` wraps
every one of those calls so an exporter failure can never propagate into
a feed/park/finalize call site (PORT-OBS-002/003) — observation is
non-authoritative and nonfatal by construction.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Gauge, Histogram
from vllm.logger import init_logger

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.streaming_transport import ChunkReadyHandle, StreamingObserver

logger = init_logger(__name__)

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


def _cadence_period_s(cadence_ms: str) -> float | None:
    """The admitted cadence period in seconds, or ``None`` if unbounded."""
    if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
        return None
    return float(cadence_ms) / 1000.0


class OmniStreamingMetrics:
    """Label-bound observe API for the production streaming families.

    One instance per serving app state (PORT-OBS-001/003). Every observe
    method early-returns while ``log_stats`` is disabled (PORT-OBS-002),
    then defensively skips any label value outside its bounded
    enumeration — never registering a new label value on a family
    (PORT-OBS-001).
    """

    def __init__(self, model_name: str, log_stats: bool = True) -> None:
        self._model_name = model_name
        self._log_stats = log_stats

    # ---- Session lifecycle (PORT-OBS-006) ----------------------------------

    def inc_sessions_active(self, cadence_ms: str) -> None:
        """One successful session open (opens - finished == active)."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
            return
        _sessions_active_family.labels(model_name=self._model_name, cadence_ms=cadence_ms).inc()

    def observe_session_finished(self, cadence_ms: str, reason: str) -> None:
        """The session's single idempotent terminal-disposition section:
        the active gauge decrements and the finished counter increments
        together, so opens - finished == active."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
            return
        if reason not in defs.STREAMING_FINISHED_REASONS:
            return
        _sessions_active_family.labels(model_name=self._model_name, cadence_ms=cadence_ms).dec()
        _sessions_finished_family.labels(model_name=self._model_name, cadence_ms=cadence_ms, reason=reason).inc()

    def inc_open_rejection(self, reason: str) -> None:
        """A denied session open — diagnostic only (PORT-OBS-007)."""
        if not self._log_stats:
            return
        if reason not in defs.STREAMING_OPEN_REJECTION_REASONS:
            return
        _open_rejections_family.labels(model_name=self._model_name, reason=reason).inc()

    def inc_input_audio_seconds(self, cadence_ms: str, seconds: float) -> None:
        """Accepted-audio seconds at the common PORT acceptance event."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
            return
        if seconds <= 0:
            return
        _input_audio_seconds_family.labels(model_name=self._model_name, cadence_ms=cadence_ms).inc(seconds)

    # ---- Chunk lifecycle (PORT-OBS-004/005) --------------------------------

    def observe_chunk_latency(self, cadence_ms: str, chunk_type: str, latency_s: float) -> None:
        """Ready-to-park latency for one parked unit only."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES or chunk_type not in defs.STREAMING_CHUNK_TYPES:
            return
        _chunk_latency_family.labels(
            model_name=self._model_name, cadence_ms=cadence_ms, chunk_type=chunk_type
        ).observe(max(latency_s, 0.0))

    def observe_chunk_outcome(self, cadence_ms: str, chunk_type: str, outcome: str) -> None:
        """One unit's terminal disposition (parked/aborted/error)."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES or chunk_type not in defs.STREAMING_CHUNK_TYPES:
            return
        if outcome not in defs.STREAMING_CHUNK_OUTCOMES:
            return
        _chunks_family.labels(
            model_name=self._model_name, cadence_ms=cadence_ms, chunk_type=chunk_type, outcome=outcome
        ).inc()

    def inc_deadline_miss(self, cadence_ms: str, chunk_type: str) -> None:
        """Observed latency strictly exceeded the admitted cadence period."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES or chunk_type not in defs.STREAMING_CHUNK_TYPES:
            return
        _deadline_misses_family.labels(model_name=self._model_name, cadence_ms=cadence_ms, chunk_type=chunk_type).inc()

    def inc_backlog(self, cadence_ms: str) -> None:
        """One unit crossed readiness (+1 at ready)."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
            return
        _backlog_family.labels(model_name=self._model_name, cadence_ms=cadence_ms).inc()

    def dec_backlog(self, cadence_ms: str, count: int = 1) -> None:
        """One or more units reached terminal disposition (-1 each)."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
            return
        if count <= 0:
            return
        _backlog_family.labels(model_name=self._model_name, cadence_ms=cadence_ms).dec(count)

    def inc_backlog_overflow(self, kind: str) -> None:
        """A bound trip on the accepted-audio queue or armed ledger."""
        if not self._log_stats:
            return
        if kind not in defs.STREAMING_OVERFLOW_KINDS:
            return
        _backlog_overflow_family.labels(model_name=self._model_name, kind=kind).inc()

    # ---- Engine sub-stat (PORT-OBS-008/009) --------------------------------

    def observe_chunk_batch_size(
        self,
        stage: str,
        replica: str,
        cadence_ms: str,
        rows: int,
    ) -> None:
        """One executed nonempty CHUNK geometry bucket's row count."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
            return
        _chunk_batch_size_family.labels(
            model_name=self._model_name, stage=stage, replica=replica, cadence_ms=cadence_ms
        ).observe(rows)


class PrometheusStreamingObserver(StreamingObserver):
    """The production ``StreamingObserver`` implementation.

    ``install_streaming_observer`` installs and returns one of these per
    app state. Adapts :class:`OmniStreamingMetrics` (Prometheus-specific)
    to the neutral, Prometheus-free ``StreamingObserver`` protocol
    (``vllm_omni.metrics.streaming_transport``, PORT-OBS-003's "Observer
    protocol home" decision) — statically typed against it (method
    signatures use ``ChunkReadyHandle``, not ``object``), which importing
    that neutral module allows (metrics -> neutral is a legal direction;
    only neutral -> metrics, or model -> metrics, would invert layering).
    The model package imports the SAME protocol from that neutral module,
    never from here, so this generic metrics package stays reusable by
    any future cache-aware streaming model (port-design.md §Production
    metric export: "the family is streaming-generic").

    Owns the per-``session_key`` {waiting-ready deque, single in-flight
    handle} state machine (PORT-OBS-003's "Park-correlation authority"
    decision): a park observation completes exactly the in-flight unit
    via :meth:`complete_inflight`; a park with no in-flight unit — a
    carrierless async echo or FLUSH — resolves to ``None`` and leaves
    every waiting-ready unit untouched. Every underlying metrics call is
    wrapped (:meth:`_safe`) so an exporter failure can never propagate
    into a feed/park/finalize call site (PORT-OBS-002/003).
    """

    def __init__(self, metrics: OmniStreamingMetrics) -> None:
        self._metrics = metrics
        self._waiting: dict[str, list[ChunkReadyHandle]] = {}
        self._inflight: dict[str, ChunkReadyHandle | None] = {}
        # Idempotency guard: a handle enters here on its FIRST terminal
        # disposition (park or clear) and every later disposition on the
        # same handle is a full no-op — backlog decrements exactly once
        # per unit even under a racing double-terminal call.
        self._disposed: set[ChunkReadyHandle] = set()

    @property
    def metrics(self) -> OmniStreamingMetrics:
        """The underlying :class:`OmniStreamingMetrics`.

        PORT-OBS-008/009: the orchestrator's batch-size sub-stat sink is
        engine-output-driven, not chunk-lifecycle-driven, so it dispatches
        straight to ``OmniStreamingMetrics.observe_chunk_batch_size``
        rather than through this observer's own chunk-lifecycle methods —
        this accessor lets a caller holding the one installed observer
        (``resolve_installed_observer``) reach it without a second,
        independently-configured ``OmniStreamingMetrics`` instance.
        """
        return self._metrics

    def _safe(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """Observation is non-authoritative and nonfatal: an exporter
        failure must never propagate into the caller (PORT-OBS-002)."""
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.exception("streaming observer sink failed in %s; observation is nonfatal", fn)

    def session_opened(self, *, cadence_ms: str) -> None:
        self._safe(self._metrics.inc_sessions_active, cadence_ms)

    def session_finished(self, *, cadence_ms: str, reason: str) -> None:
        self._safe(self._metrics.observe_session_finished, cadence_ms, reason)

    def session_open_rejected(self, *, reason: str) -> None:
        self._safe(self._metrics.inc_open_rejection, reason)

    def accepted_audio_seconds(self, *, cadence_ms: str, seconds: float) -> None:
        self._safe(self._metrics.inc_input_audio_seconds, cadence_ms, seconds)

    def unit_ready(
        self,
        *,
        session_key: str = "default",
        cadence_ms: str,
        chunk_type: str,
        ready_stamp_s: float,
    ) -> ChunkReadyHandle:
        handle = ChunkReadyHandle(
            session_key=session_key,
            cadence_ms=cadence_ms,
            chunk_type=chunk_type,
            ready_stamp_s=ready_stamp_s,
        )
        self._waiting.setdefault(session_key, []).append(handle)
        self._safe(self._metrics.inc_backlog, cadence_ms)
        return handle

    def unit_minted(self, handle: ChunkReadyHandle) -> None:
        waiting = self._waiting.get(handle.session_key)
        if waiting is not None and handle in waiting:
            waiting.remove(handle)
        self._inflight[handle.session_key] = handle

    def complete_inflight(self, session_key: str) -> ChunkReadyHandle | None:
        handle = self._inflight.get(session_key)
        self._inflight[session_key] = None
        return handle

    def unit_parked(self, handle: ChunkReadyHandle, *, park_stamp_s: float) -> None:
        if not self._dispose(handle):
            return
        latency_s = max(park_stamp_s - handle.ready_stamp_s, 0.0)
        self._safe(self._metrics.observe_chunk_latency, handle.cadence_ms, handle.chunk_type, latency_s)
        period_s = _cadence_period_s(handle.cadence_ms)
        if period_s is not None and latency_s > period_s:
            self._safe(self._metrics.inc_deadline_miss, handle.cadence_ms, handle.chunk_type)
        self._safe(self._metrics.observe_chunk_outcome, handle.cadence_ms, handle.chunk_type, "parked")

    def unit_cleared(self, handle: ChunkReadyHandle, *, outcome: str) -> None:
        if not self._dispose(handle):
            return
        self._safe(self._metrics.observe_chunk_outcome, handle.cadence_ms, handle.chunk_type, outcome)

    def overflow(self, *, kind: str) -> None:
        self._safe(self._metrics.inc_backlog_overflow, kind)

    def clear_all_outstanding(self, session_key: str, *, outcome: str) -> None:
        handles = list(self._waiting.get(session_key, ()))
        inflight = self._inflight.get(session_key)
        if inflight is not None:
            handles.append(inflight)
        for handle in handles:
            self.unit_cleared(handle, outcome=outcome)

    def _dispose(self, handle: ChunkReadyHandle) -> bool:
        """Idempotent terminal disposition: decrements backlog exactly
        once per handle. Returns ``False`` (a full no-op for the caller)
        when ``handle`` was already disposed."""
        waiting = self._waiting.get(handle.session_key)
        if waiting is not None and handle in waiting:
            waiting.remove(handle)
        if self._inflight.get(handle.session_key) is handle:
            self._inflight[handle.session_key] = None
        if handle in self._disposed:
            return False
        self._disposed.add(handle)
        self._safe(self._metrics.dec_backlog, handle.cadence_ms)
        return True
