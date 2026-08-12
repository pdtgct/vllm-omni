# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OmniStreamingMetrics — production streaming Prometheus families.

PORT-OBS-001..010 (port-specs.md §Capture and observability) and
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

from dataclasses import dataclass, field
from typing import Any

from prometheus_client import Counter, Gauge, Histogram
from vllm.logger import init_logger

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.streaming_transport import ChunkReadyHandle, StreamingObserver

logger = init_logger(__name__)


@dataclass
class _SessionRecord:
    """The bounded per-session observer state (PORT-OBS-003 as amended).

    A record exists if and only if the session was opened: open-at-
    construction is authoritative, and a ready arriving without an open
    is ignored loudly rather than manufacturing implicit session state
    (review round 2026-07-28, F6/F7).
    """

    cadence_ms: str
    waiting: list[ChunkReadyHandle] = field(default_factory=list)
    inflight: ChunkReadyHandle | None = None


_cadence_labels = list(defs.STREAMING_CADENCE_LABELS)
_finished_labels = list(defs.STREAMING_FINISHED_LABELS)
_chunk_latency_labels = list(defs.STREAMING_CHUNK_LATENCY_LABELS)
_chunks_labels = list(defs.STREAMING_CHUNKS_LABELS)
_deadline_miss_labels = list(defs.STREAMING_DEADLINE_MISS_LABELS)
_backlog_labels = list(defs.STREAMING_BACKLOG_LABELS)
_overflow_labels = list(defs.STREAMING_OVERFLOW_LABELS)
_open_rejection_labels = list(defs.STREAMING_OPEN_REJECTION_LABELS)
_admission_rejection_labels = list(defs.STREAMING_ADMISSION_REJECTION_LABELS)
_persistent_state_slot_labels = list(defs.PERSISTENT_STATE_SLOT_LABELS)
_persistent_state_service_demand_labels = list(defs.PERSISTENT_STATE_SERVICE_DEMAND_LABELS)
_persistent_state_execution_claims_labels = list(defs.PERSISTENT_STATE_EXECUTION_CLAIMS_LABELS)
_persistent_state_admission_headroom_labels = list(defs.PERSISTENT_STATE_ADMISSION_HEADROOM_LABELS)
_persistent_state_admission_pending_labels = list(defs.PERSISTENT_STATE_ADMISSION_PENDING_LABELS)
_persistent_state_admission_wait_labels = list(defs.PERSISTENT_STATE_ADMISSION_WAIT_LABELS)
_input_audio_labels = list(defs.STREAMING_INPUT_AUDIO_LABELS)
_batch_size_labels = list(defs.STREAMING_BATCH_SIZE_LABELS)


# ----------------------------------------------------------------------------
# Family declarations (port-design.md §Production metric export, the
# "Exported families" table). Counters gain "_total" at exposition.
# ----------------------------------------------------------------------------
_sessions_active_family = Gauge(
    defs.STREAMING_SESSIONS_ACTIVE,
    "Active model sessions after successful construction; manager resident "
    "inventory may additionally include preconstruction or cleanup leases.",
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
_admission_rejections_family = Counter(
    defs.STREAMING_ADMISSION_REJECTIONS,
    "Persistent-state admission denied by capacity or service availability.",
    labelnames=_admission_rejection_labels,
)
_persistent_state_slots_family = Gauge(
    defs.PERSISTENT_STATE_SLOTS,
    "Manager-sourced persistent-state resident inventory and capacity limits.",
    labelnames=_persistent_state_slot_labels,
)
_persistent_state_service_demand_family = Gauge(
    defs.PERSISTENT_STATE_SERVICE_DEMAND_RATIO,
    "Normalized persistent-state service budget and charged demand.",
    labelnames=_persistent_state_service_demand_labels,
)
_persistent_state_execution_claims_family = Gauge(
    defs.PERSISTENT_STATE_EXECUTION_CLAIMS,
    "Persistent-state execution claims and resolved scheduler ceiling.",
    labelnames=_persistent_state_execution_claims_labels,
)
_persistent_state_admission_headroom_family = Gauge(
    defs.PERSISTENT_STATE_ADMISSION_HEADROOM,
    "Additional hard and nominally dispatchable sessions by cadence.",
    labelnames=_persistent_state_admission_headroom_labels,
)
_persistent_state_admission_pending_family = Gauge(
    defs.PERSISTENT_STATE_ADMISSION_PENDING,
    "Bounded pre-audio and committed-cleanup admission state.",
    labelnames=_persistent_state_admission_pending_labels,
)
_persistent_state_admission_wait_family = Histogram(
    defs.PERSISTENT_STATE_ADMISSION_WAIT_S,
    "Pre-audio controller wait to admission or terminal disposition.",
    labelnames=_persistent_state_admission_wait_labels,
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

    def inc_admission_rejection(self, reason: str) -> None:
        """One definitive manager-backed admission denial."""
        if not self._log_stats:
            return
        if reason not in defs.STREAMING_ADMISSION_REJECTION_REASONS:
            return
        _admission_rejections_family.labels(
            model_name=self._model_name,
            reason=reason,
        ).inc()

    def observe_persistent_state_slots(
        self,
        stage: str,
        replica: str,
        inventory: dict[str, int],
    ) -> None:
        """Replace the complete manager-sourced slot projection."""
        if not self._log_stats:
            return
        if not stage or not replica:
            return
        if set(inventory) != set(defs.PERSISTENT_STATE_SLOT_KINDS):
            return
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in inventory.values()):
            return
        for kind in defs.PERSISTENT_STATE_SLOT_KINDS:
            _persistent_state_slots_family.labels(
                model_name=self._model_name,
                stage=stage,
                replica=replica,
                kind=kind,
            ).set(inventory[kind])

    # @spec PORT-OBS-012
    def observe_persistent_state_capacity(
        self,
        stage: str,
        replica: str,
        *,
        admission_policy: str,
        service_source: str | None,
        service_budget: float | None,
        charged_demand: float | None,
        execution_claims: int,
        max_num_seqs: int,
        headroom_by_cadence: dict[str, dict[str, int]],
        pending_by_cadence: dict[str, dict[str, int]],
    ) -> None:
        """Replace one consistent fixed-cardinality capacity projection."""
        if not self._log_stats:
            return
        if not stage or not replica:
            return
        if admission_policy not in {"profile", "hard_cap"}:
            return
        if admission_policy == "profile":
            if (
                service_source not in defs.PERSISTENT_STATE_SERVICE_DEMAND_SOURCES
                or service_budget is None
                or charged_demand is None
            ):
                return
            for kind, value in (
                ("budget", service_budget),
                ("charged_demand", charged_demand),
            ):
                _persistent_state_service_demand_family.labels(
                    model_name=self._model_name,
                    stage=stage,
                    replica=replica,
                    kind=kind,
                    source=service_source,
                ).set(value)
        for kind, value in (
            ("claims", execution_claims),
            ("max_num_seqs", max_num_seqs),
        ):
            _persistent_state_execution_claims_family.labels(
                model_name=self._model_name,
                stage=stage,
                replica=replica,
                kind=kind,
            ).set(value)
        for cadence_ms, headroom in headroom_by_cadence.items():
            if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
                continue
            expected_kinds = (
                set(defs.PERSISTENT_STATE_ADMISSION_HEADROOM_KINDS) if admission_policy == "profile" else {"hard"}
            )
            if set(headroom) != expected_kinds:
                continue
            for kind, value in headroom.items():
                _persistent_state_admission_headroom_family.labels(
                    model_name=self._model_name,
                    stage=stage,
                    replica=replica,
                    cadence_ms=cadence_ms,
                    kind=kind,
                ).set(value)
        for cadence_ms, pending in pending_by_cadence.items():
            if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
                continue
            for state in defs.PERSISTENT_STATE_ADMISSION_PENDING_STATES:
                _persistent_state_admission_pending_family.labels(
                    model_name=self._model_name,
                    stage=stage,
                    replica=replica,
                    cadence_ms=cadence_ms,
                    state=state,
                ).set(pending.get(state, 0))

    # @spec PORT-OBS-012
    def observe_persistent_state_admission_wait(
        self,
        stage: str,
        replica: str,
        *,
        cadence_ms: str,
        outcome: str,
        wait_s: float,
    ) -> None:
        """Observe one bounded pre-audio attempt's residence time."""
        if not self._log_stats:
            return
        if cadence_ms not in defs.STREAMING_CADENCE_MS_VALUES:
            return
        if outcome not in defs.PERSISTENT_STATE_ADMISSION_WAIT_OUTCOMES:
            return
        _persistent_state_admission_wait_family.labels(
            model_name=self._model_name,
            stage=stage,
            replica=replica,
            cadence_ms=cadence_ms,
            outcome=outcome,
        ).observe(max(wait_s, 0.0))

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
        _chunk_latency_family.labels(model_name=self._model_name, cadence_ms=cadence_ms, chunk_type=chunk_type).observe(
            max(latency_s, 0.0)
        )

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
        # One bounded record per session key (PORT-OBS-003 as amended):
        # {cadence, waiting-ready handles, single in-flight handle},
        # created at first open (or first ready, for observation-only
        # flows that never open), popped exactly once at terminal
        # finish. Disposition succeeds only while its handle is waiting
        # or in-flight in its session's record, which makes duplicate
        # and late dispositions natural no-ops WITHOUT a process-
        # lifetime disposed-handle set — no state outlives the record.
        self._sessions: dict[str, _SessionRecord] = {}

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

    def session_opened(self, *, session_key: str, cadence_ms: str) -> None:
        record = self._sessions.get(session_key)
        if record is None:
            self._sessions[session_key] = _SessionRecord(cadence_ms=cadence_ms)
            self._safe(self._metrics.inc_sessions_active, cadence_ms)
            return
        # Duplicate open of an active session: idempotent no-op; a
        # conflicting cadence is logged and ignored (PORT-OBS-003).
        if record.cadence_ms != cadence_ms:
            logger.warning(
                "duplicate session open with conflicting cadence (recorded=%s, ignored=%s); keeping the first",
                record.cadence_ms,
                cadence_ms,
            )

    def session_finished(self, *, session_key: str, reason: str) -> None:
        record = self._sessions.get(session_key)
        if record is None:
            # Unknown or duplicate finish: no-op.
            return
        # Divergence defense-in-depth: units still waiting/in-flight at
        # terminal finish are cleared as error before the record is
        # released, and the EFFECTIVE reason is normalized to error —
        # never the caller's reason alongside error chunk outcomes
        # (review round 2026-07-28, F4).
        remaining = self.clear_all_outstanding(session_key, outcome="error")
        if remaining:
            logger.warning(
                "session finished (caller reason=%s) with %d undisposed "
                "unit(s) — lifecycle divergence, cleared as error and the "
                "session finishes as error",
                reason,
                remaining,
            )
            reason = "error"
        # observe_session_finished is the single terminal section: it
        # decrements active and increments finished together.
        self._safe(self._metrics.observe_session_finished, record.cadence_ms, reason)
        del self._sessions[session_key]

    def session_record_count(self) -> int:
        """The number of live per-session records (bounded-state check)."""
        return len(self._sessions)

    def session_open_rejected(self, *, reason: str) -> None:
        self._safe(self._metrics.inc_open_rejection, reason)

    def accepted_audio_seconds(self, *, cadence_ms: str, seconds: float) -> None:
        self._safe(self._metrics.inc_input_audio_seconds, cadence_ms, seconds)

    def unit_ready(
        self,
        *,
        session_key: str,
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
        record = self._sessions.get(session_key)
        if record is None:
            # Open-at-construction is authoritative (PORT-OBS-003): a
            # ready without an open is ignored LOUDLY — no implicit
            # record, no backlog movement; the returned handle is inert
            # (all dispositions on it no-op via record membership).
            logger.warning(
                "unit_ready for un-opened session key — ignored; session open must precede ready (PORT-OBS-003)"
            )
            return handle
        record.waiting.append(handle)
        self._safe(self._metrics.inc_backlog, cadence_ms)
        return handle

    def unit_minted(self, handle: ChunkReadyHandle) -> None:
        """Promote a WAITING handle to the single in-flight slot.

        Strict transitions (review round 2026-07-28, F1): a duplicate
        mint of the current in-flight handle is idempotent; a mint of a
        handle that was never waiting (foreign or already disposed) is a
        no-op; a mint while a DIFFERENT unit is in flight is an illegal
        transition under PORT-SESS-001's one-in-flight invariant and
        no-ops with the handle left waiting — it can be minted legally
        once the in-flight unit reaches its terminal disposition.
        """
        record = self._sessions.get(handle.session_key)
        if record is None:
            return
        if record.inflight is handle:
            return
        if handle not in record.waiting:
            return
        if record.inflight is not None:
            return
        record.waiting.remove(handle)
        record.inflight = handle

    def complete_inflight(self, session_key: str) -> ChunkReadyHandle | None:
        """Resolve (without disposing) the single in-flight unit.

        The returned handle stays in the record until its terminal
        disposition removes it — disposal is the one place membership
        changes, which is what makes duplicate/late dispositions no-ops
        under the bounded-record design. A second resolve after the
        disposition (e.g. a carrierless park echo) returns ``None``.
        """
        record = self._sessions.get(session_key)
        if record is None:
            return None
        return record.inflight

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

    def clear_all_outstanding(self, session_key: str, *, outcome: str) -> int:
        """Clear every still-outstanding unit; returns the number cleared."""
        record = self._sessions.get(session_key)
        if record is None:
            return 0
        handles = list(record.waiting)
        if record.inflight is not None:
            handles.append(record.inflight)
        for handle in handles:
            self.unit_cleared(handle, outcome=outcome)
        return len(handles)

    def _dispose(self, handle: ChunkReadyHandle) -> bool:
        """Idempotent terminal disposition: decrements backlog exactly
        once per handle. Succeeds only while the handle is waiting or
        in-flight in its session's record (PORT-OBS-003 as amended) —
        duplicate and late dispositions, including any after the record
        was released, are full no-ops with no process-lifetime state."""
        record = self._sessions.get(handle.session_key)
        if record is None:
            return False
        if handle in record.waiting:
            record.waiting.remove(handle)
        elif record.inflight is handle:
            record.inflight = None
        else:
            return False
        self._safe(self._metrics.dec_backlog, handle.cadence_ms)
        return True
