# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The neutral, Prometheus-free streaming-observability transport module.

Owns the contracts and observations that must NOT live in the model package or the Prometheus
metrics package (decisions/... "Observer protocol home", "Park-correlation
authority"; PORT-OBS-003):

- :class:`ChunkReadyHandle` and the :class:`StreamingObserver` protocol —
  the model package (``nemotron_asr/session.py``) imports these; the
  Prometheus adapter (``vllm_omni.metrics.streaming.PrometheusStreamingObserver``)
  is statically typed against them. A metrics-package-owned protocol would
  force the model package to import Prometheus; a model-package-owned
  protocol would force this metrics-generic capability to import the model.
  Neither direction is acceptable, so the protocol's home is this neutral
  module instead.
- The PORT-OBS-008/009 runner->scheduler batch-stat transport hops
  (:func:`drain_batch_stats_into_runner_output`,
  :func:`forward_batch_stats_to_engine_core_outputs`) — these run on the
  GPU worker/scheduler side, which must stay Prometheus-free.

This module deliberately imports NEITHER ``prometheus_client`` NOR
``vllm_omni.metrics.streaming`` (source-level control, enforced by
``test_streaming_transport_import_graph.py``'s source-scan half; the
sys.modules-delta half documents a pre-existing, unrelated fact about this
package — see that test's docstring).

Phase-5 tests-first stub: the two transport functions are unconditionally
unimplemented — see each docstring's ``Raises``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkReadyHandle:
    """Opaque per-unit identity carried through ready -> minted ->
    parked/cleared (PORT-OBS-003/004/005).

    Frozen values can compare equal for distinct accepted units. Lifecycle
    collaborators must compare handles by object identity (``is``), never
    by dataclass equality or readiness timestamps.

    Attributes:
        session_key: The owning session's identity (the native
            connection's per-generation request id, or the leased path's
            lease request id). The observer's internal per-session
            {waiting ready deque, current in-flight handle} state is keyed
            by this field — never reconstructed from the handle's other
            fields.
        cadence_ms: The admitted cadence label.
        chunk_type: ``"regular"`` or ``"final_tail"``.
        ready_stamp_s: The same-process monotonic ready stamp.
    """

    session_key: str
    cadence_ms: str
    chunk_type: str
    ready_stamp_s: float


@runtime_checkable
class StreamingObserver(Protocol):
    """Prometheus-free chunk/session observer protocol (PORT-OBS-003).

    Unit lifecycle: ready -> minted (promotion to in-flight at CHUNK
    submission; PORT-SESS-001 admits at most one in-flight unit per
    session) -> parked/cleared. The SINGLE IN-FLIGHT HANDLE is the
    park-correlation authority: :meth:`complete_inflight` resolves a park
    observation to exactly the in-flight unit for that session, or to
    ``None`` when there is none (a carrierless async park echo or FLUSH —
    ignored, and every waiting-ready unit for that session remains
    outstanding). A ready-order FIFO is deliberately NOT the authority:
    with several units ready before the first park, a FIFO would
    misclassify a carrierless echo as completing the next waiting unit.

    One observer is installed per serving app state and wired at the
    native connection adapter, the transport-neutral factory/lease, and
    the orchestrator batch-stat sink. Absent (``None``), every call site
    is a no-op: session construction and behavior are identical to the
    pre-metrics baseline.
    """

    def session_opened(self, *, session_key: str, cadence_ms: str) -> None:
        """One successful session open (PORT-OBS-006): opens the active
        gauge, keyed by the per-generation correlation key (the engine
        request id, minted once before session construction — PORT-OBS-003
        as amended). Native open is observer-bearing model-session
        construction after successful validation and before engine
        request creation — never WebSocket acceptance. A duplicate open
        of an active session is an idempotent no-op; a conflicting
        cadence is logged and ignored. The key is internal correlation
        state, never a metric label."""
        ...

    def session_finished(self, *, session_key: str, reason: str) -> None:
        """The session's single idempotent terminal-disposition section,
        keyed like :meth:`session_opened`: active decrements and finished
        increments together (cadence resolved from the record captured at
        open), so opens - finished == active. Unknown or duplicate finish
        is a no-op; terminal finish releases the session's entire bounded
        record, clearing any still-outstanding units as ``error``
        (lifecycle-divergence defense in depth)."""
        ...

    def session_open_rejected(self, *, reason: str) -> None:
        """A denied session open — diagnostic only, never a scaling
        signal (PORT-OBS-007)."""
        ...

    def accepted_audio_seconds(self, *, cadence_ms: str, seconds: float) -> None:
        """Accepted-audio seconds at the common PORT acceptance event."""
        ...

    def unit_ready(
        self,
        *,
        session_key: str,
        cadence_ms: str,
        chunk_type: str,
        ready_stamp_s: float,
    ) -> ChunkReadyHandle:
        """One regular/final-tail unit crossing readiness (PORT-OBS-004/005).
        Enters ``session_key``'s waiting-ready deque. ``ready_stamp_s`` for
        a final-tail unit is captured at finalize acceptance and passed in
        — never reconstructed at generator resumption."""
        ...

    def unit_minted(self, handle: ChunkReadyHandle) -> None:
        """Promote ``handle`` from waiting-ready to in-flight at CHUNK
        submission (PORT-SESS-001: at most one in-flight unit per
        session). Becomes the sole target :meth:`complete_inflight` may
        resolve for ``handle.session_key`` until its disposition."""
        ...

    def complete_inflight(self, session_key: str) -> ChunkReadyHandle | None:
        """Resolve a park observation against the single in-flight
        handle for ``session_key``.

        Returns:
            The in-flight handle (which the caller then disposes via
            :meth:`unit_parked`), or ``None`` when no unit is in-flight —
            the carrierless-echo/FLUSH case, correctly ignored with every
            waiting-ready unit left outstanding.
        """
        ...

    def unit_parked(self, handle: ChunkReadyHandle, *, park_stamp_s: float) -> None:
        """Committed legal park: the unit's terminal disposition."""
        ...

    def unit_cleared(self, handle: ChunkReadyHandle, *, outcome: str) -> None:
        """Client-initiated (``outcome="aborted"``) or named-failure
        (``outcome="error"``) clearing — the unit's other terminal
        disposition, mutually exclusive with ``unit_parked``."""
        ...

    def overflow(self, *, kind: str) -> None:
        """A bound trip on the accepted-audio queue (``kind="input_queue"``)
        or the armed receipt ledger (``kind="carrier"``/``"receipt"``)."""
        ...

    def clear_all_outstanding(self, session_key: str, *, outcome: str) -> int:
        """Terminal bulk clear: every unit still outstanding for
        ``session_key`` — waiting-ready AND the in-flight unit, if any —
        receives ``outcome`` as its terminal disposition in one pass,
        decrementing backlog by exactly the number of units cleared
        (PORT-OBS-005's decrement-by-exact-count). Returns that number,
        so a terminal caller can detect lifecycle divergence (an
        ostensibly clean end that still had outstanding work).

        Discovered during Phase-6 implementation: a session-terminal
        cleanup site (the lease's ledger-failure path, the native
        connection's end-of-generation cleanup) has no independent local
        record of every ready handle the observer minted — only the
        observer's own {waiting, in-flight} state does — so this is the
        single terminal-disposition entry point those call sites use
        instead of re-deriving handles themselves. Idempotent: a session
        with nothing outstanding is a no-op, and re-invoking after a
        first clear clears nothing further (every handle it would have
        touched is already disposed).
        """
        ...


@dataclass(slots=True)
class _ServiceTimingSlot:
    handle: ChunkReadyHandle | None = None
    logical_sequence: int | None = None
    carrier_sequence: int | None = None
    kind: str = ""
    r: float | None = None
    e_ns: int | None = None
    s_ns: int | None = None
    p: float | None = None
    disposition: str | None = None


# @spec PORT-OBS-002, PORT-OBS-003, PORT-OBS-004, PORT-PERF-007
class ServiceTimingTrace:
    """Session-owned diagnostic, mutated on the existing serving event loop.

    Slots are allocated at open. Handles are matched by object identity;
    authoritative logical/carrier sequences arrive at submission. r/p retain
    the observer's raw monotonic seconds; e/s retain existing monotonic ns
    until terminal serialization. Controls have no inferred readiness/park.
    This is observational state, never a dispatch or completion authority.
    """

    capacity = 256

    def __init__(self, session_key: str) -> None:
        self.session_key = session_key
        self.slots = [_ServiceTimingSlot() for _ in range(self.capacity)]
        self.count = 0
        self.overflow = 0
        self.valid = True
        self.next_sequence = 0
        self.engine_epoch: str | None = None
        self.lease_generation: int | None = None

    def input_identity(self, unit: Any, *, engine_epoch: str, lease_generation: int) -> str:
        """Carry producer identity; ``None`` identifies the unsequenced FLUSH."""
        if self.engine_epoch is not None and (self.engine_epoch, self.lease_generation) != (
            engine_epoch,
            lease_generation,
        ):
            self.valid = False
            raise ValueError("service timing generation changed within a session")
        self.engine_epoch = engine_epoch
        self.lease_generation = lease_generation
        return json.dumps(
            dict(
                session=self.session_key,
                engine_epoch=engine_epoch,
                lease_generation=lease_generation,
                logical_sequence=None if unit is None else unit.logical_sequence,
                carrier_sequence=None if unit is None else unit.carrier_sequence,
                kind="flush" if unit is None else unit.kind,
            ),
            separators=(",", ":"),
        )

    def _reserve(self) -> _ServiceTimingSlot | None:
        if self.count == self.capacity:
            self.overflow += 1
            self.valid = False
            return None
        slot = self.slots[self.count]
        self.count += 1
        return slot

    def _find(self, handle: ChunkReadyHandle) -> _ServiceTimingSlot | None:
        for index in range(self.count):
            slot = self.slots[index]
            if slot.handle is handle:
                return slot
        self.valid = False
        return None

    def ready(self, handle: ChunkReadyHandle) -> None:
        slot = self._reserve()
        if slot is not None:
            slot.handle = handle
            slot.kind = handle.chunk_type
            slot.r = handle.ready_stamp_s

    def submitted(
        self,
        handle: ChunkReadyHandle | None,
        logical_sequence: int,
        carrier_sequence: int | None,
        kind: str,
        eligible_ns: int | None,
        submitted_ns: int,
    ) -> None:
        if logical_sequence != self.next_sequence:
            self.valid = False
        self.next_sequence = logical_sequence + 1
        if kind == "forced_eou" and handle is None:
            slot = self._reserve()
        elif handle is not None:
            slot = self._find(handle)
        else:
            self.valid = False
            return
        if slot is None:
            return
        if slot.s_ns is not None or (slot.kind and slot.kind != kind):
            self.valid = False
            return
        slot.logical_sequence = logical_sequence
        slot.carrier_sequence = carrier_sequence
        slot.kind = kind
        slot.e_ns = eligible_ns
        slot.s_ns = submitted_ns

    def disposed(self, handle: ChunkReadyHandle, park_s: float | None, outcome: str) -> None:
        slot = self._find(handle)
        if slot is None:
            return
        if slot.disposition is not None:
            self.valid = False
            return
        slot.p = park_s
        slot.disposition = outcome

    def finish(self, reason: str) -> dict[str, Any]:
        """Serialize only at terminal cleanup, outside measured unit intervals.

        One caller-owned logger record includes count/capacity/overflow and
        an end marker: consumers must reject missing/truncated records and
        reconcile audio_count and sum(p-r) with independent histograms only
        for valid complete traces. count includes every bounded record,
        including controls; audio_count includes regular/final-tail only.
        Completion covers ordinary audio only, not control timing.
        """
        rows = []
        complete = reason == "completed" and self.overflow == 0
        final_tails = 0
        audio_count = 0
        previous_s: float | None = None
        previous_p: float | None = None
        predecessor = "first"
        for slot in sorted(
            self.slots[: self.count], key=lambda x: -1 if x.logical_sequence is None else x.logical_sequence
        ):
            e = None if slot.e_ns is None else slot.e_ns / 1e9
            s = None if slot.s_ns is None else slot.s_ns / 1e9
            if slot.kind == "forced_eou":
                # Accepted-audio controls have a logical identity but no audio
                # carrier, ready/eligibility/park stamp, or unit disposition.
                if (
                    slot.logical_sequence is None
                    or slot.logical_sequence < 0
                    or slot.carrier_sequence is not None
                    or slot.s_ns is None
                    or slot.r is not None
                    or slot.e_ns is not None
                    or slot.p is not None
                    or slot.disposition is not None
                    or slot.handle is not None
                ):
                    self.valid = False
            elif slot.kind in ("regular", "final_tail"):
                audio_count += 1
                if slot.logical_sequence is not None and slot.carrier_sequence != slot.logical_sequence % (2**24):
                    self.valid = False
                if previous_p is not None and s is not None and s < previous_p:
                    self.valid = False
                final_tails += slot.kind == "final_tail"
                if (
                    slot.r is None
                    or e is None
                    or s is None
                    or slot.p is None
                    or slot.logical_sequence is None
                    or slot.carrier_sequence is None
                    or slot.disposition != "parked"
                ):
                    complete = False
                elif not slot.r <= e <= s <= slot.p:
                    self.valid = False
            else:
                self.valid = False
            if s is not None:
                if previous_s is not None and s < previous_s:
                    self.valid = False
                previous_s = s
            rows.append(
                dict(
                    logical_sequence=slot.logical_sequence,
                    carrier_sequence=slot.carrier_sequence,
                    kind=slot.kind,
                    r=slot.r,
                    e=e,
                    s=s,
                    p=slot.p,
                    disposition=slot.disposition,
                    predecessor_attribution=predecessor,
                )
            )
            if slot.kind == "forced_eou":
                previous_p = None
                predecessor = "control_unobserved"
            else:
                previous_p = slot.p
                predecessor = "ordinary"
        complete = complete and final_tails == 1
        return dict(
            schema=1,
            session=self.session_key,
            engine_epoch=self.engine_epoch,
            lease_generation=self.lease_generation,
            clock_host=socket.gethostname(),
            process_id=os.getpid(),
            units=rows,
            time_unit="monotonic_seconds",
            capacity=self.capacity,
            count=self.count,
            audio_count=audio_count,
            overflow=self.overflow,
            reason=reason,
            completion_scope="ordinary_audio",
            complete=complete,
            valid=self.valid and complete,
            end=True,
        )


class EngineServiceTimingTrace:
    """Bounded scheduler-side companion to ``ServiceTimingTrace``.

    This records observations only. A schedule stamp is copied from the core
    SCHEDULED admission event (schedule-call start), not GPU execution start.
    Segment generation is
    the scheduler's counter; producer logical/carrier sequences are independent.
    Clocks may be joined only on a verified common host/monotonic clock domain.
    Overflow or missing terminal records invalidate an engine timing analysis.
    """

    capacity = ServiceTimingTrace.capacity * 8

    def __init__(self, request_id: str, identity: dict[str, Any]) -> None:
        self.request_id = request_id
        self.identity: dict[str, Any] | None = identity
        self.session_identity = (identity["session"], identity["engine_epoch"], identity["lease_generation"])
        self.events: list[dict[str, Any]] = []
        self.overflow = 0
        self.valid = True

    def read_input_identity(self, info: dict[str, Any]) -> dict[str, Any] | None:
        """Mark identity loss explicitly without changing the active unit."""
        try:
            identity = engine_service_timing_identity(info)
        except Exception:
            identity = None
        if identity is None:
            self.valid = False
        return identity

    def record_input_end(self, request: Any, timestamp: float) -> None:
        """The core's non-resumable sentinel closes input without a unit."""
        session, engine_epoch, lease_generation = self.session_identity
        self.record(
            request,
            "input_end",
            timestamp=timestamp,
            identity=dict(
                session=session,
                engine_epoch=engine_epoch,
                lease_generation=lease_generation,
                logical_sequence=None,
                carrier_sequence=None,
                kind="input_end",
            ),
        )

    def record(
        self,
        request: Any,
        event: str,
        *,
        timestamp: float | None = None,
        identity: dict[str, Any] | None = None,
        schedule_context: dict[str, Any] | None = None,
    ) -> None:
        if len(self.events) == self.capacity:
            self.overflow += 1
            return
        observed_identity = self.identity if identity is None else identity
        if (
            observed_identity is not None
            and tuple(observed_identity[key] for key in ("session", "engine_epoch", "lease_generation"))
            != self.session_identity
        ):
            self.valid = False
        self.events.append(
            dict(
                event=event,
                timestamp=time.monotonic() if timestamp is None else timestamp,
                identity=observed_identity,
                segment_generation=int(getattr(request, "_omni_segment_generation", 0)),
                queue_depth=len(getattr(request, "streaming_queue", None) or ()),
                schedule_context=schedule_context,
            )
        )

    def finish(self, reason: str) -> dict[str, Any]:
        return dict(
            schema=1,
            request_id=self.request_id,
            clock_host=socket.gethostname(),
            process_id=os.getpid(),
            time_unit="monotonic_seconds",
            events=self.events,
            capacity=self.capacity,
            count=len(self.events),
            overflow=self.overflow,
            reason=reason,
            completion_scope="observations_only",
            valid=self.valid and self.overflow == 0,
            schedule_semantics="waiting_admission_at_schedule_call_start_not_gpu_start",
            clock_join_requires="verified_common_monotonic_domain_or_bounded_offset",
            end=True,
        )


def engine_service_timing_identity(info: dict[str, Any]) -> dict[str, Any] | None:
    """Read a small explicit join key; never infer identity from queue order."""
    raw = info.get("meta", {}).get("service_timing")
    if not isinstance(raw, str) or len(raw) > 2048:
        return None
    identity = json.loads(raw)
    if not isinstance(identity, dict) or set(identity) != {
        "session",
        "engine_epoch",
        "lease_generation",
        "logical_sequence",
        "carrier_sequence",
        "kind",
    }:
        return None
    if not isinstance(identity["session"], str) or identity["kind"] not in (
        "regular",
        "final_tail",
        "forced_eou",
        "flush",
    ):
        return None
    if not isinstance(identity["engine_epoch"], str):
        return None
    if type(identity["lease_generation"]) is not int or identity["lease_generation"] < 0:
        return None
    carrier = identity["carrier_sequence"]
    if identity["kind"] == "flush":
        return identity if identity["logical_sequence"] is None and carrier is None else None
    if type(identity["logical_sequence"]) is not int or identity["logical_sequence"] < 0:
        return None
    if identity["kind"] == "forced_eou":
        if carrier is not None:
            return None
    elif type(carrier) is not int or carrier != identity["logical_sequence"] % (2**24):
        return None
    return identity


def record_engine_service_timing(request: Any, event: str, **kwargs: Any) -> None:
    """Keep optional diagnostics out of scheduling and completion authority."""
    trace = None
    try:
        trace = getattr(request, "_omni_service_timing", None)
        if trace is not None:
            trace.record(request, event, **kwargs)
    except Exception:
        if trace is not None:
            try:
                trace.valid = False
            except Exception:
                pass


def finish_engine_service_timing(request: Any) -> None:
    try:
        trace = getattr(request, "_omni_service_timing", None)
        if trace is not None:
            request._omni_service_timing = None
            _logger.info("Engine service timing trace %s", json.dumps(trace.finish(str(request.status))))
    except Exception:
        pass


def observe_safely(observer_method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke one bound ``StreamingObserver`` method, swallowing failure.

    Observation is non-authoritative (PORT-OBS-002): a misbehaving
    observer implementation must never break feed/flush/park/generation.
    Every production call site across the model package, the lease
    consumer, and the native adapter goes through this helper rather than
    invoking an observer method directly, so a single defensive posture
    covers all of them.

    Args:
        observer_method: A bound method of an installed
            :class:`StreamingObserver` (e.g. ``observer.unit_ready``).
        *args: Forwarded positional arguments.
        **kwargs: Forwarded keyword arguments.

    Returns:
        The method's return value, or ``None`` if it raised.
    """
    try:
        return observer_method(*args, **kwargs)
    except Exception:
        _logger.exception(
            "streaming observer call %r failed; observation is nonfatal (PORT-OBS-002)",
            getattr(observer_method, "__name__", observer_method),
        )
        return None


def drain_batch_stats_into_runner_output(model: Any, runner_output: Any) -> None:
    """The runner-side drain hop: model -> ``OmniModelRunnerOutput``.

    Calls ``model.consume_batch_stats()`` (the PORT-OBS-008 consume-once
    hook — ``advance_model_rows``' ``consume_batch_stats`` in production;
    any object exposing that method in a unit test) once per execution
    and attaches the result to
    ``runner_output.streaming_chunk_batch_stats``, mirroring the existing
    plan-slot pattern. Production seam: ``GPUARModelRunner``'s
    ``OmniModelRunnerOutput`` construction in
    ``vllm_omni/worker/gpu_ar_model_runner.py``.

    Args:
        model: An object exposing ``consume_batch_stats() -> list[tuple[
            int, int]] | None``.
        runner_output: The ``OmniModelRunnerOutput`` to attach the drained
            list to.
    """
    runner_output.streaming_chunk_batch_stats = model.consume_batch_stats()


def forward_batch_stats_to_engine_core_outputs(
    runner_output: Any,
    engine_core_outputs: Any,
    *,
    stats_enabled: bool,
) -> None:
    """The scheduler-side hop: ``OmniModelRunnerOutput`` ->
    ``OmniEngineCoreOutputs``, gated by host statistics collection.

    The omni scheduler owns the host statistics setting (``self.log_stats``
    on ``OmniARScheduler``, invisible to the worker at the pin) and
    forwards the drained list only while statistics are enabled; disabled
    collection must leave ``engine_core_outputs.streaming_chunk_batch_stats``
    as ``None`` regardless of what the runner drained. Production seam:
    ``vllm_omni/core/sched/omni_ar_scheduler.py``'s ``update_from_output``.

    Args:
        runner_output: The ``OmniModelRunnerOutput`` carrying the
            runner-drained ``streaming_chunk_batch_stats``.
        engine_core_outputs: The ``OmniEngineCoreOutputs`` to forward onto.
        stats_enabled: Whether host statistics collection is enabled.
    """
    if not stats_enabled:
        engine_core_outputs.streaming_chunk_batch_stats = None
        return
    # getattr, not attribute access: idle scheduler steps carry vLLM's
    # vanilla ModelRunnerOutput (the shared empty singleton), which has
    # no such attribute — an unconditional read raised AttributeError
    # inside the engine-core busy loop and killed the engine on the
    # first idle step after a generation (2026-07-28 GPU round).
    engine_core_outputs.streaming_chunk_batch_stats = getattr(runner_output, "streaming_chunk_batch_stats", None)
