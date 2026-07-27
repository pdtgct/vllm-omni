# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The neutral, Prometheus-free streaming-observability transport module.

Owns two things that must NOT live in the model package or the Prometheus
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

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ChunkReadyHandle:
    """Opaque per-unit identity carried through ready -> minted ->
    parked/cleared (PORT-OBS-003/004/005).

    Frozen (hence hashable) so callers may key sets/dicts by handle
    identity — e.g. asserting "every ready handle received exactly one
    disposition" over a ``set`` of handles.

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

    def session_opened(self, *, cadence_ms: str) -> None:
        """One successful session open (PORT-OBS-006): opens the active
        gauge. Native open is observer-bearing model-session construction
        after successful validation and before engine request creation —
        never WebSocket acceptance."""
        ...

    def session_finished(self, *, cadence_ms: str, reason: str) -> None:
        """The session's single idempotent terminal-disposition section:
        active decrements and finished increments together, so
        opens - finished == active."""
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

    Raises:
        NotImplementedError: Always, until Phase 6 wires this hop.
    """
    raise NotImplementedError


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

    Raises:
        NotImplementedError: Always, until Phase 6 wires this hop.
    """
    raise NotImplementedError
