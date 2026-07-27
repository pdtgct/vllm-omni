# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one-time streaming-metrics installation seam (PORT-OBS-001/003/009).

Two ordered steps at the pin (port-design.md §Production metric export):

1. The observer is created after the serving-model registry exists
   (``app.state.openai_serving_models.model_name()``) and before any native
   session construction is possible.
2. The orchestrator batch-stat sink is attached before application-plugin
   participant entry and before HTTP serving starts.

Both are one-time per app state: a duplicate installation fails startup
rather than silently rebinding, and the installer asserts the pinned
single-API-server invariant rather than inheriting it.

The PORT-OBS-008/009 runner->scheduler batch-stat transport hops
(``drain_batch_stats_into_runner_output``,
``forward_batch_stats_to_engine_core_outputs``) live in the neutral,
Prometheus-free ``vllm_omni.metrics.streaming_transport`` module instead
of here — this module (and its ``PrometheusStreamingObserver``/
``OmniStreamingMetrics`` imports) is Prometheus-coupled and must not be
imported by the GPU worker/scheduler.

Phase-5 tests-first stub: every function below is a typed, unimplemented
seam — see each docstring's ``Raises``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.metrics.streaming import OmniStreamingMetrics, PrometheusStreamingObserver
from vllm_omni.metrics.streaming_transport import StreamingObserver


def install_streaming_observer(app_state: Any) -> PrometheusStreamingObserver:
    """Install the app-owned streaming observer exactly once.

    Must complete (and be resolvable via :func:`resolve_installed_observer`)
    before any native session construction is possible — the ordering
    the design names as step 1 of the two-step install sequence. The
    installed object is a :class:`PrometheusStreamingObserver` (itself
    satisfying the model package's ``StreamingObserver`` protocol by
    shape), wrapping a fresh :class:`OmniStreamingMetrics`.

    Args:
        app_state: The serving app state. Must already carry
            ``openai_serving_models`` (the served-name registry authority);
            construction resolves ``model_name`` from it once.

    Returns:
        The installed :class:`PrometheusStreamingObserver`.

    Raises:
        NotImplementedError: Always, until Phase 6 wires the registry read,
            the duplicate-install guard, and the single-API-server assertion.
    """
    raise NotImplementedError


def resolve_installed_observer(app_state: Any) -> StreamingObserver | None:
    """Read the observer :func:`install_streaming_observer` installed.

    This is the seam the realtime route/serving setup calls to fetch the
    installed observer before constructing a native session (route-to-
    session injection) — never a parallel construction or a second
    install.

    Args:
        app_state: The serving app state.

    Returns:
        The installed observer, or ``None`` if none was installed.

    Raises:
        NotImplementedError: Always, until Phase 6 wires the app-state
            read.
    """
    raise NotImplementedError


def assert_single_api_server_invariant(api_server_count: int = 1) -> None:
    """Assert the pinned single-API-server-process serve invariant.

    Re-checked on every pin bump (port-design.md §Production metric export);
    gauges are process-local and this assertion is what makes that true
    rather than merely inherited.

    Args:
        api_server_count: The number of API-server worker processes the
            resolved config/app indicates.

    Raises:
        NotImplementedError: Always, until Phase 6 implements the check
            (which must raise when ``api_server_count`` is not exactly 1).
    """
    raise NotImplementedError


def observe_chunk_batch_stats(
    metrics: OmniStreamingMetrics,
    stats: list[tuple[int, int]] | None,
    *,
    stage: str,
    replica: str,
    cadence_ms_by_geometry: Mapping[int, str],
) -> None:
    """Defensive-skip dispatch for the orchestrator batch-stat sink.

    ``stats`` is the payload drained from ``advance_model_rows``' PORT-OBS-008
    consume-once hook, forwarded through ``OmniModelRunnerOutput`` /
    ``OmniEngineCoreOutputs``. ``None`` means not collecting; ``[]`` means a
    transaction that executed no nonempty CHUNK bucket. Both are skipped
    without observation (PORT-OBS-009) — this part of the dispatch is real,
    not a stub; only the underlying ``observe_chunk_batch_size`` recording is
    unimplemented.

    Args:
        metrics: The installed streaming metrics wrapper.
        stats: The drained ``(geometry_id, rows)`` list, or ``None``.
        stage: The stable stage identity for the histogram's labels.
        replica: The stable replica identity for the histogram's labels.
        cadence_ms_by_geometry: Geometry id -> admitted cadence-ms label.
    """
    if not stats:
        return
    for geometry_id, rows in stats:
        cadence_ms = cadence_ms_by_geometry[geometry_id]
        metrics.observe_chunk_batch_size(stage, replica, cadence_ms, rows)
