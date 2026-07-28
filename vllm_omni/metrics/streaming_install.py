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

Each function below is real: see the docstring's ``Raises`` for the
validation it performs.
"""

from __future__ import annotations

from typing import Any

from vllm_omni.metrics.streaming import OmniStreamingMetrics, PrometheusStreamingObserver
from vllm_omni.metrics.streaming_transport import StreamingObserver

#: The app-state attribute the installed observer is stored under.
#: Attribute-based (not a module-level dict keyed by app_state) because
#: app_state stand-ins (e.g. ``SimpleNamespace``) are commonly unhashable.
_INSTALLED_ATTR = "_vllm_omni_streaming_observer"


def install_streaming_observer(app_state: Any, *, log_stats: bool) -> PrometheusStreamingObserver:
    """Install the app-owned streaming observer exactly once.

    Must complete (and be resolvable via :func:`resolve_installed_observer`)
    before any native session construction is possible — the ordering
    the design names as step 1 of the two-step install sequence. The
    installed object is a :class:`PrometheusStreamingObserver` (itself
    satisfying the model package's ``StreamingObserver`` protocol by
    shape), wrapping a fresh :class:`OmniStreamingMetrics`.

    PORT-OBS-001: this seam is reached from the omni API server's
    app-state init — that server mounts ``/v1/realtime`` for every
    deployment, so it *is* the streaming serving path family
    registration is scoped to; deployments not running it never import
    these families. Never gate the call on the engine task vocabulary:
    ``AsyncOmniEngine`` derives ``supported_tasks`` only from
    ``{"generate", "speech"}``, so a ``"realtime"`` membership test is
    False in every real deployment (2026-07-28 GPU-round regression).

    Args:
        app_state: The serving app state. Must already carry
            ``openai_serving_models`` (the served-name registry authority);
            construction resolves ``model_name`` from it once.
        log_stats: PORT-OBS-002 — the API server's own host-statistics
            switch (``not args.disable_log_stats``, statistics default
            ON), threaded straight through to ``OmniStreamingMetrics``.
            Required, no default: a caller that silently defaulted this
            would drift from the server's actual statistics setting.

    Returns:
        The installed :class:`PrometheusStreamingObserver`.

    Raises:
        ValueError: If ``app_state.openai_serving_models`` is absent, or
            its ``model_name()`` resolves empty.
        RuntimeError: If installation is late (HTTP serving has already
            started for this app state) or a duplicate (an observer is
            already installed for this app state).
    """
    if getattr(app_state, "http_serving_started", False):
        raise RuntimeError(
            "streaming-metrics install is late: HTTP serving has already "
            "started for this app state — install must complete before "
            "application-plugin participant entry and before HTTP serving "
            "starts (PORT-OBS-009)"
        )
    if getattr(app_state, _INSTALLED_ATTR, None) is not None:
        raise RuntimeError(
            "streaming-metrics observer is already installed for this app "
            "state — duplicate installation fails startup rather than "
            "silently rebinding (PORT-OBS-009)"
        )
    registry = getattr(app_state, "openai_serving_models", None)
    if registry is None:
        raise ValueError(
            "streaming-metrics install requires app_state.openai_serving_models "
            "(the served-name registry authority) to already exist (PORT-OBS-001)"
        )
    model_name = registry.model_name()
    if not model_name:
        raise ValueError("app_state.openai_serving_models.model_name() resolved empty")
    metrics = OmniStreamingMetrics(model_name=model_name, log_stats=log_stats)
    observer = PrometheusStreamingObserver(metrics)
    setattr(app_state, _INSTALLED_ATTR, observer)
    return observer


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
    """
    return getattr(app_state, _INSTALLED_ATTR, None)


def assert_single_api_server_invariant(api_server_count: int = 1) -> None:
    """Assert the pinned single-API-server-process serve invariant.

    Re-checked on every pin bump (port-design.md §Production metric export);
    gauges are process-local and this assertion is what makes that true
    rather than merely inherited.

    Args:
        api_server_count: The number of API-server worker processes the
            resolved config/app indicates.

    Raises:
        RuntimeError: If ``api_server_count`` is not exactly 1.
    """
    if api_server_count != 1:
        raise RuntimeError(
            "vLLM-Omni streaming metrics require the pinned single-API-server "
            f"invariant (process-local gauges); got api_server_count={api_server_count}"
        )


def observe_chunk_batch_stats(
    metrics: OmniStreamingMetrics,
    stats: list[tuple[str, int]] | None,
    *,
    stage: str,
    replica: str,
) -> None:
    """Defensive-skip dispatch for the orchestrator batch-stat sink.

    ``stats`` is the payload drained from ``advance_model_rows``' PORT-OBS-008
    consume-once hook, forwarded through ``OmniModelRunnerOutput`` /
    ``OmniEngineCoreOutputs``. ``None`` means not collecting; ``[]`` means a
    transaction that executed no nonempty CHUNK bucket. Both are skipped
    without observation (PORT-OBS-009).

    Args:
        metrics: The installed streaming metrics wrapper.
        stats: The drained ``(cadence_ms, rows)`` list, or ``None``.
            PORT-OBS-008 (amended, Phase-6 round 2 Q3): cadence is
            already resolved from the geometry authority at recording
            time (``advance.py``, the model package's own manifest
            table) — this dispatch, and every layer above it, stays
            model-agnostic; it never sees a geometry id.
        stage: The stable stage identity for the histogram's labels.
        replica: The stable replica identity for the histogram's labels.
    """
    if not stats:
        return
    for cadence_ms, rows in stats:
        metrics.observe_chunk_batch_size(stage, replica, cadence_ms, rows)
