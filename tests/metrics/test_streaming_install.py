# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: the streaming-metrics install seam.

Specs: PORT-OBS-001 (registry-resolved model_name; single-API-server
invariant asserted at install), PORT-OBS-009 (one-time install per app
state; orchestrator-side defensive-skip of None/[] batch-stat payloads;
OmniModelRunnerOutput/OmniEngineCoreOutputs field defaults).

``install_streaming_observer``/``assert_single_api_server_invariant``
unconditionally raise ``NotImplementedError`` (Phase-5 stub in
``streaming_install.py``), so the install-behavior tests below are
EXPECTED RED with that exact failure mode. ``observe_chunk_batch_stats``'s
None/[] defensive-skip branches ARE implemented (real dispatch plumbing,
not the metric itself), so those tests are GREEN today.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.metrics.streaming import OmniStreamingMetrics, PrometheusStreamingObserver
from vllm_omni.metrics.streaming_install import (
    assert_single_api_server_invariant,
    install_streaming_observer,
    observe_chunk_batch_stats,
    resolve_installed_observer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _app_state_with_registry(model_name: str = "install-test-model") -> Any:
    registry = SimpleNamespace(model_name=lambda: model_name)
    return SimpleNamespace(openai_serving_models=registry)


# ---- install requires the registry (PORT-OBS-001) — RED ----------------------


# @spec PORT-OBS-001
def test_install_requires_the_served_model_registry_to_exist() -> None:
    app_state = SimpleNamespace()  # no openai_serving_models attribute

    with pytest.raises(ValueError, match="openai_serving_models"):
        install_streaming_observer(app_state)


# @spec PORT-OBS-001
def test_install_resolves_model_name_from_the_registry_once() -> None:
    app_state = _app_state_with_registry("resolved-alias")

    observer = install_streaming_observer(app_state)

    assert isinstance(observer, PrometheusStreamingObserver)
    assert isinstance(observer._metrics, OmniStreamingMetrics)
    assert observer._metrics._model_name == "resolved-alias"


# @spec PORT-OBS-003
def test_installed_object_satisfies_the_streaming_observer_protocol() -> None:
    """Structural check (correction 6): the object install_streaming_observer
    installs must satisfy the model package's Prometheus-free
    StreamingObserver protocol (runtime-checkable isinstance) — even
    though this metrics package never imports that protocol itself."""
    from vllm_omni.model_executor.models.nemotron_asr.session import (
        StreamingObserver,
    )

    app_state = _app_state_with_registry()
    observer = install_streaming_observer(app_state)
    assert isinstance(observer, StreamingObserver)


# ---- duplicate install into the same app state fails (PORT-OBS-009) — RED ----


# @spec PORT-OBS-009
def test_duplicate_install_into_the_same_app_state_raises() -> None:
    app_state = _app_state_with_registry()

    install_streaming_observer(app_state)
    with pytest.raises(RuntimeError, match="already installed"):
        install_streaming_observer(app_state)


# @spec PORT-OBS-009
def test_install_into_a_different_app_state_is_independent() -> None:
    first = _app_state_with_registry("model-a")
    second = _app_state_with_registry("model-b")

    install_streaming_observer(first)
    install_streaming_observer(second)  # must not raise


# @spec PORT-OBS-009
def test_late_installation_after_http_serving_started_fails() -> None:
    """Step 2 of the design's install ordering: the sink must attach
    before application-plugin participant entry and before HTTP serving
    starts. Installing after that point is "late" and must fail through
    the same host readiness-failure path as a duplicate install."""
    app_state = _app_state_with_registry()
    app_state.http_serving_started = True

    with pytest.raises(RuntimeError, match="late|already"):
        install_streaming_observer(app_state)


# ---- install ordering: must precede native session construction -------------
# (PORT-OBS-001/003, design step 1 of the two-step install sequence) — RED


# @spec PORT-OBS-001, PORT-OBS-003
def test_observer_is_not_resolvable_before_install() -> None:
    app_state = _app_state_with_registry()
    assert resolve_installed_observer(app_state) is None


# @spec PORT-OBS-001, PORT-OBS-003
def test_install_makes_the_observer_resolvable_for_native_session_construction() -> None:
    """The ordering claim: once install completes, the SAME instance is
    what the native route/serving setup resolves before constructing a
    session — never a second, independently-constructed observer."""
    app_state = _app_state_with_registry()

    metrics = install_streaming_observer(app_state)

    assert resolve_installed_observer(app_state) is metrics


# ---- route-to-session injection (PORT-OBS-003) — RED --------------------------


# @spec PORT-OBS-003
def test_route_setup_injects_the_installed_observer_into_native_session_construction() -> None:
    """The realtime route/serving setup must pass the INSTALLED observer
    into native session construction — modeled here at the seam level:
    resolving the observer off app_state and constructing a
    NemotronRealtimeSession with it must yield that same observer on the
    session, never a fresh/independent one."""
    from vllm_omni.model_executor.models.nemotron_asr.session import (
        NemotronRealtimeSession,
    )

    app_state = _app_state_with_registry()
    install_streaming_observer(app_state)
    observer = resolve_installed_observer(app_state)

    hf_config = SimpleNamespace(
        eos_token_id=1,
        audio_chunk_token_id=2,
        prompt_dictionary={"auto": 0},
        num_prompts=1,
    )
    session = NemotronRealtimeSession.from_model_config(hf_config, observer=observer)

    assert session.observer is observer
    assert observer is not None


# ---- single-API-server assertion (PORT-OBS-001) — RED --------------------------


# @spec PORT-OBS-001
def test_single_api_server_assertion_hook_exists_and_is_callable() -> None:
    assert callable(assert_single_api_server_invariant)
    # Pinning the intended behavior once implemented: a single-process
    # serve invocation must assert cleanly (never raise) here.
    assert_single_api_server_invariant(api_server_count=1)


# @spec PORT-OBS-001
def test_single_api_server_assertion_rejects_multiple_workers() -> None:
    with pytest.raises(RuntimeError, match="single"):
        assert_single_api_server_invariant(api_server_count=2)


# ---- orchestrator-side observe skips None and [] payloads (real, GREEN) ------


class _RecordingMetrics:
    def __init__(self) -> None:
        self.observed: list[tuple[str, str, str, int]] = []

    def observe_chunk_batch_size(self, stage: str, replica: str, cadence_ms: str, rows: int) -> None:
        self.observed.append((stage, replica, cadence_ms, rows))


# @spec PORT-OBS-009
def test_none_payload_is_skipped_without_observation() -> None:
    metrics: Any = _RecordingMetrics()
    observe_chunk_batch_stats(metrics, None, stage="0", replica="0", cadence_ms_by_geometry={})
    assert metrics.observed == []


# @spec PORT-OBS-009
def test_empty_list_payload_is_skipped_without_observation() -> None:
    metrics: Any = _RecordingMetrics()
    observe_chunk_batch_stats(metrics, [], stage="0", replica="0", cadence_ms_by_geometry={})
    assert metrics.observed == []


# @spec PORT-OBS-008, PORT-OBS-009
def test_nonempty_payload_dispatches_one_observation_per_geometry_entry() -> None:
    metrics: Any = _RecordingMetrics()
    cadence_by_geometry = {0: "80", 2: "320"}
    observe_chunk_batch_stats(
        metrics,
        [(0, 4), (2, 9)],
        stage="0",
        replica="1",
        cadence_ms_by_geometry=cadence_by_geometry,
    )
    assert metrics.observed == [("0", "1", "80", 4), ("0", "1", "320", 9)]


# ---- runner/scheduler batch-stat transport hops now live in --------------------
# tests/metrics/test_streaming_transport.py (moved with the functions
# themselves, correction 3: they belong to the neutral, Prometheus-free
# vllm_omni.metrics.streaming_transport module, not this Prometheus-
# coupled install seam).


# ---- orchestrator consumption from a real OmniEngineCoreOutputs (real, GREEN) -
# (PORT-OBS-009)


# @spec PORT-OBS-008, PORT-OBS-009
def test_orchestrator_observes_batch_size_from_engine_core_outputs_entries() -> None:
    from vllm_omni.engine import OmniEngineCoreOutputs

    metrics: Any = _RecordingMetrics()
    engine_core_outputs = OmniEngineCoreOutputs(streaming_chunk_batch_stats=[(0, 4), (2, 9)])

    observe_chunk_batch_stats(
        metrics,
        engine_core_outputs.streaming_chunk_batch_stats,
        stage="0",
        replica="1",
        cadence_ms_by_geometry={0: "80", 2: "320"},
    )

    assert metrics.observed == [("0", "1", "80", 4), ("0", "1", "320", 9)]


# @spec PORT-OBS-009
def test_orchestrator_skips_none_batch_stats_on_engine_core_outputs() -> None:
    from vllm_omni.engine import OmniEngineCoreOutputs

    metrics: Any = _RecordingMetrics()
    engine_core_outputs = OmniEngineCoreOutputs()  # defaults to None

    observe_chunk_batch_stats(
        metrics,
        engine_core_outputs.streaming_chunk_batch_stats,
        stage="0",
        replica="1",
        cadence_ms_by_geometry={},
    )

    assert metrics.observed == []


# @spec PORT-OBS-009
def test_orchestrator_skips_empty_list_batch_stats_on_engine_core_outputs() -> None:
    from vllm_omni.engine import OmniEngineCoreOutputs

    metrics: Any = _RecordingMetrics()
    engine_core_outputs = OmniEngineCoreOutputs(streaming_chunk_batch_stats=[])

    observe_chunk_batch_stats(
        metrics,
        engine_core_outputs.streaming_chunk_batch_stats,
        stage="0",
        replica="1",
        cadence_ms_by_geometry={},
    )

    assert metrics.observed == []


# ---- OmniModelRunnerOutput/OmniEngineCoreOutputs field defaults (real) -------


# @spec PORT-OBS-008, PORT-OBS-009
def test_omni_model_runner_output_batch_stats_field_defaults_none() -> None:
    from vllm_omni.outputs import OmniModelRunnerOutput

    out = OmniModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
    assert out.streaming_chunk_batch_stats is None


# @spec PORT-OBS-008, PORT-OBS-009
def test_omni_engine_core_outputs_batch_stats_field_defaults_none() -> None:
    from vllm_omni.engine import OmniEngineCoreOutputs

    out = OmniEngineCoreOutputs()
    assert out.streaming_chunk_batch_stats is None


# @spec PORT-OBS-008, PORT-OBS-009
def test_omni_model_runner_output_batch_stats_field_survives_construction() -> None:
    from vllm_omni.outputs import OmniModelRunnerOutput

    payload = [(0, 3), (1, 7)]
    out = OmniModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        streaming_chunk_batch_stats=payload,
    )
    assert out.streaming_chunk_batch_stats == payload


# @spec PORT-OBS-008, PORT-OBS-009
def test_omni_engine_core_outputs_batch_stats_field_survives_construction() -> None:
    from vllm_omni.engine import OmniEngineCoreOutputs

    payload = [(0, 3)]
    out = OmniEngineCoreOutputs(streaming_chunk_batch_stats=payload)
    assert out.streaming_chunk_batch_stats == payload
