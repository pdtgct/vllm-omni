# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first Prometheus projection for persistent-state admission capacity."""

from __future__ import annotations

from typing import Any, NoReturn

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.streaming import OmniStreamingMetrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODEL = "capacity-metrics-contract"


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _required(module: Any, name: str) -> Any:
    try:
        return getattr(module, name)
    except AttributeError:
        _fail(f"PORT-OBS-012 missing metrics symbol {name}")


def _sample(prefix: str) -> float | None:
    for line in generate_latest(REGISTRY).decode().splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[-1])
    return None


def _observe(metrics: OmniStreamingMetrics) -> None:
    method = getattr(metrics, "observe_persistent_state_capacity", None)
    if not callable(method):
        _fail("PORT-OBS-012 missing observe_persistent_state_capacity")
    method(
        "0",
        "0",
        service_source="qualified_profile",
        service_budget=1.0,
        committed_demand=0.25,
        execution_claims=2,
        max_num_seqs=8,
        headroom_by_cadence={
            "80": 0,
            "160": 1,
            "320": 2,
            "560": 3,
            "1120": 4,
        },
    )


def test_additive_capacity_families_have_operator_vocabulary() -> None:
    """@spec PORT-OBS-001 / PORT-OBS-012: A27 remains additive."""

    assert _required(defs, "PERSISTENT_STATE_SERVICE_DEMAND_RATIO") == (
        "vllm_omni:persistent_state_service_demand_ratio"
    )
    assert _required(defs, "PERSISTENT_STATE_EXECUTION_CLAIMS") == (
        "vllm_omni:persistent_state_execution_claims"
    )
    assert _required(defs, "PERSISTENT_STATE_ADMISSION_HEADROOM") == (
        "vllm_omni:persistent_state_admission_headroom"
    )


def test_additive_capacity_labels_are_exact_and_bounded() -> None:
    """@spec PORT-OBS-001 / PORT-OBS-012."""

    assert _required(defs, "PERSISTENT_STATE_SERVICE_DEMAND_LABELS") == (
        "model_name",
        "stage",
        "replica",
        "kind",
        "source",
    )
    assert _required(defs, "PERSISTENT_STATE_EXECUTION_CLAIMS_LABELS") == (
        "model_name",
        "stage",
        "replica",
        "kind",
    )
    assert _required(defs, "PERSISTENT_STATE_ADMISSION_HEADROOM_LABELS") == (
        "model_name",
        "stage",
        "replica",
        "cadence_ms",
    )
    assert _required(defs, "PERSISTENT_STATE_SERVICE_DEMAND_KINDS") == (
        "budget",
        "committed_demand",
    )
    assert _required(defs, "PERSISTENT_STATE_SERVICE_DEMAND_SOURCES") == (
        "qualified_profile",
        "measured_fallback",
    )
    assert _required(defs, "PERSISTENT_STATE_EXECUTION_CLAIMS_KINDS") == (
        "claims",
        "max_num_seqs",
    )


def test_capacity_projection_is_one_consistent_snapshot() -> None:
    """@spec PORT-OBS-012: demand, claims, and headroom share one update."""

    metrics = OmniStreamingMetrics(model_name=_MODEL, log_stats=True)
    _observe(metrics)

    budget = _required(defs, "PERSISTENT_STATE_SERVICE_DEMAND_RATIO")
    execution = _required(defs, "PERSISTENT_STATE_EXECUTION_CLAIMS")
    headroom = _required(defs, "PERSISTENT_STATE_ADMISSION_HEADROOM")
    assert _sample(
        f'{budget}{{kind="budget",model_name="{_MODEL}",replica="0",'
        'source="qualified_profile",stage="0"}'
    ) == 1.0
    assert _sample(
        f'{budget}{{kind="committed_demand",model_name="{_MODEL}",'
        'replica="0",source="qualified_profile",stage="0"}'
    ) == 0.25
    assert _sample(
        f'{execution}{{kind="claims",model_name="{_MODEL}",replica="0",'
        'stage="0"}'
    ) == 2.0
    assert _sample(
        f'{execution}{{kind="max_num_seqs",model_name="{_MODEL}",'
        'replica="0",stage="0"}'
    ) == 8.0
    assert _sample(
        f'{headroom}{{cadence_ms="80",model_name="{_MODEL}",replica="0",'
        'stage="0"}'
    ) == 0.0
    assert _sample(
        f'{headroom}{{cadence_ms="1120",model_name="{_MODEL}",replica="0",'
        'stage="0"}'
    ) == 4.0


def test_disabled_statistics_emit_no_capacity_samples() -> None:
    """@spec PORT-OBS-002 / PORT-OBS-012."""

    model = "capacity-metrics-disabled"
    metrics = OmniStreamingMetrics(model_name=model, log_stats=False)
    _observe(metrics)

    out = generate_latest(REGISTRY).decode()
    assert f'model_name="{model}"' not in out


def test_admission_reason_enum_adds_unsupported_without_renaming() -> None:
    """@spec PORT-OBS-001 / PORT-OBS-007: only capacity is retryable."""

    assert defs.STREAMING_ADMISSION_REJECTION_REASONS == (
        "capacity",
        "unavailable",
        "unsupported",
    )
