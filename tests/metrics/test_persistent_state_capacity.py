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
        charged_demand=0.25,
        execution_claims=2,
        max_num_seqs=8,
        headroom_by_cadence={
            "80": {"hard": 0, "nominal": 0},
            "160": {"hard": 1, "nominal": 1},
            "320": {"hard": 2, "nominal": 2},
            "560": {"hard": 3, "nominal": 3},
            "1120": {"hard": 4, "nominal": 4},
        },
        pending_by_cadence={
            "80": {
                "waiting": 1,
                "submitted": 2,
                "reconciling": 3,
                "committed_cleanup": 4,
            },
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
    assert _required(defs, "PERSISTENT_STATE_ADMISSION_PENDING") == (
        "vllm_omni:persistent_state_admission_pending"
    )
    assert _required(defs, "PERSISTENT_STATE_ADMISSION_WAIT_S") == (
        "vllm_omni:persistent_state_admission_wait_s"
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
        "kind",
    )
    assert _required(defs, "PERSISTENT_STATE_ADMISSION_PENDING_LABELS") == (
        "model_name",
        "stage",
        "replica",
        "cadence_ms",
        "state",
    )
    assert _required(defs, "PERSISTENT_STATE_ADMISSION_WAIT_LABELS") == (
        "model_name",
        "stage",
        "replica",
        "cadence_ms",
        "outcome",
    )
    assert _required(defs, "PERSISTENT_STATE_SERVICE_DEMAND_KINDS") == (
        "budget",
        "charged_demand",
    )
    assert _required(defs, "PERSISTENT_STATE_SERVICE_DEMAND_SOURCES") == (
        "qualified_profile",
        "measured_fallback",
    )
    assert _required(defs, "PERSISTENT_STATE_EXECUTION_CLAIMS_KINDS") == (
        "claims",
        "max_num_seqs",
    )
    assert _required(defs, "PERSISTENT_STATE_ADMISSION_HEADROOM_KINDS") == (
        "hard",
        "nominal",
    )
    assert _required(defs, "PERSISTENT_STATE_ADMISSION_PENDING_STATES") == (
        "waiting",
        "submitted",
        "reconciling",
        "committed_cleanup",
    )
    assert _required(defs, "PERSISTENT_STATE_ADMISSION_WAIT_OUTCOMES") == (
        "admitted",
        "shed",
        "unavailable",
        "cancelled",
    )


def test_capacity_projection_is_one_consistent_snapshot() -> None:
    """@spec PORT-OBS-012: demand, claims, and headroom share one update."""

    metrics = OmniStreamingMetrics(model_name=_MODEL, log_stats=True)
    _observe(metrics)

    budget = _required(defs, "PERSISTENT_STATE_SERVICE_DEMAND_RATIO")
    execution = _required(defs, "PERSISTENT_STATE_EXECUTION_CLAIMS")
    headroom = _required(defs, "PERSISTENT_STATE_ADMISSION_HEADROOM")
    pending = _required(defs, "PERSISTENT_STATE_ADMISSION_PENDING")
    assert _sample(
        f'{budget}{{kind="budget",model_name="{_MODEL}",replica="0",'
        'source="qualified_profile",stage="0"}'
    ) == 1.0
    assert _sample(
        f'{budget}{{kind="charged_demand",model_name="{_MODEL}",'
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
        f'{headroom}{{cadence_ms="80",kind="hard",model_name="{_MODEL}",replica="0",'
        'stage="0"}'
    ) == 0.0
    assert _sample(
        f'{headroom}{{cadence_ms="1120",kind="nominal",model_name="{_MODEL}",replica="0",'
        'stage="0"}'
    ) == 4.0
    assert _sample(
        f'{pending}{{cadence_ms="80",model_name="{_MODEL}",replica="0",'
        'stage="0",state="waiting"}'
    ) == 1.0
    assert _sample(
        f'{pending}{{cadence_ms="80",model_name="{_MODEL}",replica="0",'
        'stage="0",state="committed_cleanup"}'
    ) == 4.0


def test_unmeasured_hard_cap_omits_demand_but_keeps_capacity_metrics() -> None:
    """@spec PORT-OBS-012: absence of measurement is never published as zero."""
    model = "unmeasured-hard-cap-contract"
    metrics = OmniStreamingMetrics(model_name=model, log_stats=True)
    metrics.observe_persistent_state_capacity(
        "0",
        "0",
        service_source=None,
        service_budget=None,
        charged_demand=None,
        execution_claims=2,
        max_num_seqs=8,
        headroom_by_cadence={"160": {"hard": 6, "nominal": None}},
        pending_by_cadence={
            "160": {
                "waiting": 1,
                "submitted": 0,
                "reconciling": 0,
                "committed_cleanup": 0,
            }
        },
    )

    demand = _required(defs, "PERSISTENT_STATE_SERVICE_DEMAND_RATIO")
    assert _sample(f'{demand}{{kind="budget",model_name="{model}"') is None
    execution = _required(defs, "PERSISTENT_STATE_EXECUTION_CLAIMS")
    assert _sample(
        f'{execution}{{kind="claims",model_name="{model}",replica="0",stage="0"}}'
    ) == 2.0
    headroom = _required(defs, "PERSISTENT_STATE_ADMISSION_HEADROOM")
    assert _sample(
        f'{headroom}{{cadence_ms="160",kind="hard",model_name="{model}",replica="0",stage="0"}}'
    ) == 6.0
    assert _sample(
        f'{headroom}{{cadence_ms="160",kind="nominal",model_name="{model}"'
    ) is None


def test_admission_wait_observation_uses_bounded_outcome_and_seconds() -> None:
    """@spec PORT-OBS-012: queue wait is distinct from TTFS."""

    metrics = OmniStreamingMetrics(model_name=_MODEL, log_stats=True)
    method = getattr(metrics, "observe_persistent_state_admission_wait", None)
    if not callable(method):
        _fail("PORT-OBS-012 missing admission-wait observation")
    method("0", "0", cadence_ms="320", outcome="admitted", wait_s=0.125)

    family = _required(defs, "PERSISTENT_STATE_ADMISSION_WAIT_S")
    count = _sample(
        f'{family}_count{{cadence_ms="320",model_name="{_MODEL}",'
        'outcome="admitted",replica="0",stage="0"}'
    )
    total = _sample(
        f'{family}_sum{{cadence_ms="320",model_name="{_MODEL}",'
        'outcome="admitted",replica="0",stage="0"}'
    )
    assert count == 1.0
    assert total == 0.125


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
    ), "PORT-OBS-001 missing the approved three-class rejection taxonomy"
