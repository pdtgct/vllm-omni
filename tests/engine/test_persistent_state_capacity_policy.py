# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for deployment-resolved persistent-state capacity."""

from __future__ import annotations

import importlib
from fractions import Fraction
from types import ModuleType
from typing import Any, NoReturn

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "vllm_omni.engine.persistent_state_capacity"

_SYMBOL_SPECS = {
    "OnePointServiceDemand": "PORT-STATE-025",
    "PersistentStateAdmissionController": "PORT-STATE-025",
    "ServiceProfileContext": "PORT-PERF-006",
    "ServiceRateExecution": "PORT-PERF-006",
    "TwoTermServiceDemand": "PORT-STATE-025",
    "admission_headroom": "PORT-OBS-012",
    "derive_provisional_profile": "PORT-STATE-025",
    "reduce_startup_service_rate": "PORT-PERF-006",
    "resolve_pool_capacity": "PORT-STATE-004",
    "evaluate_admission": "PORT-STATE-026",
}


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _module(spec_id: str) -> ModuleType:
    try:
        return importlib.import_module(_MODULE)
    except ModuleNotFoundError:
        _fail(f"{spec_id} missing {_MODULE}")


def _symbol(name: str) -> Any:
    spec_id = _SYMBOL_SPECS[name]
    module = _module(spec_id)
    try:
        return getattr(module, name)
    except AttributeError:
        _fail(f"{spec_id} missing {_MODULE}.{name}")


def _resolution(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "profiled_block_bound": 20,
        "safety_reserve_slots": 2,
        "max_resident_sessions": 5,
        "num_gpu_blocks_override": None,
        "page_size_bytes": 6_314_936,
    }
    values.update(overrides)
    return _symbol("resolve_pool_capacity")(**values)


def _evaluate(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "resident_counts_by_interval": (0, 0, 0, 0, 0),
        "reserved_counts_by_interval": (0, 0, 0, 0, 0),
        "candidate_interval_ms": 320,
        "allocated_slots": 8,
        "count_limit": 8,
        "resident_count": 0,
        "reserved_count": 0,
        "execution_claims": 0,
        "max_num_seqs": 8,
        "charged_demand": Fraction(0),
        "service_budget": Fraction(1),
        "candidate_demand": Fraction(1, 8),
        "transaction_duration_ms": lambda counts: Fraction(
            40 * sum(counts)
        ),
    }
    values.update(overrides)
    return _symbol("evaluate_admission")(**values)


def _controller(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "allocated_slots": 8,
        "count_limit": 8,
        "max_num_seqs": 8,
        "service_budget": Fraction(1),
        "demand_model": _symbol("OnePointServiceDemand")(
            reference_interval_ms=320,
            reference_demand=Fraction(1, 4),
        ),
        "transaction_duration_ms": lambda intervals: Fraction(
            40 * len(intervals)
        ),
    }
    values.update(overrides)
    return _symbol("PersistentStateAdmissionController")(**values)


def test_logical_count_sizes_the_pool_downward_only() -> None:
    """@spec PORT-STATE-004: policy may return memory, never invent it."""

    resolution = _resolution()

    assert resolution.profiled_block_bound == 20
    assert resolution.allocated_total_blocks == 5 + 2 + 1
    assert resolution.allocated_real_slots == 5
    assert resolution.resolved_count_limit == 5


def test_oversized_logical_count_clamps_without_sizing_up() -> None:
    """@spec PORT-STATE-004: an oversized logical count is harmlessly bounded."""

    resolution = _resolution(max_resident_sessions=100)

    assert resolution.allocated_total_blocks == 20
    assert resolution.allocated_real_slots == 17
    assert resolution.resolved_count_limit == 17
    assert resolution.count_was_clamped is True


def test_explicit_override_above_the_preoverride_bound_fails() -> None:
    """@spec PORT-STATE-004: core's self-satisfying override check is insufficient."""

    with pytest.raises(
        ValueError,
        match=r"21.*20|requested.*profiled|profiled.*bound",
    ):
        _resolution(num_gpu_blocks_override=21)


def test_explicit_override_below_the_bound_controls_allocation_only() -> None:
    """@spec PORT-STATE-004: allocation and the logical count remain distinct."""

    resolution = _resolution(
        max_resident_sessions=100,
        num_gpu_blocks_override=10,
    )

    assert resolution.allocated_total_blocks == 10
    assert resolution.allocated_real_slots == 7
    assert resolution.resolved_count_limit == 7


def test_one_point_envelope_is_exact_rational_math() -> None:
    """@spec PORT-STATE-025: D0*max(1,t0/t), with no float rounding."""

    model = _symbol("OnePointServiceDemand")(
        reference_interval_ms=320,
        reference_demand=Fraction(1, 4),
    )

    assert model.at(80) == Fraction(1)
    assert model.at(160) == Fraction(1, 2)
    assert model.at(320) == Fraction(1, 4)
    assert model.at(1120) == Fraction(1, 4)
    assert isinstance(model.at(80), Fraction)


def test_two_term_model_preserves_invocation_and_audio_terms() -> None:
    """@spec PORT-STATE-025: mixed intervals use a/t+b, never 1/t alone."""

    model = _symbol("TwoTermServiceDemand")(
        invocation_demand_ms=Fraction(16),
        audio_rate_demand=Fraction(1, 10),
    )

    assert model.at(80) == Fraction(3, 10)
    assert model.at(160) == Fraction(1, 5)
    assert model.at(320) == Fraction(3, 20)


@pytest.mark.parametrize(
    ("measured", "factor", "expected"),
    [
        (10, Fraction(1, 2), 5),
        (3, Fraction(1, 2), 1),
        (1, Fraction(1, 100), 1),
    ],
)
def test_measured_fallback_is_derated_and_never_zero(
    measured: int,
    factor: Fraction,
    expected: int,
) -> None:
    """@spec PORT-STATE-025: unknown hardware receives an unqualified floor."""

    profile = _symbol("derive_provisional_profile")(
        measured_sessions=measured,
        derating_factor=factor,
        reference_interval_ms=1120,
    )

    assert profile.capacity == expected
    assert profile.reference_demand == Fraction(1, expected)
    assert profile.source == "measured_fallback"
    assert profile.qualified is False


def _profile_execution(
    *,
    row_count: int,
    elapsed_ns: int,
    completed_rows: int | None = None,
    post_jit: bool = True,
    continuously_loaded: bool = True,
) -> Any:
    return _symbol("ServiceRateExecution")(
        row_count=row_count,
        elapsed_ns=elapsed_ns,
        completed_rows=(
            row_count if completed_rows is None else completed_rows
        ),
        post_jit=post_jit,
        continuously_loaded=continuously_loaded,
    )


def _profile_context(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "pre_override_physical_bound": 20,
        "allocated_pool": 8,
        "count_cap": 7,
        "execution_claim_ceiling": 16,
        "service_budget_source": "measured_fallback",
        "service_budget_coefficients": (Fraction(1, 4),),
        "derating_factor": Fraction(1, 2),
        "slot_bytes": 6_314_936,
        "execution_environment_key": "a100-40gb-cuda13-torch2.9",
        "precision_policy": "fp32",
        "state_profile": "nemotron-asr-fp32-v1",
    }
    values.update(overrides)
    return _symbol("ServiceProfileContext")(**values)


def test_profile_reduction_uses_only_trailing_post_jit_executions() -> None:
    """@spec PORT-PERF-006: warmup cannot inflate measured service rate."""

    result = _symbol("reduce_startup_service_rate")(
        (
            _profile_execution(
                row_count=1,
                elapsed_ns=1,
                post_jit=False,
            ),
            _profile_execution(row_count=1, elapsed_ns=100_000_000),
            _profile_execution(row_count=1, elapsed_ns=200_000_000),
            _profile_execution(row_count=1, elapsed_ns=250_000_000),
        ),
        operating_row_counts=(1,),
        trailing_executions=2,
        context=_profile_context(allocated_pool=1, count_cap=1),
    )

    assert result.measured_rate_rows_per_s == Fraction(40, 9)
    assert result.trailing_executions == 2


def test_profile_reduction_uses_least_rate_across_executed_range() -> None:
    """@spec PORT-PERF-006: the weakest executed operating point gates."""

    result = _symbol("reduce_startup_service_rate")(
        (
            _profile_execution(row_count=1, elapsed_ns=100_000_000),
            _profile_execution(row_count=1, elapsed_ns=100_000_000),
            _profile_execution(row_count=4, elapsed_ns=500_000_000),
            _profile_execution(row_count=4, elapsed_ns=500_000_000),
        ),
        operating_row_counts=(1, 4),
        trailing_executions=2,
        context=_profile_context(allocated_pool=4, count_cap=4),
    )

    assert result.rate_by_row_count == {
        1: Fraction(10),
        4: Fraction(8),
    }
    assert result.operating_row_count == 4
    assert result.measured_rate_rows_per_s == Fraction(8)


def test_profile_reduction_refuses_extrapolation_or_noncontinuous_data() -> None:
    """@spec PORT-PERF-006: fallback evidence covers its operating range."""

    reduce_profile = _symbol("reduce_startup_service_rate")
    with pytest.raises(ValueError, match=r"operating.*4|missing.*4"):
        reduce_profile(
            (_profile_execution(row_count=1, elapsed_ns=100_000_000),),
            operating_row_counts=(1, 4),
            trailing_executions=1,
            context=_profile_context(allocated_pool=4, count_cap=4),
        )
    with pytest.raises(ValueError, match=r"continuous|load"):
        reduce_profile(
            (
                _profile_execution(
                    row_count=1,
                    elapsed_ns=100_000_000,
                    continuously_loaded=False,
                ),
            ),
            operating_row_counts=(1,),
            trailing_executions=1,
            context=_profile_context(allocated_pool=1, count_cap=1),
        )
    with pytest.raises(ValueError, match=r"allocated.*4|operating.*4"):
        reduce_profile(
            (_profile_execution(row_count=1, elapsed_ns=100_000_000),),
            operating_row_counts=(1,),
            trailing_executions=1,
            context=_profile_context(allocated_pool=4, count_cap=4),
        )


def test_profile_candidate_and_receipt_stamp_all_authorities() -> None:
    """@spec PORT-PERF-006: candidates are complete and never self-promote."""

    result = _symbol("reduce_startup_service_rate")(
        (_profile_execution(row_count=4, elapsed_ns=500_000_000),),
        operating_row_counts=(4,),
        trailing_executions=1,
        context=_profile_context(allocated_pool=4, count_cap=4),
    )

    assert result.profile_candidate.execution_environment_key == (
        "a100-40gb-cuda13-torch2.9"
    )
    assert result.profile_candidate.precision_policy == "fp32"
    assert result.profile_candidate.state_profile == "nemotron-asr-fp32-v1"
    assert result.profile_candidate.qualified is False
    assert result.profile_candidate.installable is False
    assert result.receipt.pre_override_physical_bound == 20
    assert result.receipt.allocated_pool == 4
    assert result.receipt.count_cap == 4
    assert result.receipt.execution_claim_ceiling == 16
    assert result.receipt.service_budget_source == "measured_fallback"
    assert result.receipt.service_budget_coefficients == (Fraction(1, 4),)
    assert result.receipt.derating_factor == Fraction(1, 2)
    assert result.receipt.measured_rate_rows_per_s == Fraction(8)
    assert result.receipt.slot_bytes == 6_314_936
    assert result.receipt.execution_environment_key == (
        "a100-40gb-cuda13-torch2.9"
    )
    assert result.receipt.idle_device is True
    assert result.receipt.no_competing_tenant is True
    assert result.receipt.synthetic_silence is True


def test_mixed_pool_pressure_waits_instead_of_becoming_a_refusal() -> None:
    """@spec PORT-STATE-025 / PORT-STATE-026: nominal is dispatch guidance."""

    decision = _evaluate(
        resident_counts_by_interval=(0, 0, 0, 0, 1),
        resident_count=1,
        candidate_interval_ms=80,
        transaction_duration_ms=lambda counts: Fraction(80 * sum(counts)),
    )

    assert decision.candidate_supported is True
    assert decision.hard_feasible is True
    assert decision.nominal_dispatchable is False
    assert decision.nominal_reason == "transaction_time"
    assert not hasattr(decision, "refusal"), (
        "the capacity projection must not turn nominal pressure into a "
        "client-visible refusal"
    )


def test_empty_pool_latency_failure_is_nonretryable_unsupported() -> None:
    """@spec PORT-STATE-024 / PORT-STATE-026: impossible alone is not overload."""

    decision = _evaluate(
        candidate_interval_ms=80,
        transaction_duration_ms=lambda intervals: Fraction(81),
    )

    assert decision.candidate_supported is False
    assert decision.hard_feasible is False
    assert decision.nominal_dispatchable is False
    assert decision.binding_authority == "candidate_alone"


@pytest.mark.parametrize(
    ("override", "authority"),
    [
        ({"allocated_slots": 0}, "physical_slots"),
        ({"count_limit": 0}, "logical_count"),
        ({"execution_claims": 8}, "execution_claims"),
    ],
)
def test_each_hard_authority_blocks_dispatch_without_reclassifying_the_client(
    override: dict[str, Any],
    authority: str,
) -> None:
    """@spec PORT-STATE-004 / PORT-STATE-024: hard pressure waits first."""

    decision = _evaluate(**override)

    assert decision.candidate_supported is True
    assert decision.hard_feasible is False
    assert decision.binding_authority == authority
    assert not hasattr(decision, "retryable")


def test_service_budget_is_nominal_and_never_creates_physical_capacity() -> None:
    """@spec PORT-STATE-004 / PORT-STATE-025 / PORT-STATE-026."""

    decision = _evaluate(
        charged_demand=Fraction(7, 8),
        candidate_demand=Fraction(1, 4),
    )

    assert decision.candidate_supported is True
    assert decision.hard_feasible is True
    assert decision.nominal_dispatchable is False
    assert decision.nominal_reason == "service_budget"
    assert decision.hard_headroom == 8


def test_headroom_uses_fixed_cardinality_aggregates() -> None:
    """@spec PORT-OBS-012: no resident-population input or scan."""

    headroom = _symbol("admission_headroom")(
        resident_counts_by_interval=(0, 0, 0, 0, 0),
        reserved_counts_by_interval=(0, 0, 0, 0, 0),
        candidate_interval_ms=160,
        allocated_slots=8,
        count_limit=8,
        resident_count=0,
        reserved_count=0,
        execution_claims=0,
        max_num_seqs=8,
        charged_demand=Fraction(0),
        service_budget=Fraction(1),
        candidate_demand=Fraction(1, 4),
        transaction_duration_ms=lambda counts: Fraction(40 * sum(counts)),
    )

    assert headroom.hard == 8
    assert headroom.nominal == 4


def test_headroom_work_is_sublinear_in_capacity() -> None:
    """@spec PORT-PERF-008 / PORT-OBS-012: never loop once per slot."""

    calls = 0

    def transaction_duration(counts: tuple[int, ...]) -> Fraction:
        nonlocal calls
        calls += 1
        assert len(counts) == 5
        return Fraction(sum(counts))

    headroom = _symbol("admission_headroom")(
        resident_counts_by_interval=(0, 0, 0, 0, 0),
        reserved_counts_by_interval=(0, 0, 0, 0, 0),
        candidate_interval_ms=1120,
        allocated_slots=1_000_000,
        count_limit=1_000_000,
        resident_count=0,
        reserved_count=0,
        execution_claims=0,
        max_num_seqs=1_000_000,
        charged_demand=Fraction(0),
        service_budget=Fraction(1_000_000),
        candidate_demand=Fraction(1),
        transaction_duration_ms=transaction_duration,
    )

    assert headroom.hard == 1_000_000
    assert headroom.nominal == 1_000_000
    assert calls <= 32, (
        "headroom may use fixed-cardinality arithmetic or logarithmic "
        "monotone search, never one evaluation/allocation per slot"
    )


def test_execution_claim_and_demand_span_reserve_to_release() -> None:
    """@spec PORT-STATE-004 / PORT-STATE-025: preconstruction and park stay charged."""

    controller = _controller()
    claim = controller.reserve("lease-a", service_interval_ms=320)

    assert controller.snapshot.execution_claims == 1
    assert controller.snapshot.charged_demand == Fraction(1, 4)
    assert controller.snapshot.intervals_ms == (320,)
    controller.park(claim)
    assert controller.snapshot.execution_claims == 1
    assert controller.snapshot.charged_demand == Fraction(1, 4)
    controller.release(claim)
    assert controller.snapshot.execution_claims == 0
    assert controller.snapshot.charged_demand == Fraction(0)
    assert controller.snapshot.intervals_ms == ()


def test_duplicate_release_cannot_return_capacity_twice() -> None:
    """@spec PORT-STATE-014 / PORT-STATE-025."""

    controller = _controller()
    claim = controller.reserve("lease-a", service_interval_ms=560)
    controller.release(claim)
    controller.release(claim)

    assert controller.snapshot.execution_claims == 0
    assert controller.snapshot.charged_demand == Fraction(0)


def test_operator_count_never_enlarges_hard_headroom() -> None:
    """@spec PORT-STATE-004 / PORT-STATE-025."""

    headroom = _symbol("admission_headroom")(
        resident_counts_by_interval=(0, 0, 0, 0, 1),
        reserved_counts_by_interval=(0, 0, 0, 0, 0),
        candidate_interval_ms=1120,
        allocated_slots=2,
        count_limit=100,
        resident_count=1,
        reserved_count=0,
        execution_claims=1,
        max_num_seqs=1,
        charged_demand=Fraction(1, 4),
        service_budget=Fraction(1),
        candidate_demand=Fraction(1, 4),
        transaction_duration_ms=lambda counts: Fraction(40 * sum(counts)),
    )

    assert headroom.hard == 0
