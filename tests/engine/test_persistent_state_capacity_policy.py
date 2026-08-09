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
_SERVICE_INTERVALS_MS = (80, 160, 320, 560, 1_120)

_SYMBOL_SPECS = {
    "OnePointServiceDemand": "PORT-STATE-025",
    "PersistentStateAdmissionController": "PORT-STATE-025",
    "ServiceExecutionTier": "PORT-PERF-006",
    "ServiceProfileContext": "PORT-PERF-006",
    "ServiceRoundExecution": "PORT-PERF-006",
    "TwoTermServiceDemand": "PORT-STATE-025",
    "admission_headroom": "PORT-OBS-012",
    "compile_provisional_service_profile": "PORT-PERF-006",
    "compile_service_demand_profile": "PORT-STATE-025",
    "derive_provisional_profile": "PORT-STATE-025",
    "fallback_transaction_duration_ns": "PORT-STATE-026",
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


def _service_tier(tier_id: str, maximum: int) -> Any:
    return _symbol("ServiceExecutionTier")(
        tier_id=tier_id,
        max_active_population=maximum,
    )


def _service_round(
    *,
    tier_id: str,
    active_population: int,
    elapsed_ns: int,
    service_interval_ms: int = 1_120,
    geometry_id: int = 4,
    completed_legal_parks: int | None = None,
    completed_model_rows: int | None = None,
    post_jit: bool = True,
    continuously_loaded: bool = True,
    dummy_run: bool = False,
    is_profile: bool = False,
) -> Any:
    return _symbol("ServiceRoundExecution")(
        tier_id=tier_id,
        active_population=active_population,
        elapsed_ns=elapsed_ns,
        service_interval_ms=service_interval_ms,
        geometry_id=geometry_id,
        completed_legal_parks=(
            active_population
            if completed_legal_parks is None
            else completed_legal_parks
        ),
        completed_model_rows=completed_model_rows,
        post_jit=post_jit,
        continuously_loaded=continuously_loaded,
        dummy_run=dummy_run,
        is_profile=is_profile,
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
        "compiler_version": "persistent-state-capacity-v2",
        "mixed_composition_policy": "homogeneous_upper_sum",
    }
    values.update(overrides)
    return _symbol("ServiceProfileContext")(**values)


def _geometry_rounds(
    *,
    single_elapsed_ns: int,
    small_elapsed_ns: int,
    single_model_rows: int | None = None,
    small_model_rows: int | None = None,
    post_jit: bool = True,
) -> tuple[Any, ...]:
    return tuple(
        execution
        for geometry_id, service_interval_ms in enumerate(
            _SERVICE_INTERVALS_MS
        )
        for execution in (
            _service_round(
                tier_id="single",
                active_population=1,
                elapsed_ns=single_elapsed_ns,
                service_interval_ms=service_interval_ms,
                geometry_id=geometry_id,
                completed_model_rows=single_model_rows,
                post_jit=post_jit,
            ),
            _service_round(
                tier_id="small",
                active_population=4,
                elapsed_ns=small_elapsed_ns,
                service_interval_ms=service_interval_ms,
                geometry_id=geometry_id,
                completed_model_rows=small_model_rows,
                post_jit=post_jit,
            ),
        )
    )


def _compile_profile(
    executions: tuple[Any, ...],
    *,
    tiers: tuple[Any, ...] | None = None,
    max_population: int = 4,
    trailing_rounds: int = 1,
    derating_factor: Fraction = Fraction(1, 2),
) -> Any:
    return _symbol("compile_provisional_service_profile")(
        executions,
        execution_tiers=(
            (_service_tier("single", 1), _service_tier("small", 4))
            if tiers is None
            else tiers
        ),
        max_population=max_population,
        reference_interval_ms=1_120,
        reference_geometry_id=4,
        admitted_geometry_ids=(0, 1, 2, 3, 4),
        trailing_rounds=trailing_rounds,
        derating_factor=derating_factor,
        context=_profile_context(
            allocated_pool=max_population,
            count_cap=max_population,
        ),
    )


def test_profile_expands_runner_tier_upper_bounds_without_interpolation() -> None:
    """@spec PORT-PERF-006: finite tier coverage expands exactly to 1..P."""

    result = _compile_profile(
        _geometry_rounds(
            single_elapsed_ns=100_000_000,
            small_elapsed_ns=80_000_000,
        )
    )

    assert result.upper_duration_ns_by_geometry_and_population[4] == (
        100_000_000,
        100_000_000,
        100_000_000,
        100_000_000,
    )
    assert result.measured_capacity == 4
    assert result.provisional_capacity == 2
    assert result.reference_demand == Fraction(1, 2)


def test_profile_uses_worst_trailing_complete_round_and_legal_park() -> None:
    """@spec PORT-PERF-006: no favorable sample or partial row is capacity."""

    result = _compile_profile(
        (
            *_geometry_rounds(
                single_elapsed_ns=1,
                small_elapsed_ns=1,
                post_jit=False,
            ),
            *_geometry_rounds(
                single_elapsed_ns=90_000_000,
                small_elapsed_ns=900_000_000,
            ),
            *_geometry_rounds(
                single_elapsed_ns=120_000_000,
                small_elapsed_ns=1_000_000_000,
            ),
        ),
        trailing_rounds=2,
    )

    assert result.upper_duration_ns_by_geometry_and_tier[4] == {
        "single": 120_000_000,
        "small": 1_000_000_000,
    }
    assert result.measured_capacity == 4

    with pytest.raises(ValueError, match=r"legal park|complete.*round"):
        _compile_profile(
            (
                *_geometry_rounds(
                    single_elapsed_ns=100_000_000,
                    small_elapsed_ns=800_000_000,
                )[:-1],
                _service_round(
                    tier_id="small",
                    active_population=4,
                    completed_legal_parks=3,
                    elapsed_ns=800_000_000,
                    geometry_id=4,
                    service_interval_ms=1_120,
                ),
            )
        )


def test_profile_refuses_missing_underfilled_or_extrapolated_tiers() -> None:
    """@spec PORT-PERF-006: a sparse numerical curve earns no authority."""

    compile_profile = _symbol("compile_provisional_service_profile")
    complete = _geometry_rounds(
        single_elapsed_ns=100_000_000,
        small_elapsed_ns=700_000_000,
    )
    with pytest.raises(
        ValueError,
        match=r"missing.*geometry.*4.*small|geometry.*4.*tier.*small",
    ):
        _compile_profile(complete[:-1])
    with pytest.raises(ValueError, match=r"maximum.*4|underfilled|active.*4"):
        _compile_profile(
            (
                *complete[:-1],
                _service_round(
                    tier_id="small",
                    active_population=3,
                    elapsed_ns=700_000_000,
                    geometry_id=4,
                    service_interval_ms=1_120,
                ),
            )
        )
    with pytest.raises(ValueError, match=r"population.*8|extrapolat|tier"):
        compile_profile(
            complete,
            execution_tiers=(
                _service_tier("single", 1),
                _service_tier("small", 4),
            ),
            max_population=8,
            reference_interval_ms=1_120,
            reference_geometry_id=4,
            admitted_geometry_ids=(0, 1, 2, 3, 4),
            trailing_rounds=1,
            derating_factor=Fraction(1, 2),
            context=_profile_context(allocated_pool=8, count_cap=8),
        )


def test_equal_row_rates_do_not_create_equal_session_capacity() -> None:
    """@spec PORT-PERF-006 / PORT-PERF-008: legal parks, not rows/s, gate."""

    fast = _compile_profile(
        _geometry_rounds(
            single_elapsed_ns=100_000_000,
            small_elapsed_ns=800_000_000,
            single_model_rows=1,
            small_model_rows=8,
        ),
        derating_factor=Fraction(1),
    )
    slow = _compile_profile(
        _geometry_rounds(
            single_elapsed_ns=100_000_000,
            small_elapsed_ns=1_200_000_000,
            single_model_rows=1,
            small_model_rows=12,
        ),
        derating_factor=Fraction(1),
    )

    assert fast.diagnostic_rows_per_second_by_geometry_and_tier[4][
        "small"
    ] == Fraction(10)
    assert slow.diagnostic_rows_per_second_by_geometry_and_tier[4][
        "small"
    ] == Fraction(10)
    assert fast.measured_capacity == 4
    assert slow.measured_capacity == 1


@pytest.mark.parametrize("marker", ["dummy_run", "is_profile"])
def test_service_capacity_never_uses_memory_profile_dummy(
    marker: str,
) -> None:
    """@spec PORT-MIG-005 / PORT-PERF-005: the two profile paths differ."""

    marked = {marker: True}
    rounds = list(
        _geometry_rounds(
            single_elapsed_ns=100_000_000,
            small_elapsed_ns=800_000_000,
        )
    )
    rounds[0] = _service_round(
        tier_id="single",
        active_population=1,
        elapsed_ns=100_000_000,
        service_interval_ms=80,
        geometry_id=0,
        **marked,
    )
    with pytest.raises(ValueError, match=r"service prim|dummy|profile"):
        _compile_profile(tuple(rounds))


def test_fixed_point_compiler_bounds_rounding_and_signed64() -> None:
    """@spec PORT-STATE-025 / PORT-PERF-008: precision is capacity-safe."""

    source = (
        (80, Fraction(1)),
        (160, Fraction(1, 2)),
        (320, Fraction(1, 4)),
        (560, Fraction(1, 4)),
        (1_120, Fraction(1, 4)),
    )
    compiled = _symbol("compile_service_demand_profile")(
        source,
        max_charged_population=1_000,
    )

    assert compiled.scale > 0
    assert compiled.scale & (compiled.scale - 1) == 0
    assert compiled.budget == compiled.scale
    assert compiled.intervals_ms == tuple(interval for interval, _ in source)
    assert all(isinstance(units, int) for units in compiled.demand_units)
    for (_, demand), units in zip(source, compiled.demand_units):
        assert Fraction(units, compiled.scale) >= demand
        assert Fraction(units, compiled.scale) - demand < Fraction(
            1,
            compiled.scale,
        )
    assert Fraction(1_000, compiled.scale) < Fraction(1, 4)
    assert 1_000 * max(compiled.demand_units) <= 2**63 - 1


def test_fixed_point_compiler_rejects_when_precision_and_range_conflict() -> None:
    """@spec PORT-STATE-025 / PORT-PERF-008: coarse safety is no fallback."""

    with pytest.raises(ValueError, match=r"scale|precision|signed-64"):
        _symbol("compile_service_demand_profile")(
            ((1_120, Fraction(1)),),
            max_charged_population=2**62,
        )


def test_provisional_mixed_transaction_repeats_homogeneous_setup() -> None:
    """@spec PORT-STATE-026 / PORT-PERF-008: no unproved separability."""

    tables = (
        (0, 100, 180),
        (0, 200, 350),
        (0, 300, 520),
        (0, 400, 700),
        (0, 500, 880),
    )
    duration = _symbol("fallback_transaction_duration_ns")(
        resident_counts_by_geometry=(1, 1, 0, 0, 0),
        homogeneous_upper_duration_ns_by_geometry=tables,
        mixed_composition_policy="homogeneous_upper_sum",
    )
    assert duration == 300

    with pytest.raises(ValueError, match=r"homogeneous|mixed"):
        _symbol("fallback_transaction_duration_ns")(
            resident_counts_by_geometry=(1, 1, 0, 0, 0),
            homogeneous_upper_duration_ns_by_geometry=tables,
            mixed_composition_policy="homogeneous_only",
        )


def test_profile_receipt_stamps_the_complete_compiled_authority() -> None:
    """@spec PORT-PERF-006 / PORT-INT-005: receipt identity is complete."""

    result = _compile_profile(
        _geometry_rounds(
            single_elapsed_ns=100_000_000,
            small_elapsed_ns=800_000_000,
        )
    )
    receipt = result.receipt

    assert receipt.compiler_version == "persistent-state-capacity-v2"
    assert receipt.reference_interval_ms == 1_120
    assert receipt.reference_geometry_id == 4
    assert receipt.execution_tier_maxima == {"single": 1, "small": 4}
    assert receipt.measured_upper_duration_ns_by_geometry_and_tier[4] == {
        "single": 100_000_000,
        "small": 800_000_000,
    }
    assert len(receipt.expanded_duration_table_sha256) == 64
    assert receipt.mixed_composition_policy == "homogeneous_upper_sum"
    assert receipt.service_demand_scale > 0
    assert len(receipt.service_demand_units) == 5
    assert receipt.aggregate_rounding_bound == Fraction(
        receipt.maximum_charged_population,
        receipt.service_demand_scale,
    )
    assert receipt.least_rational_demand > receipt.aggregate_rounding_bound
    assert receipt.provisional_capacity == 2
    assert receipt.derating_factor == Fraction(1, 2)
    assert len(receipt.receipt_sha256) == 64
    assert result.profile_candidate.qualified is False
    assert result.profile_candidate.installable is False


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
        return Fraction(1)

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
