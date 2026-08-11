# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for exact periodic mixed-cadence capacity."""

from __future__ import annotations

import importlib
import math
import random
from fractions import Fraction
from types import ModuleType
from typing import Any, NoReturn

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "vllm_omni.engine.persistent_state_capacity"
_INT64_MAX = 2**63 - 1


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _module() -> ModuleType:
    try:
        return importlib.import_module(_MODULE)
    except ModuleNotFoundError:
        _fail(f"PORT-STATE-026 missing {_MODULE}")


def _symbol(name: str, spec_id: str = "PORT-STATE-026") -> Any:
    try:
        return getattr(_module(), name)
    except AttributeError:
        _fail(f"{spec_id} missing {_MODULE}.{name}")


def _windows(intervals_ms: tuple[int, ...]) -> tuple[int, ...]:
    hyperperiod = math.lcm(*intervals_ms)
    return tuple(
        sorted({multiple for interval in intervals_ms for multiple in range(interval, hyperperiod + 1, interval)})
    )


def _dominance() -> str:
    return "d" * 64


def _compile(
    *,
    intervals_ms: tuple[int, ...] = (320, 1_120),
    tables_ms: tuple[tuple[int, ...], ...] = (
        (100, 180, 400),
        (100, 180, 1_200),
    ),
    derating_factor: Fraction = Fraction(1),
    control_ms: dict[int, tuple[int, ...]] | None = None,
    control_dominance_sha256: str | None = None,
    evidence_class: str = "probe",
) -> Any:
    return _compile_ns(
        intervals_ms=intervals_ms,
        tables_ns=tuple(tuple(value * 1_000_000 for value in table) for table in tables_ms),
        derating_factor=derating_factor,
        control_ms=control_ms,
        control_dominance_sha256=control_dominance_sha256,
        evidence_class=evidence_class,
    )


def _compile_ns(
    *,
    intervals_ms: tuple[int, ...],
    tables_ns: tuple[tuple[int, ...], ...],
    derating_factor: Fraction,
    control_ms: dict[int, tuple[int, ...]] | None = None,
    control_dominance_sha256: str | None = None,
    evidence_class: str = "probe",
) -> Any:
    search_population = max(len(table) for table in tables_ns)
    tiers = tuple(
        _symbol("ServiceExecutionTier", "PORT-PERF-006")(
            tier_id=f"population-{population}",
            max_active_population=population,
        )
        for population in range(1, search_population + 1)
    )
    executions = tuple(
        _symbol("ServiceRoundExecution", "PORT-PERF-006")(
            scenario_id="ordinary",
            tier_id=f"population-{population}",
            active_population=population,
            elapsed_ns=table[population - 1],
            service_interval_ms=interval,
            geometry_id=geometry_id,
            completed_legal_parks=population,
            completed_model_rows=None,
            post_jit=True,
            continuously_loaded=True,
            dummy_run=False,
            is_profile=False,
        )
        for geometry_id, (interval, table) in enumerate(zip(intervals_ms, tables_ns))
        for population in range(1, search_population + 1)
    )
    control_ns = (
        None
        if control_ms is None
        else {
            window_ms * 1_000_000: tuple(value * 1_000_000 for value in table)
            for window_ms, table in control_ms.items()
        }
    )
    context = _symbol("ServiceProfileContext", "PORT-PERF-006")(
        pre_override_physical_bound=search_population + 3,
        allocated_pool=search_population,
        count_cap=search_population,
        execution_claim_ceiling=search_population,
        service_budget_source="measured_fallback",
        service_budget_coefficients=(),
        derating_factor=derating_factor,
        slot_bytes=6_314_936,
        execution_environment_key="periodic-capacity-test",
        precision_policy="fp32",
        state_profile="nemotron-asr-fp32-v1",
        compiler_version="persistent-state-capacity-v3",
        mixed_composition_policy="periodic_limited_preemption_edf",
    )
    kwargs = {
        "execution_tiers": tiers,
        "max_population": search_population,
        "reference_interval_ms": intervals_ms[-1],
        "reference_geometry_id": len(intervals_ms) - 1,
        "admitted_geometry_ids": tuple(range(len(intervals_ms))),
        "trailing_rounds": 1,
        "derating_factor": derating_factor,
        "context": context,
        "control_upper_ns_by_window_and_population": control_ns,
        "control_dominance_sha256": (
            _dominance() if control_ms is None and control_dominance_sha256 is None else control_dominance_sha256
        ),
        "evidence_class": evidence_class,
    }
    try:
        return _symbol("compile_provisional_service_profile", "PORT-PERF-006")(
            executions,
            **kwargs,
        )
    except TypeError as error:
        _fail(f"PORT-STATE-026 missing periodic arguments on the existing service-profile compiler: {error}")


def _compile_with_canaries(
    *,
    ordinary_ns: int = 100,
    forced_ns: int = 200,
    terminal_ns: int = 200,
    omit_scenario: str | None = None,
    completed_legal_parks: int = 2,
    derating_factor: Fraction = Fraction(1, 2),
) -> Any:
    execution_type = _symbol("ServiceRoundExecution", "PORT-PERF-006")
    executions = [
        execution_type(
            scenario_id="ordinary",
            tier_id="single",
            active_population=1,
            elapsed_ns=ordinary_ns,
            service_interval_ms=80,
            geometry_id=0,
            completed_legal_parks=1,
            completed_model_rows=None,
            post_jit=True,
            continuously_loaded=True,
            dummy_run=False,
            is_profile=False,
        )
    ]
    for scenario_id, elapsed_ns in (
        ("forced_eou_then_chunk", forced_ns),
        ("final_tail_then_flush", terminal_ns),
    ):
        if scenario_id == omit_scenario:
            continue
        executions.extend(
            (
                execution_type(
                    scenario_id=scenario_id,
                    tier_id="single",
                    active_population=1,
                    elapsed_ns=10_000,
                    service_interval_ms=80,
                    geometry_id=0,
                    completed_legal_parks=completed_legal_parks,
                    completed_model_rows=None,
                    post_jit=False,
                    continuously_loaded=True,
                    dummy_run=False,
                    is_profile=False,
                ),
                execution_type(
                    scenario_id=scenario_id,
                    tier_id="single",
                    active_population=1,
                    elapsed_ns=elapsed_ns,
                    service_interval_ms=80,
                    geometry_id=0,
                    completed_legal_parks=completed_legal_parks,
                    completed_model_rows=None,
                    post_jit=True,
                    continuously_loaded=True,
                    dummy_run=False,
                    is_profile=False,
                ),
            )
        )
    context = _symbol("ServiceProfileContext", "PORT-PERF-006")(
        pre_override_physical_bound=2,
        allocated_pool=1,
        count_cap=1,
        execution_claim_ceiling=1,
        service_budget_source="measured_fallback",
        service_budget_coefficients=(),
        derating_factor=derating_factor,
        slot_bytes=6_314_936,
        execution_environment_key="control-canary-test",
        precision_policy="fp32",
        state_profile="nemotron-asr-fp32-v1",
        compiler_version="persistent-state-capacity-v3",
        mixed_composition_policy="periodic_limited_preemption_edf",
    )
    return _symbol("compile_provisional_service_profile", "PORT-PERF-006")(
        tuple(executions),
        execution_tiers=(
            _symbol("ServiceExecutionTier", "PORT-PERF-006")(
                tier_id="single",
                max_active_population=1,
            ),
        ),
        max_population=1,
        reference_interval_ms=80,
        reference_geometry_id=0,
        admitted_geometry_ids=(0,),
        trailing_rounds=1,
        derating_factor=derating_factor,
        context=context,
        control_upper_ns_by_window_and_population=None,
        control_dominance_sha256=None,
        evidence_class="probe",
    )


def _compile_with_bulk_canary(
    *,
    single_ordinary_ns: int = 50_000_000,
    bulk_ordinary_ns: int = 397_265_692,
    bulk_terminal_ns: int = 823_425_490,
) -> Any:
    execution_type = _symbol("ServiceRoundExecution", "PORT-PERF-006")
    tiers = (
        _symbol("ServiceExecutionTier", "PORT-PERF-006")(
            tier_id="single",
            max_active_population=1,
        ),
        _symbol("ServiceExecutionTier", "PORT-PERF-006")(
            tier_id="eager-bulk",
            max_active_population=8,
        ),
    )
    executions = []
    single_canary_ns = 2 * single_ordinary_ns
    for tier, ordinary_ns, forced_ns, terminal_ns in (
        (
            tiers[0],
            single_ordinary_ns,
            single_canary_ns,
            single_canary_ns,
        ),
        (tiers[1], bulk_ordinary_ns, 150_000_000, bulk_terminal_ns),
    ):
        executions.append(
            execution_type(
                scenario_id="ordinary",
                tier_id=tier.tier_id,
                active_population=tier.max_active_population,
                elapsed_ns=ordinary_ns,
                service_interval_ms=320,
                geometry_id=0,
                completed_legal_parks=tier.max_active_population,
                completed_model_rows=None,
                post_jit=True,
                continuously_loaded=True,
                dummy_run=False,
                is_profile=False,
            )
        )
        for scenario_id, elapsed_ns in (
            ("forced_eou_then_chunk", forced_ns),
            ("final_tail_then_flush", terminal_ns),
        ):
            executions.append(
                execution_type(
                    scenario_id=scenario_id,
                    tier_id=tier.tier_id,
                    active_population=tier.max_active_population,
                    elapsed_ns=elapsed_ns,
                    service_interval_ms=320,
                    geometry_id=0,
                    completed_legal_parks=(
                        2 * tier.max_active_population
                    ),
                    completed_model_rows=None,
                    post_jit=True,
                    continuously_loaded=True,
                    dummy_run=False,
                    is_profile=False,
                )
            )
    context = _symbol("ServiceProfileContext", "PORT-PERF-006")(
        pre_override_physical_bound=8,
        allocated_pool=8,
        count_cap=8,
        execution_claim_ceiling=8,
        service_budget_source="measured_fallback",
        service_budget_coefficients=(),
        derating_factor=Fraction(1, 2),
        slot_bytes=6_314_936,
        execution_environment_key="irrelevant-bulk-canary-test",
        precision_policy="fp32",
        state_profile="nemotron-asr-fp32-v1",
        compiler_version="persistent-state-capacity-v3",
        mixed_composition_policy="periodic_limited_preemption_edf",
    )
    return _symbol(
        "compile_provisional_service_profile",
        "PORT-PERF-006",
    )(
        tuple(executions),
        execution_tiers=tiers,
        max_population=8,
        reference_interval_ms=320,
        reference_geometry_id=0,
        admitted_geometry_ids=(0,),
        trailing_rounds=1,
        derating_factor=Fraction(1, 2),
        context=context,
        control_upper_ns_by_window_and_population=None,
        control_dominance_sha256=None,
        evidence_class="probe",
    )


def _evaluate(profile: Any, *counts: int) -> Any:
    intervals = tuple(profile.service_interval_ms_by_geometry.values())
    return _symbol("evaluate_periodic_schedulability")(
        profile=profile,
        population_by_interval=dict(zip(intervals, counts)),
    )


def _fragmentation(table: tuple[int, ...]) -> tuple[int, ...]:
    result = [0]
    for population in range(1, len(table) + 1):
        result.append(max(result[population - bucket] + table[bucket - 1] for bucket in range(1, population + 1)))
    return tuple(result[1:])


def _independent_schedulable(
    *,
    intervals_ms: tuple[int, ...],
    tables_ms: tuple[tuple[int, ...], ...],
    counts: tuple[int, ...],
    derating_factor: Fraction,
    use_sum_blocking: bool = True,
) -> bool:
    fragmentation = tuple(_fragmentation(table) for table in tables_ms)
    numerator = derating_factor.numerator
    denominator = derating_factor.denominator
    for window in _windows(intervals_ms):
        demand = sum(
            (window // interval) * table[count - 1]
            for interval, table, count in zip(intervals_ms, fragmentation, counts)
            if count
        )
        if demand == 0:
            continue
        blockers = [
            table[count - 1]
            for interval, table, count in zip(intervals_ms, tables_ms, counts)
            if count and interval > window
        ]
        blocking = sum(blockers) if use_sum_blocking else max(blockers, default=0)
        if denominator * (demand + blocking) > numerator * window:
            return False
    return True


def test_compiler_builds_fragmentation_hyperperiod_and_coherent_frontiers() -> None:
    """@spec PORT-STATE-026 / PORT-PERF-006: one compiled authority."""

    profile = _compile()

    assert profile.fragmentation_duration_ns_by_geometry_and_population == {
        0: (100_000_000, 200_000_000, 400_000_000),
        1: (100_000_000, 200_000_000, 1_200_000_000),
    }
    assert profile.hyperperiod_ns == 2_240_000_000
    assert profile.window_ns == tuple(window * 1_000_000 for window in _windows((320, 1_120)))
    assert profile.aligned_frontier_by_interval == {320: 2, 1_120: 2}
    assert profile.fragmentation_frontier_by_interval == {320: 2, 1_120: 2}
    assert profile.homogeneous_capacity_by_interval == {320: 2, 1_120: 2}


def test_table_proven_mix_is_not_rejected_by_the_old_additive_ratio() -> None:
    """@spec PORT-STATE-026: exact tables, never sum(n[g]/C[g])."""

    profile = _compile()
    result = _evaluate(profile, 2, 1)

    assert Fraction(2, 2) + Fraction(1, 2) == Fraction(3, 2)
    assert result.schedulable is True
    assert result.binding_window_ns is None
    assert result.charged_demand_ratio == Fraction(5, 7)

    projection = _symbol("project_fixed_dispatch_capacity")(
        profile=profile,
        inventory={
            "resident_count": 2,
            "effective_capacity": 3,
            "configured_limit": 3,
        },
        resident_counts_by_interval={320: 2, 1_120: 0},
        submitted_counts_by_interval={320: 0, 1_120: 0},
        authority_open=True,
        admission_policy="profile",
    )
    assert projection.nominal_dispatchable_by_interval[1_120] == 1


def test_hard_cap_dispatches_to_hard_ceiling_without_rewriting_profile() -> None:
    """@spec PORT-STATE-026 / PORT-OBS-012: characterize, do not lie."""
    profile = _compile(
        intervals_ms=(320,),
        tables_ms=((200,),),
        derating_factor=Fraction(1, 2),
    )
    empty = {320: 0}
    inventory = {
        "resident_count": 0,
        "effective_capacity": 4,
        "configured_limit": 4,
    }
    project = _symbol("project_fixed_dispatch_capacity")

    guarded = project(
        profile=profile,
        inventory=inventory,
        resident_counts_by_interval=empty,
        submitted_counts_by_interval=empty,
        authority_open=True,
        admission_policy="profile",
    )
    characterization = project(
        profile=profile,
        inventory=inventory,
        resident_counts_by_interval=empty,
        submitted_counts_by_interval=empty,
        authority_open=True,
        admission_policy="hard_cap",
    )

    assert guarded.candidate_supported_by_interval == {320: False}
    assert guarded.dispatchable_by_interval == {320: 0}
    assert characterization.candidate_supported_by_interval == {320: True}
    assert characterization.dispatchable_by_interval == {320: 1}
    assert characterization.nominal_dispatchable_by_interval == {320: 0}
    assert characterization.hard_headroom == 1
    assert characterization.admission_policy == "hard_cap"

    at_search_ceiling = project(
        profile=profile,
        inventory={**inventory, "resident_count": 1},
        resident_counts_by_interval={320: 1},
        submitted_counts_by_interval=empty,
        authority_open=True,
        admission_policy="hard_cap",
    )
    assert at_search_ceiling.hard_headroom == 0
    assert at_search_ceiling.dispatchable_by_interval == {320: 0}


def test_dispatch_projection_rejects_unknown_admission_policy() -> None:
    """@spec PORT-STATE-026: no third or implicit serving policy."""
    profile = _compile(intervals_ms=(320,), tables_ms=((10,),))

    with pytest.raises(ValueError, match="admission policy"):
        _symbol("project_fixed_dispatch_capacity")(
            profile=profile,
            inventory={
                "resident_count": 0,
                "effective_capacity": 1,
                "configured_limit": 1,
            },
            resident_counts_by_interval={320: 0},
            submitted_counts_by_interval={320: 0},
            authority_open=True,
            admission_policy="unsafe",
        )


def test_dense_short_cadence_and_long_cadence_mix_uses_measured_work() -> None:
    """@spec PORT-STATE-026: streams are not audio-duration packing.

    This is a model-contract fixture with deliberately non-physical duration
    tables. It proves only that table-established work governs a mixed vector;
    it is not hardware-capacity evidence for this population.
    """

    profile = _compile(
        intervals_ms=(80, 1_120),
        tables_ms=(
            tuple(2 * count for count in range(1, 16)),
            tuple(20 * count for count in range(1, 16)),
        ),
    )

    assert _evaluate(profile, 14, 1).schedulable is True


def test_zero_demand_window_never_invents_short_cadence_blocking() -> None:
    """@spec PORT-STATE-026: pure long cadence has no ghost 80-ms gate."""

    profile = _compile(
        intervals_ms=(80, 1_120),
        tables_ms=((20,), (900,)),
    )
    result = _evaluate(profile, 0, 1)

    assert result.schedulable is True
    assert result.checked_window_count == 1


def test_selected_long_bucket_suffix_is_summed_not_maximized() -> None:
    """@spec PORT-STATE-026: synchronous bucket loop charges every suffix."""

    intervals = (80, 320, 1_120)
    # Three rows are admitted, so P_search covers the aggregate population
    # even though each selected geometry bucket has population one.
    tables = ((10, 10, 10), (40, 40, 40), (40, 40, 40))
    assert _independent_schedulable(
        intervals_ms=intervals,
        tables_ms=tables,
        counts=(1, 1, 1),
        derating_factor=Fraction(1),
        use_sum_blocking=False,
    )
    assert not _independent_schedulable(
        intervals_ms=intervals,
        tables_ms=tables,
        counts=(1, 1, 1),
        derating_factor=Fraction(1),
        use_sum_blocking=True,
    )

    result = _evaluate(
        _compile(intervals_ms=intervals, tables_ms=tables),
        1,
        1,
        1,
    )
    assert result.schedulable is False
    assert result.binding_window_ns == 80_000_000
    assert result.blocking_ns == 80_000_000


@pytest.mark.parametrize(
    ("duration_ns", "expected"),
    [(53_333_333, True), (53_333_334, False)],
)
def test_derating_uses_exact_cross_multiplication(
    duration_ns: int,
    expected: bool,
) -> None:
    """@spec PORT-STATE-025 / PORT-STATE-026: no float boundary drift."""

    profile = _compile_ns(
        intervals_ms=(80,),
        tables_ns=((duration_ns,),),
        derating_factor=Fraction(2, 3),
    )

    assert _evaluate(profile, 1).schedulable is expected


def test_control_work_is_not_free_or_double_counted() -> None:
    """@spec PORT-STATE-026: selected control excess has one authority."""

    without_control = _compile(
        intervals_ms=(80,),
        tables_ms=((80,),),
    )
    with_control = _compile(
        intervals_ms=(80,),
        tables_ms=((80,),),
        control_ms={80: (1,)},
        control_dominance_sha256=None,
    )

    assert _evaluate(without_control, 1).schedulable is True
    controlled = _evaluate(with_control, 1)
    assert controlled.schedulable is False
    assert controlled.control_ns == 1_000_000
    with pytest.raises(ValueError, match="control|dominance"):
        _compile_ns(
            intervals_ms=(80,),
            tables_ns=((1,),),
            derating_factor=Fraction(1),
            control_dominance_sha256=None,
            control_ms={},
        )


def test_exceptional_control_canaries_derive_zero_control_identity() -> None:
    """@spec PORT-STATE-026 / PORT-PERF-006: measured margin is authority."""

    profile = _compile_with_canaries()

    assert profile.control_upper_ns_by_window_and_population is None
    assert profile.control_dominance_sha256 is not None
    assert len(profile.control_dominance_sha256) == 64
    assert profile.upper_duration_ns_by_geometry_and_population[0] == (100,)
    assert profile.receipt.control_dominance_evidence["version"] == ("derated-control-dominance-v2")
    assert len(profile.receipt.control_dominance_evidence["cells"]) == 2


def test_infeasible_bulk_canary_cannot_block_supported_single_service() -> None:
    """@spec PORT-PERF-006: irrelevant stress cells cannot close readiness."""

    profile = _compile_with_bulk_canary()

    assert profile.homogeneous_capacity_by_interval[320] == 1
    cells = profile.receipt.control_dominance_evidence["cells"]
    terminal_bulk = next(
        cell
        for cell in cells
        if cell["scenario_id"] == "final_tail_then_flush"
        and cell["tier_id"] == "eager-bulk"
    )
    assert terminal_bulk["dominance_passed"] is False
    assert terminal_bulk["admission_relevant"] is False
    assert terminal_bulk["ordinary_feasible_populations"] == ()


def test_bulk_canary_still_gates_a_partly_feasible_population_step() -> None:
    """@spec PORT-PERF-006: relevance covers the tier's entire step."""

    with pytest.raises(ValueError, match="final_tail_then_flush|dominance"):
        _compile_with_bulk_canary(
            single_ordinary_ns=30_000_000,
            bulk_ordinary_ns=100_000_000,
            bulk_terminal_ns=201_000_000,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"forced_ns": 201}, "forced_eou_then_chunk|dominance|201"),
        ({"terminal_ns": 201}, "final_tail_then_flush|dominance|201"),
        ({"omit_scenario": "forced_eou_then_chunk"}, "forced_eou_then_chunk|missing"),
        ({"completed_legal_parks": 1}, "park|two"),
        (
            {
                "forced_ns": _INT64_MAX,
                "derating_factor": Fraction(2, 3),
            },
            "signed-64|overflow|dominance",
        ),
    ],
)
def test_exceptional_control_canaries_fail_closed(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """@spec PORT-PERF-006 / PORT-PERF-008: no missing or weak proof."""

    with pytest.raises(ValueError, match=match):
        _compile_with_canaries(**kwargs)


def test_exceptional_control_leading_round_is_not_the_upper() -> None:
    """@spec PORT-PERF-006: JIT-establishing canaries remain excluded."""

    profile = _compile_with_canaries()

    assert profile.control_dominance_sha256 is not None


def test_probe_evidence_cannot_claim_qualified_service_support() -> None:
    """@spec PORT-PERF-006 / PORT-INT-005: silence stays provisional."""

    profile = _compile(evidence_class="probe")

    assert profile.evidence_class == "probe"
    assert profile.qualified is False
    assert all(supported is False for supported in profile.qualified_support_by_interval.values())
    with pytest.raises(ValueError, match="public.speech|qualified|evidence"):
        _compile(evidence_class="qualified")


def test_receipt_stamps_the_exact_periodic_authority() -> None:
    """@spec PORT-PERF-006 / PORT-INT-005: evidence is reconstructable."""

    profile = _compile(derating_factor=Fraction(2, 3))
    receipt = profile.receipt

    assert receipt.evidence_class == "probe"
    assert receipt.derating_factor == Fraction(2, 3)
    assert receipt.hyperperiod_ns == profile.hyperperiod_ns
    assert receipt.window_ns == profile.window_ns
    assert receipt.control_dominance_sha256 == _dominance()
    assert len(receipt.fragmentation_table_sha256) == 64
    assert receipt.homogeneous_capacity_by_interval == (profile.homogeneous_capacity_by_interval)


@pytest.mark.parametrize(
    ("intervals_ms", "tables_ns", "factor", "match"),
    [
        (
            (2**62, 2**62 - 1),
            {0: (1,), 1: (1,)},
            Fraction(1),
            "lcm|hyperperiod|signed-64|overflow",
        ),
        (
            (1,),
            {0: (_INT64_MAX, _INT64_MAX)},
            Fraction(1),
            "fragment|add|signed-64|overflow",
        ),
        (
            (1,),
            {0: (1,)},
            Fraction(1, 2**63),
            "multiply|signed-64|overflow|derating",
        ),
    ],
)
def test_compiler_fails_closed_before_checked_integer_overflow(
    intervals_ms: tuple[int, ...],
    tables_ns: dict[int, tuple[int, ...]],
    factor: Fraction,
    match: str,
) -> None:
    """@spec PORT-STATE-025 / PORT-PERF-008: checked positive int64."""

    with pytest.raises(ValueError, match=match):
        _compile_ns(
            intervals_ms=intervals_ms,
            tables_ns=tuple(tables_ns.values()),
            derating_factor=factor,
        )


def test_population_above_search_coverage_fails_without_table_indexing() -> None:
    """@spec PORT-STATE-026: physical slots never invent evidence."""

    result = _evaluate(_compile(), 4, 0)

    assert result.schedulable is False
    assert result.binding_authority == "search_population"


def test_schedulability_is_componentwise_monotone_through_population_1000() -> None:
    """@spec PORT-PERF-008: randomized monotonicity at target scale."""

    population = 1_000
    profile = _compile(
        intervals_ms=(80, 320, 1_120),
        tables_ms=tuple(tuple(per_row * count for count in range(1, population + 1)) for per_row in (4, 10, 20)),
    )
    generator = random.Random(5212)
    for _ in range(256):
        base = tuple(generator.randrange(0, 251) for _ in range(3))
        growth = tuple(generator.randrange(0, 251) for _ in range(3))
        larger = tuple(left + right for left, right in zip(base, growth))
        if sum(larger) > population:
            continue
        if not _evaluate(profile, *base).schedulable:
            assert not _evaluate(profile, *larger).schedulable


def test_headroom_uses_bounded_bisection_at_population_1000() -> None:
    """@spec PORT-OBS-012 / PORT-PERF-008: no per-slot hot-path loop."""

    population = 1_000
    profile = _compile(
        intervals_ms=(1_120,),
        tables_ms=(tuple(count for count in range(1, population + 1)),),
    )
    result = _symbol(
        "periodic_admission_headroom",
        "PORT-OBS-012",
    )(
        profile=profile,
        population_by_interval={1_120: 0},
        candidate_interval_ms=1_120,
        hard_headroom=population,
    )

    assert result.headroom == population
    assert result.predicate_evaluations <= 11
