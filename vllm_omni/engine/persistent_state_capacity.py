# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deployment-resolved capacity for manager-backed persistent state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import floor
from typing import Protocol

_INT64_MAX = 2**63 - 1
_SERVICE_INTERVALS_MS = (80, 160, 320, 560, 1_120)


class ServiceDemand(Protocol):
    """A deployment-local service-demand model."""

    def at(self, service_interval_ms: int) -> Fraction: ...


@dataclass(frozen=True)
class PoolCapacityResolution:
    profiled_block_bound: int
    allocated_total_blocks: int
    allocated_real_slots: int
    resolved_count_limit: int
    count_was_clamped: bool
    page_size_bytes: int


# @spec PORT-STATE-004
def resolve_pool_capacity(
    *,
    profiled_block_bound: int,
    safety_reserve_slots: int,
    max_resident_sessions: int | None,
    num_gpu_blocks_override: int | None,
    page_size_bytes: int,
) -> PoolCapacityResolution:
    """Resolve physical allocation separately from logical admission policy."""
    if profiled_block_bound <= 0:
        raise ValueError("profiled block bound must be positive")
    if safety_reserve_slots < 0:
        raise ValueError("safety reserve slots must be non-negative")
    if page_size_bytes <= 0:
        raise ValueError("persistent-state page size must be positive")
    if max_resident_sessions is not None and max_resident_sessions <= 0:
        raise ValueError("max resident sessions must be positive when set")
    if num_gpu_blocks_override is not None:
        if num_gpu_blocks_override <= 0:
            raise ValueError("num_gpu_blocks_override must be positive")
        if num_gpu_blocks_override > profiled_block_bound:
            raise ValueError(
                "requested num_gpu_blocks_override "
                f"{num_gpu_blocks_override} exceeds profiled bound "
                f"{profiled_block_bound} for {page_size_bytes}-byte slots"
            )

    fixed_blocks = safety_reserve_slots + 1
    available_total = (
        profiled_block_bound
        if num_gpu_blocks_override is None
        else num_gpu_blocks_override
    )
    available_real = available_total - fixed_blocks
    if available_real < 1:
        raise ValueError(
            "profiled persistent-state capacity cannot fit null block, "
            "safety reserve, and one real slot: "
            f"bound={available_total}, page_size_bytes={page_size_bytes}"
        )

    requested_count = max_resident_sessions or available_real
    resolved_count = min(requested_count, available_real)
    allocated_total = available_total
    if num_gpu_blocks_override is None and requested_count <= available_real:
        allocated_total = requested_count + fixed_blocks
    return PoolCapacityResolution(
        profiled_block_bound=profiled_block_bound,
        allocated_total_blocks=allocated_total,
        allocated_real_slots=allocated_total - fixed_blocks,
        resolved_count_limit=resolved_count,
        count_was_clamped=requested_count > available_real,
        page_size_bytes=page_size_bytes,
    )


@dataclass(frozen=True)
class OnePointServiceDemand:
    reference_interval_ms: int
    reference_demand: Fraction

    def __post_init__(self) -> None:
        if self.reference_interval_ms <= 0:
            raise ValueError("reference interval must be positive")
        if self.reference_demand <= 0:
            raise ValueError("reference demand must be positive")

    # @spec PORT-STATE-025
    def at(self, service_interval_ms: int) -> Fraction:
        if service_interval_ms <= 0:
            raise ValueError("service interval must be positive")
        scale = max(
            Fraction(1),
            Fraction(self.reference_interval_ms, service_interval_ms),
        )
        return self.reference_demand * scale


@dataclass(frozen=True)
class TwoTermServiceDemand:
    invocation_demand_ms: Fraction
    audio_rate_demand: Fraction

    def __post_init__(self) -> None:
        if self.invocation_demand_ms < 0 or self.audio_rate_demand < 0:
            raise ValueError("service-demand coefficients must be non-negative")

    # @spec PORT-STATE-025
    def at(self, service_interval_ms: int) -> Fraction:
        if service_interval_ms <= 0:
            raise ValueError("service interval must be positive")
        return (
            self.invocation_demand_ms / service_interval_ms
            + self.audio_rate_demand
        )


@dataclass(frozen=True)
class ProvisionalServiceProfile:
    capacity: int
    reference_interval_ms: int
    reference_demand: Fraction
    source: str = "measured_fallback"
    qualified: bool = False


# @spec PORT-STATE-025, PORT-PERF-006
def derive_provisional_profile(
    *,
    measured_sessions: int,
    derating_factor: Fraction,
    reference_interval_ms: int,
) -> ProvisionalServiceProfile:
    if measured_sessions <= 0:
        raise ValueError("measured sessions must be positive")
    if derating_factor <= 0:
        raise ValueError("derating factor must be positive")
    if reference_interval_ms <= 0:
        raise ValueError("reference interval must be positive")
    capacity = max(1, floor(measured_sessions * derating_factor))
    return ProvisionalServiceProfile(
        capacity=capacity,
        reference_interval_ms=reference_interval_ms,
        reference_demand=Fraction(1, capacity),
    )


@dataclass(frozen=True)
class CompiledServiceDemandProfile:
    intervals_ms: tuple[int, ...]
    demand_units: tuple[int, ...]
    scale: int
    budget: int
    maximum_charged_population: int
    least_rational_demand: Fraction
    aggregate_rounding_bound: Fraction

    def units_at(self, service_interval_ms: int) -> int:
        try:
            index = self.intervals_ms.index(service_interval_ms)
        except ValueError as exc:
            raise ValueError(
                f"unsupported service interval {service_interval_ms}"
            ) from exc
        return self.demand_units[index]


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


# @spec PORT-STATE-025, PORT-PERF-008
def compile_service_demand_profile(
    source: Sequence[tuple[int, Fraction]],
    *,
    max_charged_population: int,
) -> CompiledServiceDemandProfile:
    """Compile rational demand into a signed-64-safe fixed-point profile."""
    if max_charged_population <= 0:
        raise ValueError("maximum charged population must be positive")
    if not source:
        raise ValueError("service-demand source cannot be empty")
    intervals = tuple(interval for interval, _ in source)
    demands = tuple(demand for _, demand in source)
    if len(set(intervals)) != len(intervals) or any(x <= 0 for x in intervals):
        raise ValueError("service intervals must be unique and positive")
    if any(demand <= 0 for demand in demands):
        raise ValueError("service demands must be positive")
    least_demand = min(demands)

    for exponent in range(62, -1, -1):
        scale = 1 << exponent
        units = tuple(_ceil_fraction(scale * demand) for demand in demands)
        if max_charged_population * max(units) > _INT64_MAX:
            continue
        if Fraction(max_charged_population, scale) >= least_demand:
            continue
        return CompiledServiceDemandProfile(
            intervals_ms=intervals,
            demand_units=units,
            scale=scale,
            budget=scale,
            maximum_charged_population=max_charged_population,
            least_rational_demand=least_demand,
            aggregate_rounding_bound=Fraction(
                max_charged_population,
                scale,
            ),
        )
    raise ValueError(
        "no power-of-two scale satisfies signed-64 range and precision"
    )


@dataclass(frozen=True)
class ServiceExecutionTier:
    tier_id: str
    max_active_population: int

    def __post_init__(self) -> None:
        if not self.tier_id or self.max_active_population <= 0:
            raise ValueError("service execution tier must be named and positive")


@dataclass(frozen=True)
class ServiceRoundExecution:
    tier_id: str
    active_population: int
    elapsed_ns: int
    service_interval_ms: int
    geometry_id: int
    completed_legal_parks: int
    completed_model_rows: int | None
    post_jit: bool
    continuously_loaded: bool
    dummy_run: bool
    is_profile: bool

    def __post_init__(self) -> None:
        if self.active_population <= 0 or self.elapsed_ns <= 0:
            raise ValueError("service-round population and duration must be positive")
        if self.service_interval_ms <= 0 or self.geometry_id < 0:
            raise ValueError("service-round geometry must be valid")


@dataclass(frozen=True)
class ServiceProfileContext:
    pre_override_physical_bound: int
    allocated_pool: int
    count_cap: int | None
    execution_claim_ceiling: int
    service_budget_source: str
    service_budget_coefficients: tuple[Fraction, ...]
    derating_factor: Fraction | None
    slot_bytes: int
    execution_environment_key: str
    precision_policy: str
    state_profile: str
    compiler_version: str
    mixed_composition_policy: str


@dataclass(frozen=True)
class ServiceProfileCandidate:
    execution_environment_key: str
    precision_policy: str
    state_profile: str
    qualified: bool = False
    installable: bool = False


@dataclass(frozen=True)
class ServiceProfileReceipt:
    compiler_version: str
    reference_interval_ms: int
    reference_geometry_id: int
    execution_tier_maxima: Mapping[str, int]
    measured_upper_duration_ns_by_geometry_and_tier: Mapping[
        int, Mapping[str, int]
    ]
    expanded_duration_table_sha256: str
    mixed_composition_policy: str
    service_demand_scale: int
    service_demand_units: tuple[int, ...]
    aggregate_rounding_bound: Fraction
    maximum_charged_population: int
    least_rational_demand: Fraction
    provisional_capacity: int
    derating_factor: Fraction
    pre_override_physical_bound: int
    allocated_pool: int
    count_cap: int | None
    execution_claim_ceiling: int
    service_budget_source: str
    service_budget_coefficients: tuple[Fraction, ...]
    slot_bytes: int
    execution_environment_key: str
    precision_policy: str
    state_profile: str
    receipt_sha256: str


@dataclass(frozen=True)
class StartupServiceProfile:
    upper_duration_ns_by_geometry_and_population: Mapping[int, tuple[int, ...]]
    upper_duration_ns_by_geometry_and_tier: Mapping[int, Mapping[str, int]]
    diagnostic_rows_per_second_by_geometry_and_tier: Mapping[
        int, Mapping[str, Fraction]
    ]
    measured_capacity: int
    provisional_capacity: int
    reference_demand: Fraction
    compiled_demand: CompiledServiceDemandProfile
    profile_candidate: ServiceProfileCandidate
    receipt: ServiceProfileReceipt


def _hash_json(value: object) -> str:
    encoded = json.dumps(
        value,
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_tiers(
    tiers: Sequence[ServiceExecutionTier],
    max_population: int,
) -> tuple[ServiceExecutionTier, ...]:
    ordered = tuple(tiers)
    if not ordered:
        raise ValueError("at least one execution tier is required")
    maxima = tuple(tier.max_active_population for tier in ordered)
    if any(left >= right for left, right in zip(maxima, maxima[1:])):
        raise ValueError("execution tier maxima must be strictly increasing")
    if maxima[-1] < max_population:
        raise ValueError(
            f"population {max_population} requires measured execution tier"
        )
    return ordered


# @spec PORT-PERF-006, PORT-PERF-008
def compile_provisional_service_profile(
    executions: Sequence[ServiceRoundExecution],
    *,
    execution_tiers: Sequence[ServiceExecutionTier],
    max_population: int,
    reference_interval_ms: int,
    reference_geometry_id: int,
    admitted_geometry_ids: Sequence[int],
    trailing_rounds: int,
    derating_factor: Fraction,
    context: ServiceProfileContext,
) -> StartupServiceProfile:
    """Compile complete post-JIT legal-park rounds into startup authority."""
    if max_population <= 0 or trailing_rounds <= 0:
        raise ValueError("population and trailing-round counts must be positive")
    if derating_factor <= 0:
        raise ValueError("derating factor must be positive")
    tiers = _validate_tiers(execution_tiers, max_population)
    geometries = tuple(admitted_geometry_ids)
    if len(set(geometries)) != len(geometries) or not geometries:
        raise ValueError("admitted geometries must be unique and non-empty")
    if reference_geometry_id not in geometries:
        raise ValueError("reference geometry must be admitted")
    samples = tuple(executions)
    if any(sample.dummy_run or sample.is_profile for sample in samples):
        raise ValueError("service priming cannot use dummy/profile executions")

    upper_by_tier: dict[int, dict[str, int]] = {}
    row_rates: dict[int, dict[str, Fraction]] = {}
    geometry_intervals: dict[int, int] = {}
    for geometry_id in geometries:
        upper_by_tier[geometry_id] = {}
        row_rates[geometry_id] = {}
        for tier in tiers:
            matching = tuple(
                sample
                for sample in samples
                if sample.geometry_id == geometry_id
                and sample.tier_id == tier.tier_id
                and sample.active_population == tier.max_active_population
                and sample.post_jit
            )
            if len(matching) < trailing_rounds:
                underfilled = any(
                    sample.geometry_id == geometry_id
                    and sample.tier_id == tier.tier_id
                    for sample in samples
                )
                if underfilled:
                    raise ValueError(
                        f"geometry {geometry_id} tier {tier.tier_id} is "
                        f"underfilled; active population must equal maximum "
                        f"{tier.max_active_population}"
                    )
                raise ValueError(
                    f"missing geometry {geometry_id} tier {tier.tier_id}"
                )
            trailing = matching[-trailing_rounds:]
            if any(not sample.continuously_loaded for sample in trailing):
                raise ValueError("service priming rounds must be continuously loaded")
            if any(
                sample.completed_legal_parks != sample.active_population
                for sample in trailing
            ):
                raise ValueError("each complete round must reach one legal park per lease")
            intervals = {sample.service_interval_ms for sample in trailing}
            if len(intervals) != 1:
                raise ValueError("geometry has inconsistent service intervals")
            interval = next(iter(intervals))
            old_interval = geometry_intervals.setdefault(geometry_id, interval)
            if old_interval != interval:
                raise ValueError("geometry has inconsistent service intervals")
            upper_by_tier[geometry_id][tier.tier_id] = max(
                sample.elapsed_ns for sample in trailing
            )
            if all(sample.completed_model_rows is not None for sample in trailing):
                completed_rows = sum(
                    int(sample.completed_model_rows or 0) for sample in trailing
                )
                elapsed_ns = sum(sample.elapsed_ns for sample in trailing)
                row_rates[geometry_id][tier.tier_id] = Fraction(
                    completed_rows * 1_000_000_000,
                    elapsed_ns,
                )

    expanded: dict[int, tuple[int, ...]] = {}
    for geometry_id in geometries:
        prefix = 0
        tier_uppers: list[tuple[int, int]] = []
        for tier in tiers:
            prefix = max(prefix, upper_by_tier[geometry_id][tier.tier_id])
            tier_uppers.append((tier.max_active_population, prefix))
        expanded[geometry_id] = tuple(
            next(value for maximum, value in tier_uppers if population <= maximum)
            for population in range(1, max_population + 1)
        )

    if geometry_intervals[reference_geometry_id] != reference_interval_ms:
        raise ValueError("reference geometry does not match reference interval")
    reference_upper = expanded[reference_geometry_id]
    interval_ns = reference_interval_ms * 1_000_000
    measured_capacity = max(
        (
            population
            for population, duration in enumerate(reference_upper, start=1)
            if duration <= interval_ns
        ),
        default=0,
    )
    if measured_capacity == 0:
        raise ValueError("reference service interval cannot support one session")
    provisional = max(1, floor(measured_capacity * derating_factor))
    reference_demand = Fraction(1, provisional)
    demand_model = OnePointServiceDemand(reference_interval_ms, reference_demand)
    demand_source = tuple(
        (
            geometry_intervals[geometry_id],
            demand_model.at(geometry_intervals[geometry_id]),
        )
        for geometry_id in geometries
    )
    compiled = compile_service_demand_profile(
        demand_source,
        max_charged_population=max_population,
    )
    expanded_hash = _hash_json(
        {str(key): list(value) for key, value in expanded.items()}
    )
    tier_maxima = {
        tier.tier_id: tier.max_active_population for tier in tiers
    }
    candidate = ServiceProfileCandidate(
        execution_environment_key=context.execution_environment_key,
        precision_policy=context.precision_policy,
        state_profile=context.state_profile,
    )
    receipt_payload = {
        "compiler_version": context.compiler_version,
        "reference_interval_ms": reference_interval_ms,
        "reference_geometry_id": reference_geometry_id,
        "execution_tier_maxima": tier_maxima,
        "upper": upper_by_tier,
        "expanded_sha256": expanded_hash,
        "mixed_composition_policy": context.mixed_composition_policy,
        "scale": compiled.scale,
        "demand_units": compiled.demand_units,
        "provisional_capacity": provisional,
        "derating_factor": str(derating_factor),
        "context": {
            key: str(value) if isinstance(value, Fraction) else value
            for key, value in asdict(context).items()
        },
    }
    receipt_hash = _hash_json(receipt_payload)
    receipt = ServiceProfileReceipt(
        compiler_version=context.compiler_version,
        reference_interval_ms=reference_interval_ms,
        reference_geometry_id=reference_geometry_id,
        execution_tier_maxima=tier_maxima,
        measured_upper_duration_ns_by_geometry_and_tier=upper_by_tier,
        expanded_duration_table_sha256=expanded_hash,
        mixed_composition_policy=context.mixed_composition_policy,
        service_demand_scale=compiled.scale,
        service_demand_units=compiled.demand_units,
        aggregate_rounding_bound=compiled.aggregate_rounding_bound,
        maximum_charged_population=max_population,
        least_rational_demand=compiled.least_rational_demand,
        provisional_capacity=provisional,
        derating_factor=derating_factor,
        pre_override_physical_bound=context.pre_override_physical_bound,
        allocated_pool=context.allocated_pool,
        count_cap=context.count_cap,
        execution_claim_ceiling=context.execution_claim_ceiling,
        service_budget_source=context.service_budget_source,
        service_budget_coefficients=context.service_budget_coefficients,
        slot_bytes=context.slot_bytes,
        execution_environment_key=context.execution_environment_key,
        precision_policy=context.precision_policy,
        state_profile=context.state_profile,
        receipt_sha256=receipt_hash,
    )
    return StartupServiceProfile(
        upper_duration_ns_by_geometry_and_population=expanded,
        upper_duration_ns_by_geometry_and_tier=upper_by_tier,
        diagnostic_rows_per_second_by_geometry_and_tier=row_rates,
        measured_capacity=measured_capacity,
        provisional_capacity=provisional,
        reference_demand=reference_demand,
        compiled_demand=compiled,
        profile_candidate=candidate,
        receipt=receipt,
    )


# @spec PORT-STATE-026, PORT-PERF-008
def fallback_transaction_duration_ns(
    *,
    resident_counts_by_geometry: Sequence[int],
    homogeneous_upper_duration_ns_by_geometry: Sequence[Sequence[int]],
    mixed_composition_policy: str,
) -> int:
    counts = tuple(resident_counts_by_geometry)
    tables = tuple(tuple(table) for table in homogeneous_upper_duration_ns_by_geometry)
    if len(counts) != len(tables) or any(count < 0 for count in counts):
        raise ValueError("geometry counts and duration tables must align")
    populated = sum(count > 0 for count in counts)
    if mixed_composition_policy == "homogeneous_only" and populated > 1:
        raise ValueError("homogeneous-only profile cannot admit a mixed pool")
    if mixed_composition_policy not in {
        "homogeneous_only",
        "homogeneous_upper_sum",
    }:
        raise ValueError("unknown mixed composition policy")
    total = 0
    for count, table in zip(counts, tables):
        if count >= len(table):
            raise ValueError("population exceeds measured homogeneous table")
        total += table[count]
    return total


@dataclass(frozen=True)
class AdmissionDecision:
    candidate_supported: bool
    hard_feasible: bool
    nominal_dispatchable: bool
    binding_authority: str | None
    nominal_reason: str | None
    hard_headroom: int


@dataclass(frozen=True)
class AdmissionHeadroom:
    hard: int
    nominal: int


TransactionDuration = Callable[[tuple[int, ...]], Fraction]


def _candidate_index(candidate_interval_ms: int) -> int:
    try:
        return _SERVICE_INTERVALS_MS.index(candidate_interval_ms)
    except ValueError as exc:
        raise ValueError(
            f"unsupported service interval {candidate_interval_ms}"
        ) from exc


def _next_counts(
    resident: Sequence[int],
    reserved: Sequence[int],
    candidate_index: int,
    count: int = 1,
) -> tuple[int, ...]:
    if len(resident) != len(_SERVICE_INTERVALS_MS) or len(reserved) != len(
        _SERVICE_INTERVALS_MS
    ):
        raise ValueError("capacity histograms must cover every service interval")
    values = [left + right for left, right in zip(resident, reserved)]
    values[candidate_index] += count
    return tuple(values)


# @spec PORT-STATE-004, PORT-STATE-024, PORT-STATE-025, PORT-STATE-026
def evaluate_admission(
    *,
    resident_counts_by_interval: Sequence[int],
    reserved_counts_by_interval: Sequence[int],
    candidate_interval_ms: int,
    allocated_slots: int,
    count_limit: int,
    resident_count: int,
    reserved_count: int,
    execution_claims: int,
    max_num_seqs: int,
    charged_demand: Fraction,
    service_budget: Fraction,
    candidate_demand: Fraction,
    transaction_duration_ms: TransactionDuration,
) -> AdmissionDecision:
    index = _candidate_index(candidate_interval_ms)
    hard_headroom = max(
        0,
        min(
            allocated_slots - resident_count - reserved_count,
            count_limit - resident_count - reserved_count,
            max_num_seqs - execution_claims,
        ),
    )
    candidate_counts = tuple(1 if position == index else 0 for position in range(5))
    candidate_supported = (
        candidate_demand <= service_budget
        and transaction_duration_ms(candidate_counts) <= candidate_interval_ms
    )
    if not candidate_supported:
        return AdmissionDecision(
            candidate_supported=False,
            hard_feasible=False,
            nominal_dispatchable=False,
            binding_authority="candidate_alone",
            nominal_reason=None,
            hard_headroom=hard_headroom,
        )
    authority = None
    if allocated_slots <= resident_count + reserved_count:
        authority = "physical_slots"
    elif count_limit <= resident_count + reserved_count:
        authority = "logical_count"
    elif max_num_seqs <= execution_claims:
        authority = "execution_claims"
    if authority is not None:
        return AdmissionDecision(
            candidate_supported=True,
            hard_feasible=False,
            nominal_dispatchable=False,
            binding_authority=authority,
            nominal_reason=None,
            hard_headroom=hard_headroom,
        )
    counts = _next_counts(
        resident_counts_by_interval,
        reserved_counts_by_interval,
        index,
    )
    if charged_demand + candidate_demand > service_budget:
        nominal_reason = "service_budget"
    elif transaction_duration_ms(counts) > min(
        interval
        for interval, count in zip(_SERVICE_INTERVALS_MS, counts)
        if count
    ):
        nominal_reason = "transaction_time"
    else:
        nominal_reason = None
    return AdmissionDecision(
        candidate_supported=True,
        hard_feasible=True,
        nominal_dispatchable=nominal_reason is None,
        binding_authority=None,
        nominal_reason=nominal_reason,
        hard_headroom=hard_headroom,
    )


# @spec PORT-OBS-012, PORT-PERF-008
def admission_headroom(
    *,
    resident_counts_by_interval: Sequence[int],
    reserved_counts_by_interval: Sequence[int],
    candidate_interval_ms: int,
    allocated_slots: int,
    count_limit: int,
    resident_count: int,
    reserved_count: int,
    execution_claims: int,
    max_num_seqs: int,
    charged_demand: Fraction,
    service_budget: Fraction,
    candidate_demand: Fraction,
    transaction_duration_ms: TransactionDuration,
) -> AdmissionHeadroom:
    index = _candidate_index(candidate_interval_ms)
    hard = max(
        0,
        min(
            allocated_slots - resident_count - reserved_count,
            count_limit - resident_count - reserved_count,
            max_num_seqs - execution_claims,
        ),
    )
    if hard == 0 or candidate_demand <= 0:
        return AdmissionHeadroom(hard=hard, nominal=0)
    budget_room = max(Fraction(), service_budget - charged_demand)
    budget_cap = floor(budget_room / candidate_demand)
    upper = min(hard, budget_cap)
    if upper <= 0:
        return AdmissionHeadroom(hard=hard, nominal=0)

    def fits(count: int) -> bool:
        counts = _next_counts(
            resident_counts_by_interval,
            reserved_counts_by_interval,
            index,
            count,
        )
        deadline = min(
            interval
            for interval, population in zip(_SERVICE_INTERVALS_MS, counts)
            if population
        )
        return transaction_duration_ms(counts) <= deadline

    low, high = 0, upper
    while low < high:
        midpoint = (low + high + 1) // 2
        if fits(midpoint):
            low = midpoint
        else:
            high = midpoint - 1
    return AdmissionHeadroom(hard=hard, nominal=low)


@dataclass(frozen=True)
class AdmissionClaim:
    lease_key: str
    service_interval_ms: int
    demand: Fraction


@dataclass(frozen=True)
class AdmissionSnapshot:
    execution_claims: int
    charged_demand: Fraction
    intervals_ms: tuple[int, ...]


class PersistentStateAdmissionController:
    """Compatibility authority for exact-rational admission tests."""

    # @spec PORT-STATE-004, PORT-STATE-025
    def __init__(
        self,
        *,
        allocated_slots: int,
        count_limit: int,
        max_num_seqs: int,
        service_budget: Fraction,
        demand_model: ServiceDemand,
        transaction_duration_ms: Callable[[Sequence[int]], Fraction],
        service_source: str = "measured_fallback",
    ) -> None:
        if service_source not in {"qualified_profile", "measured_fallback"}:
            raise ValueError("unknown service-budget source")
        self._demand_model = demand_model
        self._claims: dict[str, AdmissionClaim] = {}

    @property
    def snapshot(self) -> AdmissionSnapshot:
        claims = tuple(self._claims.values())
        return AdmissionSnapshot(
            execution_claims=len(claims),
            charged_demand=sum(
                (claim.demand for claim in claims),
                Fraction(),
            ),
            intervals_ms=tuple(claim.service_interval_ms for claim in claims),
        )

    def reserve(self, lease_key: str, *, service_interval_ms: int) -> AdmissionClaim:
        if lease_key in self._claims:
            raise ValueError(f"duplicate persistent-state claim {lease_key!r}")
        claim = AdmissionClaim(
            lease_key,
            service_interval_ms,
            self._demand_model.at(service_interval_ms),
        )
        self._claims[lease_key] = claim
        return claim

    def park(self, claim: AdmissionClaim) -> None:
        if self._claims.get(claim.lease_key) != claim:
            raise ValueError("unknown persistent-state admission claim")

    def release(self, claim: AdmissionClaim) -> None:
        if self._claims.get(claim.lease_key) == claim:
            del self._claims[claim.lease_key]
