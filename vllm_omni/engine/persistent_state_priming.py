# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Leased service-path priming for persistent-state capacity measurement."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServicePrimingRound:
    """One ordinary engine-level priming round."""

    round_id: str
    service_interval_ms: int
    geometry_id: int
    active_population: int
    schema_id: str
    profile_id: str

    def __post_init__(self) -> None:
        if not self.round_id:
            raise ValueError("service-priming round id cannot be empty")
        if self.service_interval_ms <= 0 or self.geometry_id < 0:
            raise ValueError("service-priming geometry must be valid")
        if self.active_population <= 0:
            raise ValueError("service-priming population must be positive")
        if not self.schema_id or not self.profile_id:
            raise ValueError("service-priming state identity cannot be empty")

    @property
    def dummy_run(self) -> bool:
        return False

    @property
    def is_profile(self) -> bool:
        return False

    @property
    def submission_layer(self) -> str:
        return "engine"


@dataclass(frozen=True)
class ServicePrimingObservation:
    round_id: str
    service_interval_ms: int
    geometry_id: int
    active_population: int
    elapsed_ns: int
    completed_legal_parks: int
    completed_model_rows: int | None
    dummy_run: bool = False
    is_profile: bool = False


PrimingExecutor = Callable[
    [ServicePrimingRound, tuple[Any, ...]],
    Awaitable[Any],
]


# @spec PORT-PERF-005
async def run_service_priming_round(
    *,
    service: Any,
    round_spec: ServicePrimingRound,
    execute: PrimingExecutor,
) -> ServicePrimingObservation:
    """Execute a complete round through ordinary reserve and release."""
    leases: list[Any] = []
    try:
        for index in range(round_spec.active_population):
            operation_id = f"{round_spec.round_id}:reserve:{index}"
            lease = await service.reserve(
                operation_id=operation_id,
                session_key=f"{round_spec.round_id}:session:{index}",
                schema_id=round_spec.schema_id,
                profile_id=round_spec.profile_id,
            )
            leases.append(lease)
        result = await execute(round_spec, tuple(leases))
        if bool(result.dummy_run) or bool(result.is_profile):
            raise ValueError(
                "service priming cannot consume dummy/profile execution"
            )
        if result.completed_legal_parks != round_spec.active_population:
            raise ValueError(
                "service priming must complete one legal park per lease"
            )
        return ServicePrimingObservation(
            round_id=round_spec.round_id,
            service_interval_ms=round_spec.service_interval_ms,
            geometry_id=round_spec.geometry_id,
            active_population=round_spec.active_population,
            elapsed_ns=int(result.elapsed_ns),
            completed_legal_parks=int(result.completed_legal_parks),
            completed_model_rows=(
                None
                if result.completed_model_rows is None
                else int(result.completed_model_rows)
            ),
        )
    finally:
        for lease in leases:
            await service.release(
                operation_id=f"{lease.operation_id}:release",
                lease=lease,
                reason="service_priming_complete",
            )
