# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Leased service-path priming for persistent-state capacity measurement."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServicePrimingBudgetCell:
    """One pre-control-plane upper bound for a priming plan cell."""

    geometry_id: int
    tier_id: str
    max_active_population: int
    repetitions: int

    def __post_init__(self) -> None:
        if self.geometry_id < 0 or not self.tier_id:
            raise ValueError("priming budget cell identity must be valid")
        if self.max_active_population <= 0 or self.repetitions <= 0:
            raise ValueError("priming budget cell bounds must be positive")


@dataclass(frozen=True)
class ServicePrimingBudgetDescriptor:
    """Immutable plan envelope available before control-plane construction."""

    policy_version: str
    configured_population_ceiling: int
    cells: tuple[ServicePrimingBudgetCell, ...]

    def __post_init__(self) -> None:
        if not self.policy_version or self.configured_population_ceiling <= 0:
            raise ValueError("priming budget identity and population must be valid")
        if not self.cells:
            raise ValueError("priming budget must declare at least one cell")
        identities = tuple((cell.geometry_id, cell.tier_id) for cell in self.cells)
        if len(set(identities)) != len(identities):
            raise ValueError("priming budget cell identities must be unique")
        if any(
            cell.max_active_population > self.configured_population_ceiling
            for cell in self.cells
        ):
            raise ValueError(
                "priming budget cell exceeds configured population ceiling"
            )

    @property
    def bootstrap_operation_budget(self) -> int:
        """Worst-case reserve plus release identities for the plan."""

        return 2 * sum(
            cell.max_active_population * cell.repetitions
            for cell in self.cells
        )

    @property
    def canonical_json(self) -> str:
        payload = {
            "policy_version": self.policy_version,
            "configured_population_ceiling": (
                self.configured_population_ceiling
            ),
            "cells": [
                {
                    "geometry_id": cell.geometry_id,
                    "tier_id": cell.tier_id,
                    "max_active_population": cell.max_active_population,
                    "repetitions": cell.repetitions,
                }
                for cell in sorted(
                    self.cells,
                    key=lambda item: (item.geometry_id, item.tier_id),
                )
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


@dataclass(frozen=True)
class ServicePrimingPlanIdentity:
    """Covered inventory-derived plan identity for the startup receipt."""

    sha256: str
    operation_count: int


class PersistentStatePrimingCleanupError(RuntimeError):
    """All failures observed while releasing transient priming leases."""

    def __init__(self, errors: tuple[BaseException, ...]) -> None:
        if not errors:
            raise ValueError("cleanup error aggregate cannot be empty")
        self.errors = errors
        super().__init__(
            f"persistent-state priming cleanup failed ({len(errors)} errors)"
        )


@dataclass(frozen=True)
class ServicePrimingRound:
    """One ordinary engine-level priming round."""

    round_id: str
    service_interval_ms: int
    geometry_id: int
    active_population: int
    schema_id: str
    profile_id: str
    tier_id: str = "default"
    post_jit: bool = True
    continuously_loaded: bool = True

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
    tier_id: str = "default"
    post_jit: bool = True
    continuously_loaded: bool = True
    dummy_run: bool = False
    is_profile: bool = False


PrimingExecutor = Callable[
    [ServicePrimingRound, tuple[Any, ...]],
    Awaitable[Any],
]


# @spec PORT-STATE-012 / PORT-PERF-005
def validate_service_priming_plan(
    *,
    descriptor: ServicePrimingBudgetDescriptor,
    rounds: tuple[ServicePrimingRound, ...],
) -> ServicePrimingPlanIdentity:
    """Prove an executable plan is a covered subset of its budget."""

    if not rounds:
        raise ValueError("service-priming plan cannot be empty")
    cells = {
        (cell.geometry_id, cell.tier_id): cell for cell in descriptor.cells
    }
    repetitions: dict[tuple[int, str], int] = {}
    payload: list[dict[str, object]] = []
    for round_spec in rounds:
        identity = (round_spec.geometry_id, round_spec.tier_id)
        cell = cells.get(identity)
        if cell is None:
            raise ValueError(
                f"service-priming plan cell {identity!r} is outside its budget"
            )
        if round_spec.active_population > cell.max_active_population:
            raise ValueError(
                "service-priming plan population exceeds its budget cell"
            )
        repetitions[identity] = repetitions.get(identity, 0) + 1
        if repetitions[identity] > cell.repetitions:
            raise ValueError(
                "service-priming plan repetitions exceed its budget cell"
            )
        payload.append(
            {
                "round_id": round_spec.round_id,
                "geometry_id": round_spec.geometry_id,
                "tier_id": round_spec.tier_id,
                "active_population": round_spec.active_population,
                "service_interval_ms": round_spec.service_interval_ms,
            }
        )
    operation_count = 2 * sum(
        round_spec.active_population for round_spec in rounds
    )
    if operation_count > descriptor.bootstrap_operation_budget:
        raise ValueError("service-priming plan exceeds bootstrap operation budget")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return ServicePrimingPlanIdentity(
        sha256=hashlib.sha256(encoded).hexdigest(),
        operation_count=operation_count,
    )


async def _release_priming_leases(
    *,
    service: Any,
    round_id: str,
    leases: tuple[Any, ...],
    timeout_s: float,
) -> tuple[BaseException, ...]:
    """Attempt every exact release under one shared wall-clock bound."""

    tasks = tuple(
        asyncio.create_task(
            service.release(
                operation_id=f"{round_id}:release:{index}",
                lease=lease,
                reason="service_priming_complete",
            )
        )
        for index, lease in enumerate(leases)
    )
    if not tasks:
        return ()
    done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    errors: list[BaseException] = []
    for task in tasks:
        if task not in done:
            continue
        if task.cancelled():
            errors.append(asyncio.CancelledError())
            continue
        error = task.exception()
        if error is not None:
            errors.append(error)
    if pending:
        errors.append(
            asyncio.TimeoutError(
                "persistent-state priming cleanup exceeded rollback timeout"
            )
        )
        for task in pending:
            task.cancel()
            task.add_done_callback(
                lambda completed: (
                    None if completed.cancelled() else completed.exception()
                )
            )
    return tuple(errors)


# @spec PORT-PERF-005
async def run_service_priming_round(
    *,
    service: Any,
    round_spec: ServicePrimingRound,
    execute: PrimingExecutor,
    rollback_timeout_s: float = 30.0,
) -> ServicePrimingObservation:
    """Execute a complete round through ordinary reserve and release."""
    leases: list[Any] = []
    observation: ServicePrimingObservation | None = None
    primary: BaseException | None = None
    try:
        for index in range(round_spec.active_population):
            operation_id = f"{round_spec.round_id}:reserve:{index}"
            lease = await service.reserve_for_priming(
                operation_id=operation_id,
                session_key=f"{round_spec.round_id}:session:{index}",
                schema_id=round_spec.schema_id,
                profile_id=round_spec.profile_id,
                service_interval_ms=round_spec.service_interval_ms,
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
        observation = ServicePrimingObservation(
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
            tier_id=round_spec.tier_id,
            post_jit=round_spec.post_jit,
            continuously_loaded=round_spec.continuously_loaded,
        )
    except BaseException as error:
        primary = error
    cleanup_task = asyncio.create_task(
        _release_priming_leases(
            service=service,
            round_id=round_spec.round_id,
            leases=tuple(leases),
            timeout_s=rollback_timeout_s,
        )
    )
    try:
        cleanup_errors = await asyncio.shield(cleanup_task)
    except asyncio.CancelledError as cancellation:
        if primary is None:
            primary = cancellation
        cleanup_errors = await cleanup_task
    if cleanup_errors:
        aggregate = PersistentStatePrimingCleanupError(cleanup_errors)
        if primary is not None:
            raise primary from aggregate
        raise aggregate
    if primary is not None:
        raise primary
    if observation is None:
        raise RuntimeError("service priming produced no observation")
    return observation
