# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for leased decode/service priming."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "vllm_omni.engine.persistent_state_priming"


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _module() -> ModuleType:
    try:
        return importlib.import_module(_MODULE)
    except ModuleNotFoundError:
        _fail(f"PORT-PERF-005 missing {_MODULE}")


def _symbol(name: str) -> Any:
    try:
        return getattr(_module(), name)
    except AttributeError:
        _fail(f"PORT-PERF-005 missing {_MODULE}.{name}")


@dataclass
class _PrimingService:
    events: list[tuple[str, str]]
    resident: int = 0

    async def reserve(self, **kwargs: Any) -> Any:
        del kwargs
        pytest.fail(
            "PORT-PERF-005 priming crossed the public reserve surface",
            pytrace=False,
        )

    async def reserve_for_priming(self, **kwargs: Any) -> Any:
        operation_id = str(kwargs["operation_id"])
        session_key = str(kwargs["session_key"])
        self.events.append(("reserve", operation_id))
        self.resident += 1
        return SimpleNamespace(
            operation_id=operation_id,
            session_key=session_key,
            binding_token=f"binding-{session_key}",
        )

    async def release(self, **kwargs: Any) -> None:
        lease = kwargs["lease"]
        self.events.append(("release", str(lease.operation_id)))
        self.resident -= 1


def _round(*, active_population: int = 2) -> Any:
    return _symbol("ServicePrimingRound")(
        round_id="tier-small-repeat-0",
        service_interval_ms=1_120,
        geometry_id=4,
        active_population=active_population,
        schema_id="schema-a",
        profile_id="profile-a",
    )


def test_budget_descriptor_is_immutable_hashed_and_population_bounded() -> None:
    """@spec PORT-STATE-012 / PORT-PERF-005: bootstrap demand is finite."""
    cell = _symbol("ServicePrimingBudgetCell")
    descriptor_type = _symbol("ServicePrimingBudgetDescriptor")
    descriptor = descriptor_type(
        policy_version="priming-budget-v1",
        configured_population_ceiling=8,
        cells=tuple(
            cell(
                geometry_id=geometry_id,
                tier_id=tier_id,
                max_active_population=population,
                repetitions=4,
            )
            for geometry_id in range(5)
            for tier_id, population in (("single", 1), ("eager-bulk", 8))
        ),
    )

    assert descriptor.bootstrap_operation_budget == 360
    assert len(descriptor.sha256) == 64
    assert descriptor.sha256 == hashlib.sha256(
        descriptor.canonical_json.encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="population ceiling"):
        descriptor_type(
            policy_version="priming-budget-v1",
            configured_population_ceiling=2,
            cells=(
                cell(
                    geometry_id=0,
                    tier_id="bulk",
                    max_active_population=3,
                    repetitions=1,
                ),
            ),
        )


def test_executable_plan_must_be_covered_before_reserve() -> None:
    """@spec PORT-PERF-005: an out-of-budget plan performs no page I/O."""

    async def scenario() -> None:
        cell = _symbol("ServicePrimingBudgetCell")
        descriptor = _symbol("ServicePrimingBudgetDescriptor")(
            policy_version="priming-budget-v1",
            configured_population_ceiling=2,
            cells=(
                cell(
                    geometry_id=4,
                    tier_id="default",
                    max_active_population=2,
                    repetitions=1,
                ),
            ),
        )
        service = _PrimingService([])
        out_of_budget = (_round(active_population=3),)

        with pytest.raises(ValueError, match="budget|population"):
            _symbol("validate_service_priming_plan")(
                descriptor=descriptor,
                rounds=out_of_budget,
            )
        assert service.events == []
        assert service.resident == 0

    asyncio.run(scenario())


def test_service_priming_uses_ordinary_leased_requests_through_legal_park() -> None:
    """@spec PORT-PERF-005: measurement traverses canonical leased work."""

    async def scenario() -> None:
        events: list[tuple[str, str]] = []
        service = _PrimingService(events)
        round_spec = _round()

        async def execute(spec: Any, leases: tuple[Any, ...]) -> Any:
            events.append(("execute", spec.round_id))
            assert len(leases) == 2
            assert spec.dummy_run is False
            assert spec.is_profile is False
            assert spec.submission_layer == "engine"
            return SimpleNamespace(
                completed_legal_parks=2,
                elapsed_ns=800_000_000,
                completed_model_rows=6,
                dummy_run=False,
                is_profile=False,
            )

        observation = await _symbol("run_service_priming_round")(
            service=service,
            round_spec=round_spec,
            execute=execute,
        )

        assert events[:3] == [
            ("reserve", "tier-small-repeat-0:reserve:0"),
            ("reserve", "tier-small-repeat-0:reserve:1"),
            ("execute", "tier-small-repeat-0"),
        ]
        assert set(events[3:]) == {
            ("release", "tier-small-repeat-0:reserve:0"),
            ("release", "tier-small-repeat-0:reserve:1"),
        }
        assert observation.completed_legal_parks == 2
        assert observation.active_population == 2
        assert service.resident == 0

    asyncio.run(scenario())


def test_failed_service_priming_releases_every_transient_lease() -> None:
    """@spec PORT-PERF-005: failed readiness work leaves no authority."""

    async def scenario() -> None:
        events: list[tuple[str, str]] = []
        service = _PrimingService(events)

        async def execute(spec: Any, leases: tuple[Any, ...]) -> Any:
            del spec, leases
            events.append(("execute", "failed"))
            raise RuntimeError("priming failed")

        with pytest.raises(RuntimeError, match="priming failed"):
            await _symbol("run_service_priming_round")(
                service=service,
                round_spec=_round(active_population=3),
                execute=execute,
            )

        assert events[:4] == [
            ("reserve", "tier-small-repeat-0:reserve:0"),
            ("reserve", "tier-small-repeat-0:reserve:1"),
            ("reserve", "tier-small-repeat-0:reserve:2"),
            ("execute", "failed"),
        ]
        assert set(events[4:]) == {
            ("release", "tier-small-repeat-0:reserve:0"),
            ("release", "tier-small-repeat-0:reserve:1"),
            ("release", "tier-small-repeat-0:reserve:2"),
        }
        assert service.resident == 0

    asyncio.run(scenario())


def test_cleanup_attempts_every_release_and_preserves_primary_error() -> None:
    """@spec ENV-MIG-012 / PORT-PERF-005: rollback is exhaustive."""

    @dataclass
    class FailingReleaseService(_PrimingService):
        release_failures: set[str] | None = None

        async def release(self, **kwargs: Any) -> None:
            lease = kwargs["lease"]
            operation_id = str(lease.operation_id)
            self.events.append(("release", operation_id))
            if operation_id in (self.release_failures or set()):
                raise RuntimeError(f"release failed: {operation_id}")
            self.resident -= 1

    async def scenario() -> None:
        service = FailingReleaseService(
            [],
            release_failures={"tier-small-repeat-0:reserve:0"},
        )

        async def execute(spec: Any, leases: tuple[Any, ...]) -> Any:
            del spec, leases
            raise ValueError("primary priming failure")

        with pytest.raises(ValueError, match="primary priming failure") as info:
            await _symbol("run_service_priming_round")(
                service=service,
                round_spec=_round(active_population=3),
                execute=execute,
                rollback_timeout_s=0.5,
            )

        assert {event for event in service.events if event[0] == "release"} == {
            ("release", "tier-small-repeat-0:reserve:0"),
            ("release", "tier-small-repeat-0:reserve:1"),
            ("release", "tier-small-repeat-0:reserve:2"),
        }
        aggregate_type = _symbol("PersistentStatePrimingCleanupError")
        assert isinstance(info.value.__cause__, aggregate_type)
        assert len(info.value.__cause__.errors) == 1

    asyncio.run(scenario())


def test_cleanup_failure_without_primary_fails_the_round() -> None:
    """@spec ENV-MIG-012: success is impossible with incomplete cleanup."""

    class FailingReleaseService(_PrimingService):
        async def release(self, **kwargs: Any) -> None:
            lease = kwargs["lease"]
            self.events.append(("release", str(lease.operation_id)))
            raise RuntimeError("release failed")

    async def scenario() -> None:
        service = FailingReleaseService([])

        async def execute(spec: Any, leases: tuple[Any, ...]) -> Any:
            del spec
            return SimpleNamespace(
                completed_legal_parks=len(leases),
                elapsed_ns=1,
                completed_model_rows=None,
                dummy_run=False,
                is_profile=False,
            )

        with pytest.raises(
            _symbol("PersistentStatePrimingCleanupError"),
            match="cleanup",
        ):
            await _symbol("run_service_priming_round")(
                service=service,
                round_spec=_round(active_population=2),
                execute=execute,
                rollback_timeout_s=0.5,
            )
        assert len(service.events) == 4

    asyncio.run(scenario())


def test_timeout_cancellation_cannot_interrupt_inflight_release_fanout() -> None:
    """@spec ENV-MIG-012: the primary clock cannot abandon rollback."""

    class SlowReleaseService(_PrimingService):
        def __init__(self, events: list[tuple[str, str]]) -> None:
            super().__init__(events)
            self.release_started = asyncio.Event()
            self.release_count = 0

        async def release(self, **kwargs: Any) -> None:
            lease = kwargs["lease"]
            self.events.append(("release-start", str(lease.operation_id)))
            self.release_count += 1
            if self.release_count == 2:
                self.release_started.set()
            await asyncio.sleep(0.02)
            self.resident -= 1
            self.events.append(("release-done", str(lease.operation_id)))

    async def scenario() -> None:
        service = SlowReleaseService([])

        async def execute(spec: Any, leases: tuple[Any, ...]) -> Any:
            del spec
            return SimpleNamespace(
                completed_legal_parks=len(leases),
                elapsed_ns=1,
                completed_model_rows=None,
                dummy_run=False,
                is_profile=False,
            )

        task = asyncio.create_task(
            _symbol("run_service_priming_round")(
                service=service,
                round_spec=_round(active_population=2),
                execute=execute,
                rollback_timeout_s=0.5,
            )
        )
        await service.release_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert service.resident == 0
        assert len(
            [event for event in service.events if event[0] == "release-done"]
        ) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("marker", ["dummy_run", "is_profile"])
def test_service_priming_rejects_memory_profile_markers(marker: str) -> None:
    """@spec PORT-MIG-005 / PORT-PERF-005: profile authorities never merge."""

    async def scenario() -> None:
        events: list[tuple[str, str]] = []
        service = _PrimingService(events)

        async def execute(spec: Any, leases: tuple[Any, ...]) -> Any:
            del spec, leases
            return SimpleNamespace(
                completed_legal_parks=2,
                elapsed_ns=800_000_000,
                completed_model_rows=6,
                dummy_run=marker == "dummy_run",
                is_profile=marker == "is_profile",
            )

        with pytest.raises(ValueError, match=r"dummy|profile|service priming"):
            await _symbol("run_service_priming_round")(
                service=service,
                round_spec=_round(),
                execute=execute,
            )
        assert service.resident == 0

    asyncio.run(scenario())
