# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for leased decode/service priming."""

from __future__ import annotations

import asyncio
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
