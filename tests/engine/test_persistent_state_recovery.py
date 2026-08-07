# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery contract: admission closures reopen; only epoch stays fatal.

PORT-STATE-022/023/024, from the live incident where a client storm
latched admission closed until restart. Every fail-closed edge except
engine-epoch invalidation demotes the service to un-handshaked, and the
next health probe re-runs the capability/inventory handshake and reopens
admission against a fresh engine snapshot. The handshake reconciles
orphaned engine bindings; horizon exhaustion is per-operation retryable
backpressure, never a service latch.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from vllm_omni.engine.persistent_state_service import (
    PersistentStateIndeterminate,
    PersistentStateService,
    PersistentStateServiceUnavailable,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SNAPSHOT_BASE: dict[str, Any] = {
    "engine_epoch": "epoch-a",
    "manager_revision": 0,
    "resident_count": 0,
    "physical_capacity": 4,
    "safety_reserve": 0,
    "configured_limit": 2,
    "effective_capacity": 2,
    "stage": 0,
    "replica": 0,
    "capabilities": ["resident"],
    "schema_id": "schema-a",
    "profile_id": "profile-a",
    "persistent_state_tombstone_ttl_s": 10.0,
    "persistent_state_max_tombstones": 8,
}


@dataclass
class _Clock:
    now: float = 100.0

    def __call__(self) -> float:
        return self.now


class _RecoveryStage:
    """Stage double whose failure modes are switchable mid-test."""

    def __init__(self) -> None:
        self.hang_reserve = False
        self._hang_release = asyncio.Event()
        self.reserve_error: Exception | None = None
        self.snapshot_error: Exception | None = None
        self.reserve_calls = 0
        self.release_calls: list[tuple[str, dict[str, Any]]] = []
        self.resident = 0
        self.revision = 0
        self.bindings: list[dict[str, Any]] = []
        self.snapshot_calls = 0

    def _snapshot(self) -> dict[str, Any]:
        self.snapshot_calls += 1
        snapshot = dict(_SNAPSHOT_BASE)
        snapshot["manager_revision"] = self.revision
        snapshot["resident_count"] = self.resident
        snapshot["bindings"] = [dict(b) for b in self.bindings]
        return snapshot

    async def call_utility_async(
        self, name: str, *args: Any
    ) -> dict[str, Any]:
        if name == "persistent_state_snapshot":
            if self.snapshot_error is not None:
                raise self.snapshot_error
            return self._snapshot()
        if name == "persistent_state_reserve":
            self.reserve_calls += 1
            if self.hang_reserve:
                # Parks like a slow engine RPC; releasing the hang lets
                # the in-flight call complete, as a recovered engine would.
                await self._hang_release.wait()
            if self.reserve_error is not None:
                raise self.reserve_error
            operation_id, session_key, schema_id, profile_id = args
            self.resident += 1
            self.revision += 1
            return {
                "operation_id": operation_id,
                "manager_revision": self.revision,
                "resident_count": self.resident,
                "lease": {
                    "engine_epoch": "epoch-a",
                    "session_key": session_key,
                    "generation": self.reserve_calls,
                    "schema_id": schema_id,
                    "profile_id": profile_id,
                    "location": "resident",
                    "binding_token": f"binding-{self.reserve_calls}",
                },
                "location_event": {
                    "engine_epoch": "epoch-a",
                    "session_key": session_key,
                    "generation": self.reserve_calls,
                    "location": "resident",
                    "transition": "reserved",
                },
            }
        if name == "persistent_state_release":
            operation_id, lease, reason = args
            self.release_calls.append((reason, dict(lease)))
            self.resident = max(0, self.resident - 1)
            self.revision += 1
            self.bindings = [
                b
                for b in self.bindings
                if b["binding_token"] != lease["binding_token"]
            ]
            return {
                "operation_id": operation_id,
                "manager_revision": self.revision,
                "resident_count": self.resident,
                "location_event": {
                    "engine_epoch": "epoch-a",
                    "session_key": lease["session_key"],
                    "generation": lease["generation"],
                    "location": "released",
                    "transition": "released",
                },
            }
        raise AssertionError(f"unexpected utility {name}")


def _service(
    stage: _RecoveryStage,
    clock: _Clock,
    **overrides: Any,
) -> PersistentStateService:
    kwargs: dict[str, Any] = dict(
        reserve_queue_capacity=4,
        cleanup_queue_capacity=4,
        operation_timeout_s=0.05,
        reconciliation_timeout_s=0.05,
        tombstone_ttl_s=10.0,
        max_tombstones=8,
        pending_claim_timeout_s=0.05,
        monotonic=clock,
    )
    kwargs.update(overrides)
    return PersistentStateService(stage, **kwargs)


def _lease_kwargs(n: int) -> dict[str, str]:
    return {
        "operation_id": f"op-{n}",
        "session_key": f"session-{n}",
        "schema_id": "schema-a",
        "profile_id": "profile-a",
    }


async def _open_service(
    service: PersistentStateService,
) -> None:
    await service.check_health()
    assert service.ready


# ---- PORT-STATE-022: demote and re-verify ----------------------------------


# @spec PORT-STATE-022
def test_indeterminate_reserve_demotes_then_next_probe_recovers() -> None:
    """Both reserve timeouts expire -> Indeterminate closes admission AND
    demotes; a later probe re-handshakes against the healthy engine and
    reopens. The incident's latch: closure without demotion was forever."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _service(stage, clock)
        await _open_service(service)
        stage.hang_reserve = True
        with pytest.raises(PersistentStateIndeterminate):
            await service.reserve(**_lease_kwargs(1))
        assert not service.ready
        stage.snapshot_error = RuntimeError("engine not answering")
        with pytest.raises(Exception):
            await service.check_health()  # engine unhealthy: stays closed
        assert not service.ready
        stage.snapshot_error = None
        stage.hang_reserve = False
        stage._hang_release.set()  # the parked op-1 RPC completes
        await service.check_health()
        assert service.ready
        result = await service.reserve(**_lease_kwargs(2))
        assert result.session_key == "session-2"
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-022
def test_epoch_change_stays_fatal_with_its_own_named_error() -> None:
    """Epoch invalidation is the one terminal closure: probes keep failing
    with the epoch-specific message even against a healthy engine."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _service(stage, clock)
        await _open_service(service)
        service.engine_epoch_changed("epoch-b")
        for _ in range(2):
            with pytest.raises(PersistentStateServiceUnavailable) as info:
                await service.check_health()
            assert "epoch" in str(info.value)
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-022
def test_retained_tombstones_survive_recovery() -> None:
    """Exact-retry semantics hold across a demote/re-handshake cycle: the
    same operation id returns the same result with no second engine
    reserve."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _service(stage, clock)
        await _open_service(service)
        first = await service.reserve(**_lease_kwargs(1))
        reserve_calls_before = stage.reserve_calls
        stage.hang_reserve = True
        with pytest.raises(PersistentStateIndeterminate):
            await service.reserve(**_lease_kwargs(2))
        stage.hang_reserve = False
        stage._hang_release.set()
        await service.check_health()
        assert service.ready
        again = await service.reserve(**_lease_kwargs(1))
        assert again.binding_token == first.binding_token
        assert stage.reserve_calls == reserve_calls_before + 1  # only op-2
        service.shutdown()

    asyncio.run(scenario())


# ---- PORT-STATE-023: orphan reconciliation at handshake --------------------


# @spec PORT-STATE-023
def test_handshake_releases_expired_orphan_bindings_only() -> None:
    """A binding with no live lease and an expired claim horizon is
    released during re-handshake; a live lease's binding is untouched."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _service(stage, clock)
        await _open_service(service)
        live = await service.reserve(**_lease_kwargs(1))
        stage.bindings = [
            {
                "binding_token": live.binding_token,
                "session_key": "session-1",
                "generation": live.generation,
                "schema_id": "schema-a",
                "profile_id": "profile-a",
                "engine_epoch": "epoch-a",
                "claim_expires_at": clock.now + 100.0,
            },
            {
                "binding_token": "binding-orphan",
                "session_key": "session-lost",
                "generation": 7,
                "schema_id": "schema-a",
                "profile_id": "profile-a",
                "engine_epoch": "epoch-a",
                "claim_expires_at": clock.now - 1.0,
            },
        ]
        stage.resident = 2
        stage.hang_reserve = True
        with pytest.raises(PersistentStateIndeterminate):
            await service.reserve(**_lease_kwargs(3))
        stage.hang_reserve = False
        await service.check_health()
        assert service.ready
        released_tokens = [
            lease["binding_token"] for _, lease in stage.release_calls
        ]
        assert released_tokens == ["binding-orphan"]
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-023
def test_unexpired_orphan_claims_are_never_reclaimed() -> None:
    """An orphan whose pending-claim horizon has not expired is left for
    the ordinary claim machinery."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _service(stage, clock)
        await _open_service(service)
        stage.bindings = [
            {
                "binding_token": "binding-young",
                "session_key": "session-young",
                "generation": 3,
                "schema_id": "schema-a",
                "profile_id": "profile-a",
                "engine_epoch": "epoch-a",
                "claim_expires_at": clock.now + 100.0,
            }
        ]
        stage.resident = 1
        stage.hang_reserve = True
        with pytest.raises(PersistentStateIndeterminate):
            await service.reserve(**_lease_kwargs(1))
        stage.hang_reserve = False
        await service.check_health()
        assert service.ready
        assert stage.release_calls == []
        service.shutdown()

    asyncio.run(scenario())


# ---- PORT-STATE-024: horizon exhaustion is backpressure, not a latch -------


# @spec PORT-STATE-024
def test_horizon_exhaustion_is_retryable_and_never_closes_admission() -> None:
    """The engine's typed horizon rejection sheds that one reserve as
    retryable backpressure; admission stays open and the next attempt
    succeeds once the engine admits it."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _service(stage, clock)
        await _open_service(service)
        stage.reserve_error = RuntimeError(
            "persistent_state_horizon_exhausted: retry after tombstone expiry"
        )
        with pytest.raises(Exception) as info:
            await service.reserve(**_lease_kwargs(1))
        assert getattr(info.value, "retryable", False), (
            "horizon exhaustion must surface as typed retryable backpressure"
        )
        assert service.ready, "backpressure must not close admission"
        stage.reserve_error = None
        result = await service.reserve(**_lease_kwargs(2))
        assert result.session_key == "session-2"
        service.shutdown()

    asyncio.run(scenario())
