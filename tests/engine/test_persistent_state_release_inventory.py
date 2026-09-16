# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Release recovery through real EngineCore inventory and manager ownership."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.engine.test_persistent_state_recovery import _Clock, _service
from tests.engine.test_persistent_state_service_contract import (
    _claim_payload,
    _state_core_for_cleanup_tests,
)
from vllm_omni.engine.persistent_state_service import PersistentStateServiceUnavailable

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StageTransport:
    """Real utility methods; faults exist only at the RPC delivery boundary."""

    def __init__(self) -> None:
        self.core, self.manager, self.spec = _state_core_for_cleanup_tests()
        self.manager.resident_state_scatter_warmup_complete = True
        self.release_operations: list[str] = []
        self.release_replies: list[dict[str, Any]] = []
        self.fail_before_release = False
        self.lose_release_reply = False
        self.claim_after_snapshot: dict[str, Any] | None = None
        self.physical_drops = 0
        self.initial_free_blocks = self.manager.block_pool.get_num_free_blocks()
        drop = self.manager.drop_lease

        def counted_drop(request_id: str) -> None:
            assert self.manager.get_state_binding(request_id) is not None
            self.physical_drops += 1
            drop(request_id)

        self.manager.drop_lease = counted_drop

    async def call_utility_async(self, name: str, *args: Any) -> dict[str, Any]:
        if name == "persistent_state_snapshot":
            snapshot = self.core.persistent_state_snapshot()
            if self.claim_after_snapshot is not None:
                lease, self.claim_after_snapshot = self.claim_after_snapshot, None
                self.core.claim_pending_lease(**_claim_payload(lease))
            return snapshot
        if name == "persistent_state_release":
            self.release_operations.append(args[0])
            if self.fail_before_release:
                self.fail_before_release = False
                raise RuntimeError("transient release transport failure")
            result = self.core.persistent_state_release(*args)
            self.release_replies.append(result)
            if self.lose_release_reply:
                self.lose_release_reply = False
                raise RuntimeError("release committed but reply was lost")
            return result
        if name == "persistent_state_reserve":
            return self.core.persistent_state_reserve(*args)
        raise AssertionError(f"unexpected utility {name}")


def _new_service(stage: _StageTransport) -> Any:
    return _service(stage, _Clock(), max_tombstones=4)


async def _reserve(service: Any, stage: _StageTransport) -> Any:
    await service.check_health()
    assert service.ready
    return await service.reserve(
        operation_id="reserve-a",
        session_key="session-a",
        schema_id=stage.spec.schema_id,
        profile_id="default",
        service_interval_ms=560,
    )


def _assert_charged(service: Any, stage: _StageTransport, lease: Any) -> None:
    assert not service.ready
    assert service.inventory["resident_count"] == 1
    assert service.inventory["execution_claims"] == 1
    assert service.inventory["charged_demand"] > 0
    assert lease.binding_token in service._failed_releases
    assert stage.manager.get_state_binding(lease.session_key) is not None
    assert stage.physical_drops == 0
    assert stage.manager.block_pool.get_num_free_blocks() == stage.initial_free_blocks - 1


def _assert_recovered(service: Any, stage: _StageTransport, lease: Any) -> None:
    assert service.ready
    assert service.inventory["resident_count"] == 0
    assert service.inventory["execution_claims"] == 0
    assert service.inventory["charged_demand"] == 0
    assert not service._failed_releases
    assert not any(service._failed_release_interval_counts.values())
    assert not any(service._resident_interval_counts.values())
    assert lease.binding_token not in service._live_leases
    assert stage.manager.get_state_binding(lease.session_key) is None
    assert stage.physical_drops == 1
    assert stage.manager.block_pool.get_num_free_blocks() == stage.initial_free_blocks


@pytest.mark.parametrize("owner", ["pending", "cleanup", "claimed"])
@pytest.mark.parametrize("terminal_method", [None, "mark_terminal", "free", "pop_blocks_for_free"])
def test_real_snapshot_projects_release_eligibility(owner: str, terminal_method: str | None) -> None:
    """@spec PORT-STATE-014 / PORT-STATE-022: snapshot agrees with release authority."""
    stage = _StageTransport()
    reserved = stage.core.persistent_state_reserve("reserve-a", "session-a", stage.spec.schema_id, "default")
    lease = reserved["lease"]
    if owner == "cleanup":
        assert stage.core.persistent_state_begin_pending_cleanup(lease)
    elif owner == "claimed":
        stage.core.claim_pending_lease(**_claim_payload(lease))
    if terminal_method is not None:
        getattr(stage.manager, terminal_method)("session-a")
    else:
        assert not stage.manager.is_terminal("session-a")
    inventory = stage.core.persistent_state_snapshot()["bindings"]
    assert len(inventory) == 1
    assert inventory[0].get("terminal") is (owner != "claimed" or terminal_method is not None)
    assert stage.physical_drops == 0
    assert stage.manager.get_state_binding("session-a") is not None


@pytest.mark.parametrize("owner", ["pending", "cleanup"])
def test_unclaimed_release_recovers_through_real_inventory(owner: str) -> None:
    """@spec PORT-STATE-014 / PORT-STATE-022: unclaimed retry needs no scheduler mark."""

    async def scenario() -> None:
        stage = _StageTransport()
        service = _new_service(stage)
        try:
            lease = await _reserve(service, stage)
            if owner == "cleanup":
                assert stage.core.persistent_state_begin_pending_cleanup(service._lease_payload(lease))
            stage.fail_before_release = True
            with pytest.raises(PersistentStateServiceUnavailable):
                await service.release(operation_id="release-a", lease=lease, reason="disconnect")
            _assert_charged(service, stage, lease)
            assert not stage.manager.is_terminal(lease.session_key)
            await service.check_health()
            _assert_recovered(service, stage, lease)
            assert stage.release_operations == ["release-a", "release-a"]
            await service.check_health()
            assert stage.physical_drops == 1
            later = await service.reserve(
                operation_id="reserve-b",
                session_key="session-b",
                schema_id=stage.spec.schema_id,
                profile_id="default",
                service_interval_ms=560,
            )
            await service.release(operation_id="release-b", lease=later, reason="completed")
            assert stage.physical_drops == 2
            assert stage.manager.block_pool.get_num_free_blocks() == stage.initial_free_blocks
        finally:
            service.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal_method", ["mark_terminal", "free", "pop_blocks_for_free"])
def test_claimed_release_waits_for_real_scheduler_terminality(terminal_method: str) -> None:
    """@spec PORT-STATE-014 / PORT-STATE-022: running work stays charged until fenced."""

    async def scenario() -> None:
        stage = _StageTransport()
        service = _new_service(stage)
        try:
            lease = await _reserve(service, stage)
            stage.core.claim_pending_lease(**_claim_payload(service._lease_payload(lease)))
            with pytest.raises(PersistentStateServiceUnavailable):
                await service.release(operation_id="release-a", lease=lease, reason="disconnect")
            _assert_charged(service, stage, lease)
            with pytest.raises(PersistentStateServiceUnavailable, match="not terminally reconcilable"):
                await service.check_health()
            assert stage.release_operations == ["release-a"]
            _assert_charged(service, stage, lease)
            getattr(stage.manager, terminal_method)(lease.session_key)
            await service.check_health()
            _assert_recovered(service, stage, lease)
            assert stage.release_operations == ["release-a", "release-a"]
        finally:
            service.shutdown()

    asyncio.run(scenario())


def test_pending_snapshot_cannot_release_a_racing_claimed_request() -> None:
    """@spec PORT-STATE-014 / PORT-STATE-019: RPC rechecks authority after snapshot."""

    async def scenario() -> None:
        stage = _StageTransport()
        service = _new_service(stage)
        try:
            lease = await _reserve(service, stage)
            stage.fail_before_release = True
            with pytest.raises(PersistentStateServiceUnavailable):
                await service.release(operation_id="release-a", lease=lease, reason="disconnect")
            stage.claim_after_snapshot = service._lease_payload(lease)
            with pytest.raises(PersistentStateServiceUnavailable, match="still running"):
                await service.check_health()
            assert stage.release_operations == ["release-a", "release-a"]
            _assert_charged(service, stage, lease)
            with pytest.raises(PersistentStateServiceUnavailable, match="not terminally reconcilable"):
                await service.check_health()
            assert stage.release_operations == ["release-a", "release-a"]
            stage.manager.mark_terminal(lease.session_key)
            await service.check_health()
            _assert_recovered(service, stage, lease)
            assert stage.release_operations == ["release-a"] * 3
        finally:
            service.shutdown()

    asyncio.run(scenario())


def test_lost_committed_release_reply_reconciles_tombstone_without_second_free() -> None:
    """@spec PORT-STATE-012 / PORT-STATE-014: absent binding retries the original operation."""

    async def scenario() -> None:
        stage = _StageTransport()
        service = _new_service(stage)
        try:
            lease = await _reserve(service, stage)
            stage.lose_release_reply = True
            with pytest.raises(PersistentStateServiceUnavailable):
                await service.release(operation_id="release-a", lease=lease, reason="disconnect")
            assert not service.ready
            assert stage.core.persistent_state_snapshot()["bindings"] == []
            assert stage.physical_drops == 1
            await service.check_health()
            _assert_recovered(service, stage, lease)
            assert stage.release_operations == ["release-a", "release-a"]
            reply = stage.core.persistent_state_release("release-a", service._lease_payload(lease), "disconnect")
            assert reply["resident_count"] == 0
            assert len(stage.release_replies) == 2
            assert reply["manager_revision"] == stage.release_replies[0]["manager_revision"]
            assert reply["manager_revision"] == stage.release_replies[1]["manager_revision"]
            assert stage.physical_drops == 1
            assert stage.manager.block_pool.get_num_free_blocks() == stage.initial_free_blocks
        finally:
            service.shutdown()

    asyncio.run(scenario())
