# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for native realtime state admission and cleanup."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _persistent_state_runtime_values() -> dict[str, int | float]:
    return {
        "persistent_state_safety_reserve_slots": 1,
        "max_resident_sessions": 4,
        "persistent_state_reserve_queue_capacity": 4,
        "persistent_state_operation_timeout_s": 10.0,
        "persistent_state_reconciliation_timeout_s": 30.0,
        "persistent_state_tombstone_ttl_s": 600.0,
        "persistent_state_max_tombstones": 16,
        "persistent_state_pending_claim_timeout_s": 15.0,
        "streaming_session_configuration_timeout_s": 10.0,
        "streaming_session_finalization_timeout_s": 40.0,
        "streaming_accepted_audio_capacity_samples": 480_000,
        "streaming_max_retained_transcript_bytes": 1 << 20,
    }


class _WebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed: list[int] = []

    async def send_text(self, value: str) -> None:
        self.sent.append(json.loads(value))

    async def close(self, *, code: int) -> None:
        self.closed.append(code)


class _Service:
    def __init__(self) -> None:
        self.reserve_calls: list[dict[str, Any]] = []
        self.release_calls: list[dict[str, Any]] = []
        self.pending_cleanup_calls: list[object] = []
        self.pending_cleanup_wins = True
        self.pending_claim_timeout_s = 30.0
        self.lease = object()
        self.runtime_config = SimpleNamespace(
            accepted_audio_budget_s=30.0,
            accepted_audio_capacity_samples=480_000,
            max_retained_transcript_bytes=1 << 20,
            max_session_samples=None,
            session_configuration_timeout_s=30.0,
            session_idle_timeout_s=60.0,
            session_finalization_timeout_s=40.0,
        )

    async def reserve(self, **kwargs: Any) -> object:
        self.reserve_calls.append(kwargs)
        return self.lease

    async def release(self, **kwargs: Any) -> None:
        self.release_calls.append(kwargs)

    async def claim_pending_cleanup(self, lease: object) -> bool:
        self.pending_cleanup_calls.append(lease)
        return self.pending_cleanup_wins


class _Engine:
    def __init__(self, service: _Service) -> None:
        self.service = service
        self.default_sampling_params_list: list[object] = []

    def get_persistent_state_service(self) -> _Service:
        return self.service


class _Serving:
    def __init__(self, engine: _Engine) -> None:
        self.engine_client = engine
        self.runtime_config = engine.service.runtime_config

    def _is_model_supported(self, model: str | None) -> bool:
        return model == "nemotron-asr"


def _connection() -> tuple[RealtimeConnection, _WebSocket, _Service]:
    websocket = _WebSocket()
    service = _Service()
    connection = RealtimeConnection(websocket, _Serving(_Engine(service)))
    connection._is_connected = True
    return connection, websocket, service


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_audio_before_manager_ack_is_rejected_without_buffering() -> None:
    # @spec PORT-SESS-010
    connection, websocket, service = _connection()
    audio = base64.b64encode(b"\x01\x00").decode()

    await connection.handle_event(
        {"type": "input_audio_buffer.append", "audio": audio}
    )

    assert service.reserve_calls == []
    assert connection.audio_queue.empty()
    assert websocket.sent[-1]["type"] == "error"
    assert websocket.sent[-1]["code"] == "model_not_validated"


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_first_valid_update_reserves_before_marking_model_valid() -> None:
    # @spec PORT-STATE-004 / PORT-SESS-010
    connection, _, service = _connection()

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr"}
    )

    assert len(service.reserve_calls) == 1
    call = service.reserve_calls[0]
    assert call["operation_id"]
    assert call["session_key"]
    assert call["schema_id"]
    assert call["profile_id"]
    assert connection._state_lease is service.lease
    assert connection._is_model_validated is True


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_cleanup_is_the_single_idempotent_physical_release_owner() -> None:
    # @spec PORT-STATE-014 / PORT-SESS-011
    connection, _, service = _connection()
    connection._state_lease = service.lease
    connection._state_operation_id = "release-op"

    await connection.cleanup()
    await connection.cleanup()

    assert len(service.release_calls) == 1
    assert service.release_calls[0]["operation_id"] == "release-op"
    assert service.release_calls[0]["lease"] is service.lease


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_pending_claim_timeout_uses_the_same_cleanup_owner_once() -> None:
    # @spec PORT-STATE-014 / PORT-SESS-011
    connection, _, service = _connection()
    service.pending_claim_timeout_s = 0.01

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr"}
    )
    await asyncio.sleep(0.03)

    assert service.pending_cleanup_calls == [service.lease]
    assert len(service.release_calls) == 1
    assert service.release_calls[0]["reason"] == "pending_claim_timeout"
    await connection.cleanup()
    assert len(service.release_calls) == 1


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_pending_claim_timeout_never_releases_a_claimed_generation() -> None:
    # @spec PORT-STATE-014 / PORT-STATE-019
    connection, _, service = _connection()
    service.pending_claim_timeout_s = 0.01
    service.pending_cleanup_wins = False

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr"}
    )
    await asyncio.sleep(0.03)

    assert service.pending_cleanup_calls == [service.lease]
    assert service.release_calls == []
    await connection.cleanup()
    assert len(service.release_calls) == 1


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_idle_timeout_aborts_and_releases_through_the_same_owner() -> None:
    # @spec PORT-SESS-005 / PORT-STATE-014
    connection, websocket, service = _connection()
    connection._session_lifecycle._idle_timeout_s = 0.01

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr"}
    )
    await asyncio.sleep(0.03)

    assert websocket.sent[-1]["code"] == "idle_timeout"
    assert websocket.closed == [1008]
    assert len(service.release_calls) == 1
    assert service.release_calls[0]["reason"] == "idle_timeout"
    await connection.cleanup()
    assert len(service.release_calls) == 1


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_audio_after_lifecycle_expiry_reports_session_expired() -> None:
    # @spec PORT-SESS-005
    connection, websocket, service = _connection()
    connection._is_model_validated = True
    connection._nemotron_session = SimpleNamespace()
    connection._session_lifecycle._expired = True
    audio = base64.b64encode(b"\x01\x00").decode()

    await connection.handle_event(
        {"type": "input_audio_buffer.append", "audio": audio}
    )

    assert websocket.sent[-1]["code"] == "session_expired"
    assert service.release_calls == []


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_rearming_idle_timeout_fences_the_superseded_timer() -> None:
    # @spec PORT-SESS-005
    connection, _websocket, service = _connection()
    connection._session_lifecycle._idle_timeout_s = 3600.0
    connection._arm_session_lifecycle_timeout("idle")
    stale_generation = connection._session_lifecycle.generation

    connection._arm_session_lifecycle_timeout("idle")
    connection._session_lifecycle._deadline_reached(
        stale_generation,
        "idle",
    )
    await asyncio.sleep(0)

    assert connection._session_lifecycle.expired is False
    assert service.release_calls == []
    await connection.cleanup()


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_configuration_timeout_releases_a_late_reserve_result() -> None:
    # @spec PORT-SESS-012 / PORT-STATE-014
    connection, websocket, service = _connection()
    connection.session_configuration_timeout = 0.01
    allow_reserve = asyncio.Event()

    async def delayed_reserve(**kwargs: Any) -> object:
        service.reserve_calls.append(kwargs)
        await allow_reserve.wait()
        return service.lease

    service.reserve = delayed_reserve  # type: ignore[method-assign]
    connection._configuration_timeout_task = asyncio.create_task(
        connection._configuration_timeout()
    )
    configure = asyncio.create_task(
        connection.handle_event(
            {"type": "session.update", "model": "nemotron-asr"}
        )
    )
    await asyncio.sleep(0.02)
    allow_reserve.set()
    await configure

    assert websocket.sent[-1]["code"] == "model_not_validated"
    assert websocket.closed == [1008]
    assert connection._is_model_validated is False
    assert len(service.release_calls) == 1
    assert service.release_calls[0]["reason"] == "configuration_timeout"


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_finalization_timeout_emits_no_success_and_releases_once() -> None:
    # @spec PORT-SESS-005 / PORT-OBS-006
    connection, websocket, service = _connection()
    connection._session_lifecycle._finalization_timeout_s = 0.01
    connection._state_lease = service.lease
    connection._state_operation_id = "release-op"
    connection._nemotron_session = SimpleNamespace(session_key="session-a")

    async def running_generation() -> None:
        await asyncio.sleep(60.0)

    connection.generation_task = asyncio.create_task(running_generation())
    connection._arm_session_lifecycle_timeout("finalization")
    await asyncio.sleep(0.03)

    assert websocket.sent[-1]["code"] == "finalization_timeout"
    assert all(event["type"] != "transcription.done" for event in websocket.sent)
    assert len(service.release_calls) == 1
    assert service.release_calls[0]["reason"] == "finalization_timeout"


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_cleanup_waits_for_abort_and_decrements_active_before_release() -> None:
    # @spec PORT-STATE-014 / PORT-OBS-010
    connection, _, service = _connection()
    order: list[str] = []

    async def generation() -> None:
        try:
            await asyncio.sleep(60.0)
        finally:
            order.append("generation_terminal")

    async def release(**kwargs: Any) -> None:
        order.append("physical_release")
        service.release_calls.append(kwargs)

    class _Observer:
        def clear_all_outstanding(
            self,
            session_key: str,
            *,
            outcome: str,
        ) -> None:
            del session_key, outcome
            order.append("backlog_clear")

        def session_finished(self, *, session_key: str, reason: str) -> None:
            del session_key, reason
            order.append("active_decrement")

    service.release = release  # type: ignore[method-assign]
    connection._state_lease = service.lease
    connection._state_operation_id = "release-op"
    connection._nemotron_session = SimpleNamespace(session_key="session-a")
    connection._observer = _Observer()
    connection.generation_task = asyncio.create_task(generation())
    await asyncio.sleep(0)

    await connection.cleanup()

    assert order == [
        "generation_terminal",
        "backlog_clear",
        "active_decrement",
        "physical_release",
    ]


def test_connection_arms_a_finite_configuration_timeout_at_creation() -> None:
    # @spec PORT-SESS-012
    source = inspect.getsource(RealtimeConnection)

    assert "session_configuration_timeout" in source
    assert "create_task" in source
    assert "_is_model_validated" in source


def test_connection_uses_service_errors_without_inventing_a_new_wire_shape() -> None:
    # @spec PORT-SESS-011
    source = inspect.getsource(RealtimeConnection)

    assert "capacity_exhausted" in source
    assert "service_unavailable" in source
    assert "model_not_validated" in source
    assert "ErrorEvent" not in source or "send_error" in source


def test_connection_mints_fresh_session_and_operation_ids_per_attempt() -> None:
    # @spec PORT-STATE-012 / PORT-SESS-011
    source = inspect.getsource(RealtimeConnection)

    assert "uuid4" in source
    assert "operation_id" in source
    assert "session_key" in source
    assert "operation_id = session_key" not in source


def test_realtime_generation_carries_the_acknowledged_lease_binding() -> None:
    # @spec PORT-STATE-019 / PORT-SESS-010
    source = inspect.getsource(RealtimeConnection._run_generation)

    assert "_state_lease" in source
    assert "additional_information" in source or "state_binding" in source


def test_native_session_does_not_depend_on_rfc2_application_plugins() -> None:
    # @spec PORT-STATE-004 / PORT-MIG-006
    source = inspect.getsource(RealtimeConnection)

    assert "ApplicationPlugin" not in source
    assert "vllm_riva_frontend" not in source
    assert "app.state" not in source


def test_connection_keeps_transport_and_model_admission_distinct() -> None:
    # @spec PORT-SESS-010
    connection, _, _ = _connection()

    assert connection._is_connected is True
    assert connection._is_model_validated is False
    assert getattr(connection, "_state_lease", None) is None


def test_definitive_denial_leaves_socket_unadmitted_for_a_fresh_update() -> None:
    # @spec PORT-SESS-011
    source = inspect.getsource(RealtimeConnection.handle_event)

    assert "capacity_exhausted" in source
    assert "service_unavailable" in source
    assert "_is_connected = False" not in source
    assert "_is_model_validated = True" in source


def test_post_submit_indeterminate_path_never_mints_a_parallel_attempt() -> None:
    # @spec PORT-STATE-013 / PORT-SESS-011
    source = inspect.getsource(RealtimeConnection.handle_event)

    assert "indeterminate" in source or "reconcil" in source
    assert "same operation" in source.lower() or "operation_id" in source
    assert "parallel" not in source.lower()


def test_session_factory_retrieves_the_exact_service_without_reserving() -> None:
    # @spec PORT-RTC-003
    try:
        session_module = __import__(
            "vllm_omni.model_executor.models.nemotron_asr.session",
            fromlist=["create_nemotron_session_factory"],
        )
    except ModuleNotFoundError:
        pytest.fail(
            "PORT-RTC-003 missing Nemotron realtime session factory",
            pytrace=False,
        )
    service = _Service()
    engine = _Engine(service)

    factory = session_module.create_nemotron_session_factory(engine)

    assert factory.persistent_state_service is service
    assert service.reserve_calls == []
    assert not hasattr(factory, "open_ephemeral")


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_app_state_inventories_service_before_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-INT-013, PORT-STATE-004, PORT-STATE-015
    from vllm.model_executor import model_loader

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.api_server import (
        _install_persistent_state_service,
    )

    class _PersistentModel:
        supports_persistent_state = True

    class _Stage:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def call_utility_async(self, name: str, *args: Any) -> dict[str, Any]:
            del args
            self.calls.append(name)
            return {
                "engine_epoch": "epoch-1",
                "manager_revision": 0,
                "resident_count": 0,
                "physical_capacity": 5,
                "safety_reserve": 1,
                "configured_limit": 4,
                "effective_capacity": 4,
                "stage": 0,
                "replica": 0,
                "capabilities": ["resident"],
                "schema_id": "schema",
                "profile_id": "profile",
            }

    monkeypatch.setattr(model_loader, "get_model_cls", lambda _config: _PersistentModel)
    engine = object.__new__(AsyncOmni)
    stage = _Stage()
    engine.engine = SimpleNamespace(stage_clients=[stage])
    engine._persistent_state_service = None

    await _install_persistent_state_service(
        engine,
        SimpleNamespace(
            model_config=object(),
            additional_config=_persistent_state_runtime_values(),
        ),
    )

    service = engine.get_persistent_state_service()
    assert stage.calls == ["persistent_state_snapshot"]
    assert service.ready is True
    service.shutdown()


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_failed_inventory_never_installs_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-INT-013, PORT-STATE-015
    from vllm.model_executor import model_loader

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.api_server import (
        _install_persistent_state_service,
    )

    class _PersistentModel:
        supports_persistent_state = True

    class _Stage:
        async def call_utility_async(self, name: str, *args: Any) -> dict[str, Any]:
            del name, args
            raise RuntimeError("inventory failed")

    monkeypatch.setattr(model_loader, "get_model_cls", lambda _config: _PersistentModel)
    engine = object.__new__(AsyncOmni)
    engine.engine = SimpleNamespace(stage_clients=[_Stage()])
    engine._persistent_state_service = None

    with pytest.raises(RuntimeError, match="inventory failed"):
        await _install_persistent_state_service(
            engine,
            SimpleNamespace(
                model_config=object(),
                additional_config=_persistent_state_runtime_values(),
            ),
        )

    assert engine._persistent_state_service is None
