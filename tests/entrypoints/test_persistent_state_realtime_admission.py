# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for native realtime state admission and cleanup."""

from __future__ import annotations

import base64
import inspect
import json
from typing import Any

import pytest

from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _WebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_text(self, value: str) -> None:
        self.sent.append(json.loads(value))


class _Service:
    def __init__(self) -> None:
        self.reserve_calls: list[dict[str, Any]] = []
        self.release_calls: list[dict[str, Any]] = []
        self.lease = object()

    async def reserve(self, **kwargs: Any) -> object:
        self.reserve_calls.append(kwargs)
        return self.lease

    async def release(self, **kwargs: Any) -> None:
        self.release_calls.append(kwargs)


class _Engine:
    def __init__(self, service: _Service) -> None:
        self.service = service
        self.default_sampling_params_list: list[object] = []

    def get_persistent_state_service(self) -> _Service:
        return self.service


class _Serving:
    def __init__(self, engine: _Engine) -> None:
        self.engine_client = engine

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
