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


def _persistent_state_runtime_values() -> dict[str, Any]:
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
        "persistent_state_admission_waiter_capacity": 8,
        "persistent_state_admission_max_inflight_reserves": 2,
        "persistent_state_admission_dispatch_budget": 1,
        "persistent_state_admission_aging_threshold_s": 0.05,
        "persistent_state_admission_wait_timeout_s": 0.10,
        "persistent_state_admission_retry_floor_ms": 10,
        "persistent_state_admission_retry_jitter_ms": 0,
        "persistent_state_recovery_backoff_s": (0.01, 0.02),
        "persistent_state_release_convergence_timeout_s": 60.01,
        "streaming_unadmitted_connection_timeout_s": 20.21,
        "persistent_state_service_profile_trailing_rounds": 3,
        "persistent_state_startup_priming_timeout_s": 120.0,
        "persistent_state_service_profile_derating_factor": 0.5,
        "persistent_state_admission_policy": "profile",
    }


class _StartupProvider:
    def build_priming_budget_descriptor(
        self,
        *,
        configured_population_ceiling: int,
        trailing_rounds: int,
    ) -> Any:
        from vllm_omni.engine.persistent_state_priming import (
            ServicePrimingBudgetCell,
            ServicePrimingBudgetDescriptor,
        )

        return ServicePrimingBudgetDescriptor(
            policy_version="test-v1",
            configured_population_ceiling=configured_population_ceiling,
            cells=(
                ServicePrimingBudgetCell(
                    geometry_id=0,
                    tier_id="test",
                    max_active_population=1,
                    repetitions=trailing_rounds + 1,
                ),
            ),
        )


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
        self.reserve_error: Exception | None = None
        self.ready = True
        self.lease = SimpleNamespace(
            engine_epoch="epoch-a",
            generation=1,
            binding_token="binding-a",
        )
        self.runtime_config = SimpleNamespace(
            accepted_audio_budget_s=30.0,
            accepted_audio_capacity_samples=480_000,
            max_retained_transcript_bytes=1 << 20,
            max_session_samples=None,
            session_configuration_timeout_s=0.05,
            unadmitted_connection_timeout_s=0.20,
            admission_wait_timeout_s=0.10,
            admission_retry_floor_ms=10,
            admission_retry_jitter_ms=0,
            session_idle_timeout_s=60.0,
            session_finalization_timeout_s=40.0,
        )

    async def reserve(self, **kwargs: Any) -> object:
        self.reserve_calls.append(kwargs)
        if self.reserve_error is not None:
            raise self.reserve_error
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


def _start_pre_admission_timeouts(connection: RealtimeConnection) -> None:
    start = getattr(connection, "_start_pre_admission_timeouts", None)
    if not callable(start):
        pytest.fail(
            "PORT-SESS-012 missing orthogonal pre-admission clock authority",
            pytrace=False,
        )
    start()


def _require_standard_update(websocket: _WebSocket) -> dict[str, Any]:
    updates = [
        event for event in websocket.sent if event.get("type") == "session.updated"
    ]
    if len(updates) != 1:
        pytest.fail(
            "PORT-SESS-010 expected exactly one standard session.updated "
            f"event, observed {len(updates)}",
            pytrace=False,
        )
    return updates[0]


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_connection_entry_arms_the_total_unadmitted_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-SESS-012: the real connection entry owns both clocks."""

    connection, websocket, service = _connection()
    connection.session_configuration_timeout = 0.20
    service.runtime_config.unadmitted_connection_timeout_s = 0.01

    async def hold_transport(_connection: object) -> None:
        await asyncio.sleep(0.03)

    monkeypatch.setattr(
        RealtimeConnection.__mro__[1],
        "handle_connection",
        hold_transport,
    )

    await connection.handle_connection()

    assert websocket.sent, (
        "PORT-SESS-012 connection entry did not arm total unadmitted timeout"
    )
    assert websocket.sent[-1]["code"] == "capacity_exhausted"
    assert websocket.closed == [1013]


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
    connection, websocket, service = _connection()

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr"}
    )

    assert len(service.reserve_calls) == 1
    call = service.reserve_calls[0]
    assert call["operation_id"]
    assert call["session_key"]
    assert call["schema_id"]
    assert call["profile_id"]
    assert call.get("service_interval_ms") == 560, (
        "PORT-STATE-025 requires the admitted cadence at reserve"
    )
    assert connection._state_lease is service.lease
    assert connection._is_model_validated is True
    updated = _require_standard_update(websocket)
    assert updated["session"] == {
        "model": "nemotron-asr",
        "cadence": "560ms",
        "locale": "auto",
        "endpointing": {
            "mode": "greedy_blank",
            "stop_history_ms": 800,
            "residue_frames": 2,
        },
    }, "PORT-SESS-010 requires the full resolved configuration"
    assert not {
        "binding_token",
        "engine_epoch",
        "operation_id",
        "request_id",
        "session_key",
    }.intersection(updated["session"])
    assert all(event["type"] != "session.admitted" for event in websocket.sent)


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_standard_update_ack_waits_for_manager_admission() -> None:
    """@spec PORT-STATE-004 / PORT-SESS-010."""

    connection, websocket, service = _connection()
    allow_reserve = asyncio.Event()

    async def delayed_reserve(**kwargs: Any) -> object:
        service.reserve_calls.append(kwargs)
        await allow_reserve.wait()
        return service.lease

    service.reserve = delayed_reserve  # type: ignore[method-assign]
    update = asyncio.create_task(
        connection.handle_event(
            {
                "type": "session.update",
                "model": "nemotron-asr",
                "cadence": "320ms",
                "locale": "en-US",
                "endpointing": {
                    "mode": "greedy_blank",
                    "stop_history_ms": 640,
                    "residue_frames": 1,
                },
            }
        )
    )
    await asyncio.sleep(0)

    assert service.reserve_calls
    assert websocket.sent == []
    assert connection._is_model_validated is False
    allow_reserve.set()
    await update

    assert connection._is_model_validated is True
    assert len(websocket.sent) == 1
    updated = _require_standard_update(websocket)
    assert updated["session"] == {
        "model": "nemotron-asr",
        "cadence": "320ms",
        "locale": "en-US",
        "endpointing": {
            "mode": "greedy_blank",
            "stop_history_ms": 640,
            "residue_frames": 1,
        },
    }, "PORT-SESS-010 requires requested values after resolution"


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_standard_ack_follows_real_session_construction_and_observer_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-STATE-004 / PORT-SESS-010 / PORT-OBS-003."""

    connection, websocket, service = _connection()
    events: list[str] = []
    observer = object()
    connection._observer = observer
    connection.serving.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(endpoint_history_capacity_frames=12)
    )

    original_reserve = service.reserve

    async def reserve(**kwargs: Any) -> object:
        events.append("reserve")
        return await original_reserve(**kwargs)

    service.reserve = reserve  # type: ignore[method-assign]

    def construct(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args
        assert kwargs["observer"] is observer
        events.append("session-and-observer-open")
        return SimpleNamespace(park_token_id=99)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.realtime_connection."
        "NemotronRealtimeSession.from_model_config",
        construct,
    )
    original_send = websocket.send_text

    async def send_text(value: str) -> None:
        payload = json.loads(value)
        if payload["type"] == "session.updated":
            events.append("session.updated")
        await original_send(value)

    websocket.send_text = send_text  # type: ignore[method-assign]

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr"}
    )

    assert events == [
        "reserve",
        "session-and-observer-open",
        "session.updated",
    ], "PORT-SESS-010 missing post-construction standard acknowledgement"


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_construction_failure_releases_without_standard_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-STATE-014 / PORT-SESS-010."""

    connection, websocket, service = _connection()
    connection.serving.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(endpoint_history_capacity_frames=12)
    )

    def reject(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("construction failed")

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.realtime_connection."
        "NemotronRealtimeSession.from_model_config",
        reject,
    )

    with pytest.raises(RuntimeError, match="construction failed"):
        await connection.handle_event(
            {"type": "session.update", "model": "nemotron-asr"}
        )

    assert all(event["type"] != "session.updated" for event in websocket.sent)
    assert len(service.release_calls) == 1
    assert service.release_calls[0]["reason"] == "configuration_error"


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_audio_racing_admission_is_rejected_before_standard_ack() -> None:
    """@spec PORT-SESS-010: socket serialization is not admission."""

    connection, websocket, service = _connection()
    allow_reserve = asyncio.Event()

    async def delayed_reserve(**kwargs: Any) -> object:
        service.reserve_calls.append(kwargs)
        await allow_reserve.wait()
        return service.lease

    service.reserve = delayed_reserve  # type: ignore[method-assign]
    update = asyncio.create_task(
        connection.handle_event(
            {"type": "session.update", "model": "nemotron-asr"}
        )
    )
    await asyncio.sleep(0)
    audio = base64.b64encode(b"\x01\x00").decode()
    await connection.handle_event(
        {"type": "input_audio_buffer.append", "audio": audio}
    )

    assert len(websocket.sent) == 1
    assert websocket.sent[0]["type"] == "error"
    assert websocket.sent[0]["code"] == "model_not_validated"
    allow_reserve.set()
    await update
    _require_standard_update(websocket)


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_permitted_locale_update_acks_only_after_application() -> None:
    """@spec PORT-LID-001 / PORT-SESS-010 / PORT-SESS-011."""

    connection, websocket, service = _connection()
    applied: list[str] = []

    class _Session:
        async def update_locale(self, locale: str) -> None:
            assert websocket.sent == []
            applied.append(locale)

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr"}
    )
    _require_standard_update(websocket)
    websocket.sent.clear()
    connection._nemotron_session = _Session()  # type: ignore[assignment]

    await connection.handle_event(
        {
            "type": "session.update",
            "model": "nemotron-asr",
            "locale": "de-DE",
        }
    )

    assert applied == ["de-DE"], (
        "PORT-LID-001 / PORT-SESS-010 locale update was not applied"
    )
    updated = _require_standard_update(websocket)
    assert updated["session"] == {
        "model": "nemotron-asr",
        "cadence": "560ms",
        "locale": "de-DE",
        "endpointing": {
            "mode": "greedy_blank",
            "stop_history_ms": 800,
            "residue_frames": 2,
        },
    }
    assert len(service.reserve_calls) == 1


def _service_error(name: str, *args: Any, **kwargs: Any) -> Exception:
    from vllm_omni.engine import persistent_state_service as service_module

    try:
        error_cls = getattr(service_module, name)
    except AttributeError:
        pytest.fail(f"PORT-STATE-024 missing typed {name}", pytrace=False)
    try:
        return error_cls(*args, **kwargs)
    except TypeError:
        pytest.fail(
            f"PORT-STATE-024 {name} lacks the approved typed metadata",
            pytrace=False,
        )


def test_pin_inherited_error_event_accepts_omni_retry_metadata() -> None:
    """Green upstream-assumption pin; Phase 5 behavior tests remain red.

    @spec PORT-SESS-011 / PORT-MIG-006
    """
    from vllm.entrypoints.speech_to_text.realtime.protocol import ErrorEvent

    event = ErrorEvent(
        type="error",
        error="retry later",
        code="capacity_exhausted",
        retry_after_ms=125,
    )

    assert event.model_dump(exclude_none=True)["retry_after_ms"] == 125


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_retryable_shed_carries_backoff_and_keeps_socket_open() -> None:
    # @spec PORT-STATE-024 / PORT-SESS-011
    connection, websocket, service = _connection()
    service.reserve_error = _service_error(
        "PersistentStateBackpressure",
        "service budget exhausted",
        retry_after_ms=125,
    )

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr", "cadence": "320ms"}
    )

    assert websocket.sent[-1]["code"] == "capacity_exhausted"
    assert websocket.sent[-1]["retry_after_ms"] == 125
    assert websocket.closed == []
    assert connection._is_connected is True
    assert connection._is_model_validated is False
    assert all(event["type"] != "session.updated" for event in websocket.sent)


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_unsupported_interval_is_nonretryable_and_keeps_socket_open() -> None:
    # @spec PORT-STATE-024 / PORT-STATE-026 / PORT-SESS-011
    connection, websocket, service = _connection()
    service.reserve_error = _service_error(
        "PersistentStateUnsupportedServiceInterval",
        requested_interval_ms=80,
        resolved_envelope="profile-a",
    )

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr", "cadence": "80ms"}
    )

    error = websocket.sent[-1]
    assert error["code"] == "unsupported_service_interval"
    assert "80" in error["error"]
    assert "retry_after_ms" not in error
    assert websocket.closed == []
    assert connection._is_model_validated is False
    assert all(event["type"] != "session.updated" for event in websocket.sent)


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_definitive_denial_retry_mints_fresh_identity() -> None:
    # @spec PORT-SESS-011
    connection, websocket, service = _connection()
    service.reserve_error = _service_error(
        "PersistentStateBackpressure",
        "execution claims exhausted",
        retry_after_ms=50,
    )
    update = {
        "type": "session.update",
        "model": "nemotron-asr",
        "cadence": "560ms",
    }

    await connection.handle_event(update)
    service.reserve_error = None
    await asyncio.sleep(0.06)
    await connection.handle_event(update)

    assert websocket.sent[0]["code"] == "capacity_exhausted"
    _require_standard_update(websocket)
    assert len(service.reserve_calls) == 2
    first, second = service.reserve_calls
    assert first["operation_id"] != second["operation_id"]
    assert first["session_key"] != second["session_key"]
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
async def test_total_unadmitted_timeout_releases_a_late_reserve_result() -> None:
    # @spec PORT-SESS-012 / PORT-STATE-014 / PORT-STATE-028
    connection, websocket, service = _connection()
    service.runtime_config.session_configuration_timeout_s = 0.10
    service.runtime_config.unadmitted_connection_timeout_s = 0.01
    allow_reserve = asyncio.Event()

    async def delayed_reserve(**kwargs: Any) -> object:
        service.reserve_calls.append(kwargs)
        await allow_reserve.wait()
        return service.lease

    service.reserve = delayed_reserve  # type: ignore[method-assign]
    _start_pre_admission_timeouts(connection)
    configure = asyncio.create_task(
        connection.handle_event(
            {"type": "session.update", "model": "nemotron-asr"}
        )
    )
    await asyncio.sleep(0.02)
    allow_reserve.set()
    await configure

    assert websocket.sent[-1]["code"] == "capacity_exhausted"
    assert websocket.closed == [1013]
    assert connection._is_model_validated is False
    assert len(service.release_calls) == 1
    assert service.release_calls[0]["reason"] == (
        "unadmitted_connection_timeout"
    )


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_configuration_clock_stops_before_server_admission_wait() -> None:
    """@spec PORT-SESS-012: client think time cannot charge server wait."""

    connection, websocket, service = _connection()
    connection.session_configuration_timeout = 0.01
    service.runtime_config.session_configuration_timeout_s = 0.01
    service.runtime_config.unadmitted_connection_timeout_s = 0.20
    allow_reserve = asyncio.Event()

    async def delayed_reserve(**kwargs: Any) -> object:
        service.reserve_calls.append(kwargs)
        await allow_reserve.wait()
        return service.lease

    service.reserve = delayed_reserve  # type: ignore[method-assign]
    _start_pre_admission_timeouts(connection)
    configure = asyncio.create_task(
        connection.handle_event(
            {"type": "session.update", "model": "nemotron-asr"}
        )
    )
    await asyncio.sleep(0.03)

    assert websocket.sent == []
    assert websocket.closed == []
    allow_reserve.set()
    await configure
    _require_standard_update(websocket)


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_total_unadmitted_lifetime_never_resets_and_closes_1013() -> None:
    """@spec PORT-SESS-011 / PORT-SESS-012."""

    connection, websocket, service = _connection()
    service.runtime_config.session_configuration_timeout_s = 0.02
    service.runtime_config.unadmitted_connection_timeout_s = 0.05
    service.runtime_config.admission_wait_timeout_s = 0.20
    allow_reserve = asyncio.Event()

    async def delayed_reserve(**kwargs: Any) -> object:
        service.reserve_calls.append(kwargs)
        await allow_reserve.wait()
        return service.lease

    service.reserve = delayed_reserve  # type: ignore[method-assign]
    _start_pre_admission_timeouts(connection)
    update = {"type": "session.update", "model": "nemotron-asr"}
    first = asyncio.create_task(connection.handle_event(update))
    await asyncio.sleep(0.02)
    duplicate = asyncio.create_task(connection.handle_event(update))
    await asyncio.sleep(0.04)

    assert len(service.reserve_calls) == 1
    assert websocket.closed == [1013]
    assert websocket.sent[-1]["code"] == "capacity_exhausted"
    assert all(event["type"] != "session.updated" for event in websocket.sent)
    allow_reserve.set()
    await asyncio.gather(first, duplicate)
    assert len(service.release_calls) == 1
    assert service.release_calls[0]["reason"] == "unadmitted_connection_timeout"


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_retry_hint_beyond_remaining_lifetime_closes_fresh_retry() -> None:
    """@spec PORT-SESS-011 / PORT-SESS-012: impossible same-socket advice."""

    connection, websocket, service = _connection()
    service.runtime_config.unadmitted_connection_timeout_s = 0.03
    _start_pre_admission_timeouts(connection)
    await asyncio.sleep(0.02)
    service.reserve_error = _service_error(
        "PersistentStateBackpressure",
        "controller full",
        retry_after_ms=25,
    )

    await connection.handle_event(
        {"type": "session.update", "model": "nemotron-asr"}
    )

    assert websocket.sent[-1]["code"] == "capacity_exhausted"
    assert websocket.sent[-1]["retry_after_ms"] == 25
    assert websocket.closed == [1013]
    assert connection._is_connected is False


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_definitive_refusal_cycles_cannot_extend_socket_lifetime() -> None:
    """@spec PORT-SESS-011 / PORT-SESS-012: total lifetime never rearms."""

    connection, websocket, service = _connection()
    service.runtime_config.session_configuration_timeout_s = 0.02
    service.runtime_config.unadmitted_connection_timeout_s = 0.07
    service.reserve_error = _service_error(
        "PersistentStateBackpressure",
        "controller full",
        retry_after_ms=5,
    )
    _start_pre_admission_timeouts(connection)
    update = {"type": "session.update", "model": "nemotron-asr"}

    await connection.handle_event(update)
    await asyncio.sleep(0.03)
    await connection.handle_event(update)
    await asyncio.sleep(0.05)

    assert len(service.reserve_calls) == 2
    assert websocket.closed == [1013]
    assert connection._is_connected is False
    assert websocket.sent[-1]["code"] == "capacity_exhausted"


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
    from vllm_omni.entrypoints.openai import api_server

    class _PersistentModel:
        supports_persistent_state = True
        persistent_state_startup_provider = _StartupProvider()

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
                "resident_state_scatter_warmup_complete": True,
                "schema_id": "schema",
                "profile_id": "profile",
            }

    monkeypatch.setattr(model_loader, "get_model_cls", lambda _config: _PersistentModel)

    class _PreparedService:
        ready = True

        def shutdown(self) -> None:
            pass

    async def prepare(**kwargs: Any) -> Any:
        snapshot = await kwargs["stage_client"].call_utility_async(
            "persistent_state_snapshot"
        )
        assert snapshot["resident_state_scatter_warmup_complete"] is True
        return _PreparedService()

    monkeypatch.setattr(api_server, "prepare_persistent_state_service", prepare)
    engine = object.__new__(AsyncOmni)
    stage = _Stage()
    engine.engine = SimpleNamespace(stage_clients=[stage])
    engine._persistent_state_service = None

    await api_server._install_persistent_state_service(
        engine,
        SimpleNamespace(
            model_config=object(),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            additional_config={
                **_persistent_state_runtime_values(),
                "persistent_state_max_tombstones": 40,
            },
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
    from vllm_omni.entrypoints.openai import api_server

    class _PersistentModel:
        supports_persistent_state = True
        persistent_state_startup_provider = _StartupProvider()

    class _Stage:
        async def call_utility_async(self, name: str, *args: Any) -> dict[str, Any]:
            del name, args
            raise RuntimeError("inventory failed")

    monkeypatch.setattr(model_loader, "get_model_cls", lambda _config: _PersistentModel)

    async def prepare(**kwargs: Any) -> Any:
        return await kwargs["stage_client"].call_utility_async(
            "persistent_state_snapshot"
        )

    monkeypatch.setattr(api_server, "prepare_persistent_state_service", prepare)
    engine = object.__new__(AsyncOmni)
    engine.engine = SimpleNamespace(stage_clients=[_Stage()])
    engine._persistent_state_service = None

    with pytest.raises(RuntimeError, match="inventory failed"):
        await api_server._install_persistent_state_service(
            engine,
            SimpleNamespace(
                model_config=object(),
                scheduler_config=SimpleNamespace(max_num_seqs=4),
                additional_config={
                    **_persistent_state_runtime_values(),
                    "persistent_state_max_tombstones": 40,
                },
            ),
        )

    assert engine._persistent_state_service is None
