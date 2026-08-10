# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the API-to-engine persistent-state service."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _service_module() -> Any:
    try:
        return importlib.import_module("vllm_omni.engine.persistent_state_service")
    except ModuleNotFoundError:
        pytest.fail(
            "PORT-STATE-004 missing PersistentStateService module",
            pytrace=False,
        )


def test_async_omni_installs_and_returns_one_exact_service() -> None:
    # @spec PORT-STATE-004 / PORT-RTC-003
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    engine = object.__new__(AsyncOmni)
    setattr(engine, "_persistent_state_service", None)
    service = object()

    with pytest.raises(RuntimeError, match="persistent state.*not installed|not installed"):
        engine.get_persistent_state_service()

    engine.install_persistent_state_service(service)
    assert engine.get_persistent_state_service() is service

    with pytest.raises(RuntimeError, match="already installed|duplicate|mismatch"):
        engine.install_persistent_state_service(service)


def test_service_api_is_async_and_operation_id_explicit() -> None:
    # @spec PORT-STATE-004 / PORT-STATE-012 / PORT-STATE-013
    service_cls = _service_module().PersistentStateService
    reserve = inspect.signature(service_cls.reserve)
    release = inspect.signature(service_cls.release)

    assert inspect.iscoroutinefunction(service_cls.reserve)
    assert inspect.iscoroutinefunction(service_cls.release)
    assert "operation_id" in reserve.parameters
    assert "session_key" in reserve.parameters
    assert "schema_id" in reserve.parameters
    assert "profile_id" in reserve.parameters
    assert "service_interval_ms" in reserve.parameters, (
        "PORT-STATE-025 missing reserve-to-release cadence authority"
    )
    assert "operation_id" in release.parameters
    assert "lease" in release.parameters
    assert "reason" in release.parameters
    constructor = inspect.signature(service_cls)
    assert "admission_config" in constructor.parameters, (
        "PORT-STATE-027 missing resolved controller configuration"
    )
    assert "host_fatal_callback" in constructor.parameters, (
        "PORT-STATE-014 missing host-fatal recovery escalation"
    )


def test_refusal_types_encode_retryability_without_message_matching() -> None:
    # @spec PORT-STATE-024
    module = _service_module()
    try:
        shed = module.PersistentStateBackpressure(
            "pool full",
            retry_after_ms=75,
            cause="controller_full",
        )
        capacity = module.PersistentStateCapacityExhausted(
            "physical pool exhausted",
            retry_after_ms=75,
            cause="hard_pressure",
            binding_authority="physical_slots",
        )
        unavailable = module.PersistentStateServiceUnavailable(
            "control path closed"
        )
    except TypeError:
        pytest.fail(
            "PORT-STATE-024 missing typed retry metadata on refusal errors",
            pytrace=False,
        )
    unsupported_cls = getattr(
        module,
        "PersistentStateUnsupportedServiceInterval",
        None,
    )
    if unsupported_cls is None:
        pytest.fail(
            "PORT-STATE-024 missing unsupported-service-interval type",
            pytrace=False,
        )
    unsupported = unsupported_cls(
        requested_interval_ms=80,
        resolved_envelope="profile-a",
    )

    assert isinstance(capacity, module.PersistentStateBackpressure)
    assert shed.retryable is True
    assert shed.retry_after_ms == 75
    assert shed.cause == "controller_full"
    assert capacity.retryable is True
    assert capacity.cause == "hard_pressure"
    assert capacity.binding_authority == "physical_slots"
    assert shed.telemetry_fields == {
        "cause": "controller_full",
        "retry_after_ms": 75,
    }
    assert capacity.telemetry_fields == {
        "binding_authority": "physical_slots",
        "cause": "hard_pressure",
        "retry_after_ms": 75,
    }
    assert not {
        "attempt_id",
        "connection_id",
        "operation_id",
        "request_id",
        "session_id",
    }.intersection(capacity.telemetry_fields)
    assert unavailable.retryable is False
    assert unsupported.retryable is False
    assert unsupported.requested_interval_ms == 80
    assert unsupported.retry_after_ms is None


def test_service_uses_the_existing_utility_boundary_only() -> None:
    # @spec PORT-STATE-013 / PORT-MIG-006
    module = _service_module()
    source = inspect.getsource(module.PersistentStateService)

    assert "call_utility_async" in source
    assert "persistent_state_reserve" in source
    assert "persistent_state_release" in source
    assert "persistent_state_snapshot" in source
    assert "EngineCoreRequestType" not in source
    assert "MODELS_CONFIG_MAP" not in source


def test_post_submit_timeout_reconciles_the_same_operation_under_shield() -> None:
    # @spec PORT-STATE-012 / PORT-STATE-013 / PORT-SESS-011
    service_cls = _service_module().PersistentStateService
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            service_cls.reserve,
            service_cls._reserve_direct,
            service_cls._submit_admission,
        )
    )

    assert "shield" in source
    assert "operation_id" in source
    assert "uuid" not in source.lower()
    assert "wait_for" in source or "timeout" in source
    assert "reconcil" in source.lower()


def test_cleanup_capacity_and_priority_are_declared_separately_from_reserve() -> None:
    # @spec PORT-STATE-013
    service_cls = _service_module().PersistentStateService
    source = inspect.getsource(service_cls)
    dispatcher_sources = [
        inspect.getsource(method)
        for _, method in inspect.getmembers(service_cls, inspect.isfunction)
        if "reserve_queue" in inspect.getsource(method)
        and "cleanup_queue" in inspect.getsource(method)
    ]

    assert "reserve_queue" in source
    assert "cleanup_queue" in source
    assert dispatcher_sources
    assert any(
        method_source.index("cleanup_queue")
        < method_source.index("reserve_queue")
        for method_source in dispatcher_sources
    )
    assert "coalesc" in source.lower()


def test_engine_core_process_exposes_only_named_omni_utility_methods() -> None:
    # @spec PORT-STATE-013 / PORT-MIG-006
    from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

    for method_name in (
        "persistent_state_snapshot",
        "persistent_state_reserve",
        "persistent_state_release",
    ):
        method = getattr(StageEngineCoreProc, method_name)
        source = inspect.getsource(method)
        assert "persistent_state" in source
        assert "EngineCoreRequestType" not in source


def test_engine_snapshot_attests_completed_resident_scatter_warmup() -> None:
    """@spec PORT-ADV-003 / ENV-MIG-012."""
    from tests.model_executor.persistent_state._helpers import (
        make_manager,
        require_persistent_state_module,
    )
    from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

    manager, _, _ = make_manager(
        require_persistent_state_module(), num_gpu_blocks=3
    )
    manager.resident_state_scatter_warmup_complete = True
    core = object.__new__(StageEngineCoreProc)
    core.vllm_config = SimpleNamespace(
        additional_config=dict(_EXPLICIT_A36_ENVELOPE)
    )
    core.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=(manager,))
        )
    )

    snapshot = core.persistent_state_snapshot()

    assert snapshot.get("resident_state_scatter_warmup_complete") is True, (
        "ENV-MIG-012 engine inventory omitted scatter-warmup attestation"
    )


def _publish_warmup_attestation(core: Any) -> None:
    publish = getattr(
        core,
        "_publish_persistent_state_warmup_attestation",
        None,
    )
    if not callable(publish):
        pytest.fail(
            "ENV-MIG-012 missing worker-to-manager warmup attestation join",
            pytrace=False,
        )
    publish()


@pytest.mark.parametrize("worker_attestations", ([True], [True, True]))
def test_engine_core_publishes_all_worker_warmup_attestations(
    worker_attestations: list[bool],
) -> None:
    """@spec PORT-ADV-003 / ENV-MIG-012."""
    from tests.model_executor.persistent_state._helpers import (
        make_manager,
        require_persistent_state_module,
    )
    from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

    manager, _, _ = make_manager(
        require_persistent_state_module(), num_gpu_blocks=3
    )
    core = object.__new__(StageEngineCoreProc)
    core.model_executor = SimpleNamespace(
        collective_rpc=lambda method: (
            worker_attestations
            if method == "persistent_state_warmup_attestation"
            else pytest.fail(f"unexpected worker RPC: {method}")
        )
    )
    core.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=(manager,))
        )
    )

    _publish_warmup_attestation(core)

    assert manager.resident_state_scatter_warmup_complete is True


@pytest.mark.parametrize("worker_attestations", ([], [False], [True, False]))
def test_engine_core_rejects_incomplete_worker_warmup_attestation(
    worker_attestations: list[bool],
) -> None:
    """@spec PORT-ADV-003 / ENV-MIG-012."""
    from tests.model_executor.persistent_state._helpers import (
        make_manager,
        require_persistent_state_module,
    )
    from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

    manager, _, _ = make_manager(
        require_persistent_state_module(), num_gpu_blocks=3
    )
    core = object.__new__(StageEngineCoreProc)
    core.model_executor = SimpleNamespace(
        collective_rpc=lambda method: worker_attestations
    )
    core.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=(manager,))
        )
    )

    with pytest.raises(RuntimeError, match="worker.*warmup.*attestation"):
        _publish_warmup_attestation(core)

    assert not hasattr(manager, "resident_state_scatter_warmup_complete")


def test_engine_core_init_automatically_publishes_worker_warmup_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-ADV-003 / ENV-MIG-012."""
    from vllm.v1.engine.core import EngineCoreProc

    from tests.model_executor.persistent_state._helpers import (
        make_manager,
        require_persistent_state_module,
    )
    from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

    manager, _, _ = make_manager(
        require_persistent_state_module(), num_gpu_blocks=3
    )
    events: list[str] = []

    def fake_base_init(core: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        events.append("base-init")
        core.model_executor = SimpleNamespace(
            collective_rpc=lambda method: (
                events.append(method) or [True]
            )
        )
        core.scheduler = SimpleNamespace(
            kv_cache_manager=SimpleNamespace(
                coordinator=SimpleNamespace(single_type_managers=(manager,))
            )
        )

    monkeypatch.setattr(EngineCoreProc, "__init__", fake_base_init)

    StageEngineCoreProc(object(), False, "", object, False)

    assert events == [
        "base-init",
        "persistent_state_warmup_attestation",
    ]
    assert manager.resident_state_scatter_warmup_complete is True


def test_health_includes_selected_service_readiness() -> None:
    # @spec PORT-STATE-004 / PORT-STATE-015 / PORT-STATE-016
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    source = inspect.getsource(AsyncOmni.check_health)
    assert "persistent_state" in source
    assert "check_health" in source or "ready" in source


def test_service_has_no_process_global_or_application_plugin_dependency() -> None:
    # @spec PORT-STATE-004 / PORT-RTC-003
    module = _service_module()
    source = inspect.getsource(module)

    assert "ApplicationPlugin" not in source
    assert "app.state" not in source
    assert "global " not in source
    assert "prometheus" not in source.lower()
    assert "vllm_omni.metrics" not in source


def test_location_event_never_exports_a_physical_slot_or_content() -> None:
    # @spec PORT-STATE-015
    event_cls = _service_module().StateLocationEvent
    fields = set(getattr(event_cls, "__annotations__", {}))

    assert {
        "engine_epoch",
        "session_key",
        "generation",
        "location",
        "transition",
    } <= fields
    assert {
        "slot_id",
        "block_id",
        "audio",
        "transcript",
        "state",
    }.isdisjoint(fields)


def test_state_lease_is_an_opaque_complete_logical_binding() -> None:
    # @spec PORT-STATE-004 / PORT-STATE-019
    lease_cls = _service_module().StateLease
    fields = set(getattr(lease_cls, "__annotations__", {}))

    assert {
        "engine_epoch",
        "session_key",
        "generation",
        "schema_id",
        "profile_id",
        "location",
        "binding_token",
    } <= fields
    assert {
        "slot_id",
        "block_id",
        "audio",
        "transcript",
        "state",
    }.isdisjoint(fields)


def test_utility_results_carry_post_commit_projection_authority() -> None:
    # @spec PORT-STATE-012 / PORT-STATE-015
    module = _service_module()

    for result_name in ("StateReserveResult", "StateReleaseResult"):
        result_cls = getattr(module, result_name)
        fields = set(getattr(result_cls, "__annotations__", {}))
        assert {"operation_id", "manager_revision", "resident_count"} <= fields
        assert {"location_event", "location_events"} & fields


def test_service_starts_closed_until_capability_inventory_handshake() -> None:
    # @spec PORT-STATE-004 / PORT-STATE-015
    service_cls = _service_module().PersistentStateService
    source = inspect.getsource(service_cls)

    assert "persistent_state_snapshot" in source
    assert "capabil" in source.lower()
    assert "inventory" in source.lower()
    assert "admission" in source.lower()
    assert "ready" in source.lower()


def test_engine_epoch_change_is_a_named_admission_closed_transition() -> None:
    # @spec PORT-STATE-015 / PORT-STATE-016
    service_cls = _service_module().PersistentStateService
    source = inspect.getsource(service_cls)

    assert "engine_epoch_changed" in source
    assert "close_admission" in source or "admission.close" in source
    assert "snapshot" in source


@dataclass
class _Clock:
    now: float = 100.0

    def __call__(self) -> float:
        return self.now


class _TombstoneStage:
    def __init__(self) -> None:
        self.reserve_calls = 0
        self.release_calls = 0
        self.pending_cleanup_calls = 0
        self.release_failures_remaining = 0
        self.resident = 0
        self.revision = 0

    async def call_utility_async(
        self, name: str, *args: Any
    ) -> dict[str, Any]:
        if name == "persistent_state_snapshot":
            return {
                "engine_epoch": "epoch-a",
                "manager_revision": self.revision,
                "resident_count": self.resident,
                "physical_capacity": 4,
                "safety_reserve": 0,
                "configured_limit": 1,
                "effective_capacity": 1,
                "stage": 0,
                "replica": 0,
                "capabilities": ["resident"],
                "resident_state_scatter_warmup_complete": True,
                "schema_id": "schema-a",
                "profile_id": "profile-a",
                "persistent_state_tombstone_ttl_s": 10.0,
                "persistent_state_max_tombstones": 2,
            }
        if name == "persistent_state_reserve":
            operation_id, session_key, schema_id, profile_id = args
            self.reserve_calls += 1
            self.resident += 1
            self.revision += 1
            lease = {
                "engine_epoch": "epoch-a",
                "session_key": session_key,
                "generation": self.reserve_calls,
                "schema_id": schema_id,
                "profile_id": profile_id,
                "location": "resident",
                "binding_token": f"binding-{self.reserve_calls}",
            }
            return {
                "operation_id": operation_id,
                "manager_revision": self.revision,
                "resident_count": self.resident,
                "lease": lease,
                "location_event": {
                    "engine_epoch": "epoch-a",
                    "session_key": session_key,
                    "generation": self.reserve_calls,
                    "location": "resident",
                    "transition": "reserved",
                },
            }
        if name == "persistent_state_release":
            if self.release_failures_remaining:
                self.release_failures_remaining -= 1
                raise RuntimeError("claimed lease is still running")
            operation_id, lease, _reason = args
            self.release_calls += 1
            self.resident -= 1
            self.revision += 1
            return {
                "operation_id": operation_id,
                "manager_revision": self.revision,
                "resident_count": self.resident,
                "location_event": {
                    "engine_epoch": "epoch-a",
                    "session_key": lease["session_key"],
                    "generation": lease["generation"],
                    "location": "absent",
                    "transition": "released",
                },
            }
        if name == "persistent_state_begin_pending_cleanup":
            self.pending_cleanup_calls += 1
            return True
        raise AssertionError(name)


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_public_reserve_is_unavailable_until_profile_is_sealed() -> None:
    """@spec PORT-STATE-027 / PORT-PERF-005 / ENV-MIG-012."""
    module = _service_module()
    from vllm_omni.engine.persistent_state_admission import (
        AdmissionControllerConfig,
    )

    stage = _TombstoneStage()
    service = module.PersistentStateService(
        stage,
        tombstone_ttl_s=10.0,
        max_tombstones=2,
        admission_config=AdmissionControllerConfig(
            waiter_capacity=4,
            max_inflight_reserves=2,
            dispatch_budget=1,
            aging_threshold_ns=1_000_000,
            admission_wait_timeout_s=0.1,
            retry_floor_ms=10,
            retry_jitter_ms=0,
            recovery_backoff_s=(0.01,),
            release_convergence_timeout_s=0.1,
            supported_intervals_ms=(80, 160, 320, 560, 1120),
        ),
    )

    try:
        await service.reserve(
            operation_id="public-before-seal",
            session_key="session-a",
            schema_id="schema-a",
            profile_id="profile-a",
            service_interval_ms=80,
        )
    except module.PersistentStateServiceUnavailable as error:
        assert any(
            name in str(error)
            for name in ("profile", "seal", "bootstrap", "admission")
        )
    else:
        pytest.fail(
            "PORT-STATE-027 public reserve bypassed the profile seal",
            pytrace=False,
        )

    assert stage.reserve_calls == 0
    assert not service.ready
    service.shutdown()


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_completed_operations_are_bounded_tombstones_and_retry_exactly() -> None:
    # @spec PORT-STATE-012 / PORT-STATE-013
    module = _service_module()
    clock = _Clock()
    stage = _TombstoneStage()
    service = module.PersistentStateService(
        stage,
        tombstone_ttl_s=10.0,
        max_tombstones=2,
        monotonic=clock,
    )

    lease = await service.reserve(
        operation_id="reserve-a",
        session_key="session-a",
        schema_id="schema-a",
        profile_id="profile-a",
    )
    duplicate = await service.reserve(
        operation_id="reserve-a",
        session_key="session-a",
        schema_id="schema-a",
        profile_id="profile-a",
    )
    assert duplicate == lease
    assert stage.reserve_calls == 1

    assert await service.claim_pending_cleanup(lease) is True
    assert await service.claim_pending_cleanup(lease) is True
    assert stage.pending_cleanup_calls == 2

    with pytest.raises(module.PersistentStateBackpressure) as info:
        await service.reserve(
            operation_id="reserve-b",
            session_key="session-b",
            schema_id="schema-a",
            profile_id="profile-a",
        )
    assert info.value.retryable
    assert info.value.cause == "tombstone_horizon"
    assert service.ready

    first_release = await service.release(
        operation_id="release-a",
        lease=lease,
        reason="test",
    )
    duplicate_release = await service.release(
        operation_id="release-a",
        lease=lease,
        reason="test",
    )
    assert duplicate_release == first_release
    assert stage.release_calls == 1

    clock.now += 10.0
    await service.reserve(
        operation_id="reserve-b",
        session_key="session-b",
        schema_id="schema-a",
        profile_id="profile-a",
    )
    assert stage.reserve_calls == 2
    service.shutdown()


_EXPLICIT_A36_ENVELOPE = {
    "persistent_state_safety_reserve_slots": 1,
    "max_resident_sessions": 8,
    "persistent_state_reserve_queue_capacity": 8,
    "persistent_state_operation_timeout_s": 10.0,
    "persistent_state_reconciliation_timeout_s": 30.0,
    "persistent_state_tombstone_ttl_s": 600.0,
    "persistent_state_max_tombstones": 32,
    "persistent_state_pending_claim_timeout_s": 15.0,
    "streaming_session_configuration_timeout_s": 10.0,
    "streaming_session_finalization_timeout_s": 40.0,
    "streaming_accepted_audio_capacity_samples": 480_000,
    "streaming_max_retained_transcript_bytes": 1 << 20,
    "persistent_state_admission_waiter_capacity": 16,
    "persistent_state_admission_max_inflight_reserves": 4,
    "persistent_state_admission_dispatch_budget": 1,
    "persistent_state_admission_aging_threshold_s": 1.0,
    "persistent_state_admission_wait_timeout_s": 5.0,
    "persistent_state_admission_retry_floor_ms": 100,
    "persistent_state_admission_retry_jitter_ms": 25,
    "persistent_state_recovery_backoff_s": (0.1, 0.5, 1.0),
    "persistent_state_release_convergence_timeout_s": 61.0,
    "streaming_unadmitted_connection_timeout_s": 31.125,
    "persistent_state_service_profile_trailing_rounds": 3,
    "persistent_state_service_profile_derating_factor": 0.5,
}


def test_runtime_config_requires_the_complete_a36_envelope() -> None:
    """A36 admission fields have no omission defaults before qualification."""
    # @spec PORT-STATE-012 / PORT-STATE-027 / ENV-MIG-011
    from vllm_omni.engine.persistent_state_config import (
        PersistentStateRuntimeConfig,
    )

    explicit = PersistentStateRuntimeConfig.from_vllm_config(
        SimpleNamespace(additional_config=dict(_EXPLICIT_A36_ENVELOPE))
    )
    for defaults_source in (None, {}):
        try:
            PersistentStateRuntimeConfig.from_vllm_config(
                SimpleNamespace(additional_config=defaults_source)
            )
        except ValueError as error:
            assert "persistent_state_admission" in str(error)
        else:
            pytest.fail(
                "ENV-MIG-011 omitted A36 fields silently resolved",
                pytrace=False,
            )

    assert explicit.tombstone_ttl_s == 600.0
    assert explicit.max_tombstones == 32
    assert explicit.cleanup_queue_capacity == 8
    assert explicit.session_idle_timeout_s == 60.0
    assert explicit.accepted_audio_budget_s == 30.0
    assert explicit.safe_finalization_timeout_s == pytest.approx(32.24)
    assert explicit.session_finalization_timeout_s == 40.0
    assert explicit.max_session_duration_s is None
    assert explicit.admission_waiter_capacity == 16
    assert explicit.admission_max_inflight_reserves == 4
    assert explicit.admission_dispatch_budget == 1
    assert explicit.admission_retry_floor_ms == 100
    assert explicit.unadmitted_connection_timeout_s == pytest.approx(31.125)
    assert explicit.service_profile_trailing_rounds == 3
    assert explicit.service_profile_derating_factor == pytest.approx(0.5)


def test_runtime_config_reports_every_missing_a36_field_together() -> None:
    """@spec ENV-MIG-011: one validation error names the complete missing set."""
    from vllm_omni.engine.persistent_state_config import (
        PersistentStateRuntimeConfig,
    )

    missing = {
        "persistent_state_admission_waiter_capacity",
        "persistent_state_admission_max_inflight_reserves",
        "persistent_state_admission_dispatch_budget",
        "persistent_state_admission_aging_threshold_s",
        "persistent_state_admission_wait_timeout_s",
        "persistent_state_admission_retry_floor_ms",
        "persistent_state_admission_retry_jitter_ms",
        "persistent_state_recovery_backoff_s",
        "persistent_state_release_convergence_timeout_s",
        "streaming_unadmitted_connection_timeout_s",
        "persistent_state_service_profile_trailing_rounds",
        "persistent_state_service_profile_derating_factor",
    }
    old_envelope = {
        key: value
        for key, value in _EXPLICIT_A36_ENVELOPE.items()
        if key not in missing
    }

    try:
        PersistentStateRuntimeConfig.from_vllm_config(
            SimpleNamespace(additional_config=old_envelope)
        )
    except ValueError as error:
        message = str(error)
    else:
        pytest.fail(
            "ENV-MIG-011 missing-field set did not fail validation",
            pytrace=False,
        )

    assert all(name in message for name in missing)


def test_runtime_config_partial_override_stays_self_consistent() -> None:
    """Derived defaults follow an overridden input (ENV-MIG-009)."""
    # @spec ENV-MIG-009
    from vllm_omni.engine.persistent_state_config import (
        PersistentStateRuntimeConfig,
    )

    resolved = PersistentStateRuntimeConfig.from_vllm_config(
        SimpleNamespace(
            additional_config={
                **{
                    key: value
                    for key, value in _EXPLICIT_A36_ENVELOPE.items()
                    if key
                    not in {
                        "persistent_state_reserve_queue_capacity",
                        "persistent_state_max_tombstones",
                    }
                },
                "max_resident_sessions": 16,
            }
        )
    )
    assert resolved.max_resident_sessions == 16
    assert resolved.reserve_queue_capacity == 16
    assert resolved.cleanup_queue_capacity == 16
    assert resolved.max_tombstones == 64

    # Raising the operation timeout raises the derived reconciliation
    # default with it, instead of failing the cross-field check.
    slow = PersistentStateRuntimeConfig.from_vllm_config(
        SimpleNamespace(
            additional_config={
                **{
                    key: value
                    for key, value in _EXPLICIT_A36_ENVELOPE.items()
                    if key != "persistent_state_reconciliation_timeout_s"
                },
                "persistent_state_operation_timeout_s": 45.0,
                "persistent_state_release_convergence_timeout_s": 91.0,
                "streaming_unadmitted_connection_timeout_s": 101.125,
            }
        )
    )
    assert slow.reconciliation_timeout_s == 45.0
    with pytest.raises(ValueError, match="no shorter"):
        PersistentStateRuntimeConfig.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    **_EXPLICIT_A36_ENVELOPE,
                    "persistent_state_operation_timeout_s": 45.0,
                    "persistent_state_reconciliation_timeout_s": 30.0,
                }
            )
        )

    # A larger audio budget lifts the derived finalization default to
    # the published safe drain floor.
    wide = PersistentStateRuntimeConfig.from_vllm_config(
        SimpleNamespace(
            additional_config={
                **{
                    key: value
                    for key, value in _EXPLICIT_A36_ENVELOPE.items()
                    if key != "streaming_session_finalization_timeout_s"
                },
                "streaming_accepted_audio_capacity_samples": 960_000
            }
        )
    )
    assert wide.accepted_audio_budget_s == 60.0
    assert wide.session_finalization_timeout_s == pytest.approx(
        wide.safe_finalization_timeout_s
    )


def test_runtime_config_rejects_null_invalid_and_unsafe_values() -> None:
    """Defaults never launder a bad explicit value (ENV-MIG-009)."""
    # @spec ENV-MIG-009
    from vllm_omni.engine.persistent_state_config import (
        PersistentStateRuntimeConfig,
    )

    for key in _EXPLICIT_A36_ENVELOPE:
        try:
            PersistentStateRuntimeConfig.from_vllm_config(
                SimpleNamespace(additional_config={key: None})
            )
        except ValueError as error:
            assert key in str(error)
        else:
            pytest.fail(
                f"ENV-MIG-009/011 explicit null was accepted for {key}",
                pytrace=False,
            )

    with pytest.raises(ValueError, match="streaming_session_idle_timeout_s"):
        PersistentStateRuntimeConfig.from_vllm_config(
            SimpleNamespace(
                additional_config={"streaming_session_idle_timeout_s": None}
            )
        )

    unprefixed = dict(_EXPLICIT_A36_ENVELOPE)
    unprefixed["session_idle_timeout_s"] = 60.0
    with pytest.raises(ValueError, match="streaming_ prefix"):
        PersistentStateRuntimeConfig.from_vllm_config(
            SimpleNamespace(additional_config=unprefixed)
        )

    too_short = dict(_EXPLICIT_A36_ENVELOPE)
    too_short["streaming_session_finalization_timeout_s"] = 32.0
    with pytest.raises(ValueError, match="safe drain bound"):
        PersistentStateRuntimeConfig.from_vllm_config(
            SimpleNamespace(additional_config=too_short)
        )

    with pytest.raises(ValueError, match="must be a mapping"):
        PersistentStateRuntimeConfig.from_vllm_config(
            SimpleNamespace(additional_config="not-a-mapping")
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"persistent_state_admission_dispatch_budget": 5},
            "dispatch|inflight",
        ),
        (
            {"persistent_state_admission_aging_threshold_s": 5.0},
            "aging|wait",
        ),
        (
            {"persistent_state_admission_retry_floor_ms": 0},
            "retry.*floor",
        ),
        (
            {"streaming_unadmitted_connection_timeout_s": 30.0},
            "unadmitted|retry cycle",
        ),
        (
            {"persistent_state_release_convergence_timeout_s": 60.0},
            "release.*convergence|reconciliation",
        ),
        (
            {"persistent_state_recovery_backoff_s": ()},
            "recovery.*backoff",
        ),
    ],
)
def test_runtime_config_enforces_a36_cross_field_ordering(
    updates: dict[str, Any],
    message: str,
) -> None:
    """@spec ENV-MIG-011: unsafe controller timing cannot reach serving."""
    from vllm_omni.engine.persistent_state_config import (
        PersistentStateRuntimeConfig,
    )

    try:
        PersistentStateRuntimeConfig.from_vllm_config(
            SimpleNamespace(
                additional_config={**_EXPLICIT_A36_ENVELOPE, **updates}
            )
        )
    except ValueError as error:
        assert re.search(message, str(error))
    else:
        pytest.fail(
            f"ENV-MIG-011 unsafe cross-field values were accepted: {updates}",
            pytrace=False,
        )


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_release_operation_can_reconcile_after_transient_terminal_race() -> None:
    # @spec PORT-STATE-013 / PORT-STATE-014
    module = _service_module()
    stage = _TombstoneStage()
    stage.release_failures_remaining = 1
    service = module.PersistentStateService(
        stage,
        tombstone_ttl_s=10.0,
        max_tombstones=2,
    )
    lease = await service.reserve(
        operation_id="reserve-a",
        session_key="session-a",
        schema_id="schema-a",
        profile_id="profile-a",
    )

    with pytest.raises(
        module.PersistentStateServiceUnavailable,
        match="still running",
    ):
        await service.release(
            operation_id="release-a",
            lease=lease,
            reason="cleanup",
        )
    await asyncio.sleep(0)
    result = await service.release(
        operation_id="release-a",
        lease=lease,
        reason="cleanup",
    )
    assert result.resident_count == 0
    assert stage.release_calls == 1
    service.shutdown()


def test_engine_core_never_evicts_an_unexpired_operation_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-STATE-012 / PORT-STATE-013 / PORT-STATE-015
    import vllm_omni.engine.stage_engine_core_proc as core_module
    from tests.model_executor.persistent_state._helpers import (
        make_manager,
        require_persistent_state_module,
    )
    from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

    clock = _Clock()
    monkeypatch.setattr(core_module.time, "monotonic", clock)
    manager, _, spec = make_manager(
        require_persistent_state_module(), num_gpu_blocks=3
    )
    runtime_values = dict(
        _EXPLICIT_A36_ENVELOPE,
        max_resident_sessions=1,
        persistent_state_reserve_queue_capacity=1,
        persistent_state_admission_max_inflight_reserves=1,
        persistent_state_tombstone_ttl_s=10.0,
        persistent_state_max_tombstones=2,
    )
    core = object.__new__(StageEngineCoreProc)
    core.vllm_config = SimpleNamespace(additional_config=runtime_values)
    core.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=(manager,))
        )
    )

    first = core.persistent_state_reserve(
        "reserve-a", "session-a", spec.schema_id, "default"
    )
    assert (
        core.persistent_state_reserve(
            "reserve-a", "session-a", spec.schema_id, "default"
        )
        == first
    )
    with pytest.raises(RuntimeError, match="tombstone horizon"):
        core.persistent_state_reserve(
            "reserve-b", "session-b", spec.schema_id, "default"
        )

    released = core.persistent_state_release(
        "release-a", first["lease"], "test"
    )
    assert (
        core.persistent_state_release(
            "release-a", first["lease"], "test"
        )
        == released
    )
    assert len(core._persistent_state_control_state["operations"]) == 2

    clock.now += 10.0
    second = core.persistent_state_reserve(
        "reserve-b", "session-b", spec.schema_id, "default"
    )
    assert second["lease"]["session_key"] == "session-b"
    assert len(core._persistent_state_control_state["operations"]) == 1


def _state_core_for_cleanup_tests() -> tuple[Any, Any, Any]:
    from tests.model_executor.persistent_state._helpers import (
        make_manager,
        require_persistent_state_module,
    )
    from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

    manager, _, spec = make_manager(
        require_persistent_state_module(), num_gpu_blocks=4
    )
    runtime_values = dict(
        _EXPLICIT_A36_ENVELOPE,
        max_resident_sessions=2,
        persistent_state_reserve_queue_capacity=2,
        persistent_state_admission_max_inflight_reserves=2,
        persistent_state_tombstone_ttl_s=10.0,
        persistent_state_max_tombstones=4,
    )
    core = object.__new__(StageEngineCoreProc)
    core.vllm_config = SimpleNamespace(additional_config=runtime_values)
    core.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(single_type_managers=(manager,))
        )
    )
    return core, manager, spec


def _claim_payload(lease: dict[str, Any]) -> dict[str, Any]:
    return {
        name: lease[name]
        for name in (
            "engine_epoch",
            "session_key",
            "generation",
            "schema_id",
            "profile_id",
            "binding_token",
        )
    }


def test_pending_cleanup_claim_serializes_against_scheduler_claim() -> None:
    # @spec PORT-STATE-014 / PORT-STATE-019
    core, manager, spec = _state_core_for_cleanup_tests()
    reserved = core.persistent_state_reserve(
        "reserve-a", "session-a", spec.schema_id, "default"
    )
    lease = reserved["lease"]

    assert core.persistent_state_begin_pending_cleanup(lease) is True
    assert core.persistent_state_begin_pending_cleanup(lease) is True
    with pytest.raises(ValueError, match="pending claim mismatch"):
        core.claim_pending_lease(**_claim_payload(lease))

    released = core.persistent_state_release(
        "release-a", lease, "pending_claim_timeout"
    )
    assert released["resident_count"] == 0
    assert manager.get_state_binding("session-a") is None
    revision = core._persistent_state_control_state["revision"]
    with pytest.raises(ValueError, match="stale"):
        core.persistent_state_release(
            "different-release", lease, "late_duplicate"
        )
    assert core._persistent_state_control_state["revision"] == revision


def test_claimed_state_cannot_be_released_until_scheduler_terminality() -> None:
    # @spec PORT-STATE-014 / PORT-STATE-019
    core, manager, spec = _state_core_for_cleanup_tests()
    reserved = core.persistent_state_reserve(
        "reserve-a", "session-a", spec.schema_id, "default"
    )
    lease = reserved["lease"]
    binding = core.claim_pending_lease(**_claim_payload(lease))

    assert core.persistent_state_begin_pending_cleanup(lease) is False
    with pytest.raises(RuntimeError, match="still running|terminal"):
        core.persistent_state_release("release-a", lease, "cleanup")
    assert manager.get_state_binding("session-a") == binding

    core.mark_terminal(binding)
    released = core.persistent_state_release(
        "release-a", lease, "cleanup"
    )
    assert released["resident_count"] == 0
