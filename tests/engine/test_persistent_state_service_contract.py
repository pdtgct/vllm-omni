# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the API-to-engine persistent-state service."""

from __future__ import annotations

import importlib
import inspect
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
    assert "operation_id" in release.parameters
    assert "lease" in release.parameters
    assert "reason" in release.parameters


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
    source = inspect.getsource(_service_module().PersistentStateService.reserve)

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
