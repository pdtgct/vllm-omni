# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manager-to-host metric projection for persistent state."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm_omni.engine.persistent_state_service import (
    PersistentStateCapacityExhausted,
    PersistentStateService,
    PersistentStateServiceUnavailable,
)
from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.streaming import OmniStreamingMetrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _value(prefix: str) -> float | None:
    for line in generate_latest(REGISTRY).decode().splitlines():
        if line.startswith(prefix):
            return float(line.split()[-1])
    return None


async def _reserve_with_interval(
    service: PersistentStateService,
    *,
    service_interval_ms: int = 560,
    **kwargs: Any,
) -> Any:
    if "service_interval_ms" not in inspect.signature(service.reserve).parameters:
        pytest.fail(
            "PORT-STATE-025 missing reserve-to-release service interval",
            pytrace=False,
        )
    return await service.reserve(
        service_interval_ms=service_interval_ms,
        **kwargs,
    )


class _StageClient:
    def __init__(self) -> None:
        self.resident = 0
        self.revision = 0
        self.capacity_error = False
        self.reserve_error: Exception | None = None

    async def call_utility_async(self, method: str, *args: Any) -> dict[str, Any]:
        if method == "persistent_state_snapshot":
            return {
                "engine_epoch": "epoch-1",
                "manager_revision": self.revision,
                "resident_count": self.resident,
                "physical_capacity": 8,
                "safety_reserve": 2,
                "configured_limit": 7,
                "effective_capacity": 6,
                "stage": 0,
                "replica": 0,
                "capabilities": ["resident"],
                "schema_id": "schema-1",
                "profile_id": "profile-1",
            }
        if method == "persistent_state_reserve":
            if self.reserve_error is not None:
                raise self.reserve_error
            if self.capacity_error:
                raise ValueError("persistent-state capacity exhausted")
            operation_id, session_key, schema_id, profile_id = args
            self.resident += 1
            self.revision += 1
            lease = {
                "engine_epoch": "epoch-1",
                "session_key": session_key,
                "generation": self.revision,
                "schema_id": schema_id,
                "profile_id": profile_id,
                "location": "resident",
                "binding_token": f"binding-{self.revision}",
            }
            return {
                "operation_id": operation_id,
                "manager_revision": self.revision,
                "resident_count": self.resident,
                "lease": lease,
                "location_event": {
                    "engine_epoch": "epoch-1",
                    "session_key": session_key,
                    "generation": self.revision,
                    "location": "resident",
                    "transition": "reserved",
                },
            }
        if method == "persistent_state_release":
            operation_id, lease, _reason = args
            self.resident -= 1
            self.revision += 1
            return {
                "operation_id": operation_id,
                "manager_revision": self.revision,
                "resident_count": self.resident,
                "location_event": {
                    "engine_epoch": "epoch-1",
                    "session_key": lease["session_key"],
                    "generation": lease["generation"],
                    "location": "absent",
                    "transition": "released",
                },
            }
        raise AssertionError(method)


def test_snapshot_reserve_and_release_project_manager_inventory() -> None:
    # @spec PORT-OBS-010 / PORT-STATE-015
    async def scenario() -> None:
        stage = _StageClient()
        service = PersistentStateService(stage)
        metrics = OmniStreamingMetrics(
            model_name="state-service-projection", log_stats=True
        )
        await service.check_health()
        service.install_metrics(metrics)

        resident = (
            f'{defs.PERSISTENT_STATE_SLOTS}{{kind="resident",'
            'model_name="state-service-projection",replica="0",stage="0"}'
        )
        assert _value(resident) == 0

        lease = await service.reserve(
            operation_id="reserve-1",
            session_key="session-1",
            schema_id="schema-1",
            profile_id="profile-1",
        )
        assert _value(resident) == 1

        await service.release(
            operation_id="release-1",
            lease=lease,
            reason="test",
        )
        assert _value(resident) == 0
        service.shutdown()

    asyncio.run(scenario())


def test_definitive_capacity_failure_is_counted_once() -> None:
    # @spec PORT-OBS-007
    async def scenario() -> None:
        stage = _StageClient()
        service = PersistentStateService(stage)
        metrics = OmniStreamingMetrics(
            model_name="state-service-rejection", log_stats=True
        )
        await service.check_health()
        service.install_metrics(metrics)
        stage.capacity_error = True
        prefix = (
            f'{defs.STREAMING_ADMISSION_REJECTIONS}_total{{'
            'model_name="state-service-rejection",reason="capacity"}'
        )
        before = _value(prefix) or 0.0

        with pytest.raises(PersistentStateCapacityExhausted):
            await service.reserve(
                operation_id="reserve-capacity",
                session_key="session-capacity",
                schema_id="schema-1",
                profile_id="profile-1",
            )

        assert _value(prefix) == before + 1.0
        service.shutdown()

    asyncio.run(scenario())


def test_unsupported_interval_failure_is_counted_once() -> None:
    # @spec PORT-OBS-001 / PORT-OBS-007 / PORT-STATE-024
    async def scenario() -> None:
        from vllm_omni.engine import persistent_state_service as service_module

        unsupported_cls = getattr(
            service_module,
            "PersistentStateUnsupportedServiceInterval",
            None,
        )
        if unsupported_cls is None:
            pytest.fail(
                "PORT-STATE-024 missing unsupported-service-interval type",
                pytrace=False,
            )
        stage = _StageClient()
        service = PersistentStateService(stage)
        metrics = OmniStreamingMetrics(
            model_name="state-service-unsupported", log_stats=True
        )
        await service.check_health()
        service.install_metrics(metrics)
        stage.reserve_error = RuntimeError(
            "persistent_state_unsupported_service_interval: 80ms exceeds "
            "the empty-pool transaction envelope"
        )
        prefix = (
            f'{defs.STREAMING_ADMISSION_REJECTIONS}_total{{'
            'model_name="state-service-unsupported",reason="unsupported"}'
        )
        before = _value(prefix) or 0.0

        with pytest.raises(unsupported_cls):
            await _reserve_with_interval(
                service,
                service_interval_ms=80,
                operation_id="reserve-unsupported",
                session_key="session-unsupported",
                schema_id="schema-1",
                profile_id="profile-1",
            )

        assert _value(prefix) == before + 1.0
        service.shutdown()

    asyncio.run(scenario())


def test_retryable_shed_uses_capacity_metric_reason() -> None:
    # @spec PORT-OBS-007 / PORT-STATE-024
    async def scenario() -> None:
        stage = _StageClient()
        service = PersistentStateService(stage)
        metrics = OmniStreamingMetrics(
            model_name="state-service-shed", log_stats=True
        )
        await service.check_health()
        service.install_metrics(metrics)
        stage.reserve_error = RuntimeError(
            "persistent_state_horizon_exhausted: retry after tombstone expiry"
        )
        prefix = (
            f'{defs.STREAMING_ADMISSION_REJECTIONS}_total{{'
            'model_name="state-service-shed",reason="capacity"}'
        )
        before = _value(prefix) or 0.0

        with pytest.raises(Exception) as info:
            await _reserve_with_interval(
                service,
                operation_id="reserve-shed",
                session_key="session-shed",
                schema_id="schema-1",
                profile_id="profile-1",
            )

        assert getattr(info.value, "retryable", False)
        assert _value(prefix) == before + 1.0
        service.shutdown()

    asyncio.run(scenario())


def test_service_metric_sink_is_exactly_once_and_prometheus_free() -> None:
    # @spec PORT-OBS-002 / PORT-OBS-010
    stage = _StageClient()
    service = PersistentStateService(stage)
    metrics = OmniStreamingMetrics(model_name="state-service-once", log_stats=False)
    service.install_metrics(metrics)
    with pytest.raises(RuntimeError, match="already installed"):
        service.install_metrics(metrics)

    module = __import__("vllm_omni.engine.persistent_state_service", fromlist=["*"])
    source = inspect.getsource(module)
    assert "prometheus" not in source.lower()
    assert "vllm_omni.metrics" not in source


def test_readiness_probe_is_not_an_admission_rejection() -> None:
    # @spec PORT-OBS-007 / PORT-STATE-015
    async def scenario() -> None:
        stage = _StageClient()
        service = PersistentStateService(stage)
        metrics = OmniStreamingMetrics(
            model_name="state-service-health", log_stats=True
        )
        await service.check_health()
        service.install_metrics(metrics)
        service.close_admission()
        prefix = (
            f'{defs.STREAMING_ADMISSION_REJECTIONS}_total{{'
            'model_name="state-service-health",reason="unavailable"}'
        )
        before = _value(prefix) or 0.0

        with pytest.raises(PersistentStateServiceUnavailable):
            await service.check_health()
        assert (_value(prefix) or 0.0) == before

        with pytest.raises(PersistentStateServiceUnavailable):
            await service.check_admission()
        assert _value(prefix) == before + 1.0
        service.shutdown()

    asyncio.run(scenario())
