# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sealed control-charge receipt publication and startup conformance."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from prometheus_client import REGISTRY

from tests.engine.test_persistent_state_recovery import (
    _Clock,
    _compiled_admission_profile,
    _recovering_service,
    _RecoveryStage,
    _wait_until,
)
from vllm_omni.engine.persistent_state_startup import derive_hard_cap_authority
from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.streaming import OmniStreamingMetrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_CHARGE = "vllm_omni:persistent_state_control_charge"
_ADVISORY = "vllm_omni:persistent_state_profile_advisory"
_STATUSES = ("proven_zero", "unproven", "explicit_table")
_CASES = (
    ("profile", "proven_zero"),
    ("profile", "explicit_table"),
    ("hard_cap", "proven_zero"),
    ("hard_cap", "unproven"),
    ("hard_cap", "explicit_table"),
)


def _samples(name: str, model: str) -> list[Any]:
    return [
        sample
        for family in REGISTRY.collect()
        for sample in family.samples
        if sample.name == name and sample.labels.get("model_name") == model
    ]


def _assert_charge(model: str, status: str) -> None:
    samples = _samples(_CHARGE, model)
    assert len(samples) == 3
    assert {sample.labels["status"]: sample.value for sample in samples} == {
        state: float(state == status) for state in _STATUSES
    }
    assert all(
        sample.labels == {"model_name": model, "stage": "0", "replica": "0", "status": sample.labels["status"]}
        for sample in samples
    )


class _RecordingMetrics:
    def __init__(self, model: str, *, log_stats: bool = True) -> None:
        self.metrics = OmniStreamingMetrics(model_name=model, log_stats=log_stats)
        self.charge_calls: list[tuple[str, str, str]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.metrics, name)

    def observe_persistent_state_control_charge(self, stage: str, replica: str, status: str) -> None:
        self.charge_calls.append((stage, replica, status))
        self.metrics.observe_persistent_state_control_charge(stage, replica, status)


async def _bootstrap(policy: str, status: str) -> tuple[Any, Any, Any, Any]:
    service = _recovering_service(_RecoveryStage(), _Clock())
    inventory = await service.bootstrap_handshake()
    service.configure_bootstrap_intervals((1120,))
    config = replace(service._admission_config, admission_policy=policy, supported_intervals_ms=(1120,))
    profile = _compiled_admission_profile(
        admitted_geometry_ids=(4,),
        admission_policy=policy,
        canary_excess_ns=1 if status == "unproven" else 0,
    )
    if status == "explicit_table":
        # A precompiled envelope is input to this lifecycle test; this fixture
        # establishes no qualification claim for its synthetic table.
        profile = replace(
            profile,
            receipt=replace(profile.receipt, control_charge_status=status, control_dominance_evidence=None),
            control_upper_ns_by_window_and_population={window: (0, 0) for window in profile.window_ns},
        )
    authority = derive_hard_cap_authority(
        served_intervals_ms=(1120,),
        model_profile_id=inventory["profile_id"],
        execution_environment_key=inventory["execution_environment_key"],
        precision_policy=inventory["precision_policy"],
        schema_id=inventory["schema_id"],
        slot_bytes=inventory["slot_bytes"],
        stage=inventory["stage"],
        replica=inventory["replica"],
        physical_capacity=inventory["physical_capacity"],
        configured_limit=inventory["configured_limit"],
        effective_capacity=inventory["effective_capacity"],
        max_num_seqs=inventory["execution_claim_ceiling"],
        safety_reserve=inventory["safety_reserve"],
        controller_identity=repr(config),
        service_profile_identity=profile.receipt.receipt_sha256,
    )
    return service, config, profile, authority


def _seal(service: Any, config: Any, profile: Any, authority: Any) -> None:
    if config.admission_policy == "hard_cap":
        service.seal_startup_authority(authority=authority, admission_config=config, compiled_service_profile=profile)
    else:
        service.seal_startup_profile(admission_config=config, compiled_service_profile=profile)


@pytest.mark.parametrize("policy,status", _CASES)
@pytest.mark.parametrize("install_first", (False, True))
def test_control_charge_publishes_once_when_seal_and_observer_are_ready(
    policy: str, status: str, install_first: bool
) -> None:
    """@spec PORT-OBS-013, PORT-OBS-014: both startup orders publish the sealed receipt."""

    async def scenario() -> None:
        service, config, profile, authority = await _bootstrap(policy, status)
        model = f"control-charge-{policy}-{status}-{install_first}"
        metrics = _RecordingMetrics(model)
        try:
            if install_first:
                service.install_metrics(metrics)
            service._sync_service_projection()
            assert metrics.charge_calls == []
            assert _samples(_CHARGE, model) == []
            assert _samples(_ADVISORY, model) == []
            _seal(service, config, profile, authority)
            if not install_first:
                assert _samples(_CHARGE, model) == []
                service.install_metrics(metrics)
            assert metrics.charge_calls == [("0", "0", status)]
            _assert_charge(model, status)
            advisory = _samples(_ADVISORY, model)
            assert len(advisory) == 1
            assert advisory[0].value == float(policy == "hard_cap")
            assert advisory[0].labels == {"model_name": model, "stage": "0", "replica": "0"}
        finally:
            service.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("policy,status", _CASES)
def test_control_charge_receipt_is_not_rewritten_by_reserve_release_or_projection(policy: str, status: str) -> None:
    """@spec PORT-OBS-013: dynamic capacity observations cannot write receipt state."""

    async def scenario() -> None:
        service, config, profile, authority = await _bootstrap(policy, status)
        model = f"control-charge-dynamic-{policy}-{status}"
        metrics = _RecordingMetrics(model)
        try:
            _seal(service, config, profile, authority)
            service.install_metrics(metrics)
            assert metrics.charge_calls == [("0", "0", status)]
            # Write a decoy directly through the sink, bypassing the recording
            # shim. Any illicit direct gauge write by capacity projection will
            # overwrite this value even if it avoids the service observer call.
            decoy = "proven_zero" if status != "proven_zero" else "unproven"
            metrics.metrics.observe_persistent_state_control_charge("0", "0", decoy)
            _assert_charge(model, decoy)
            lease = await service.reserve(
                operation_id="reserve-metric",
                session_key="metric",
                schema_id="schema-a",
                profile_id="profile-a",
                service_interval_ms=1120,
            )
            _assert_charge(model, decoy)
            service._sync_service_projection()
            _assert_charge(model, decoy)
            await service.release(operation_id="release-metric", lease=lease, reason="test")
            _assert_charge(model, decoy)
            service._sync_service_projection()
            assert metrics.charge_calls == [("0", "0", status)]
            _assert_charge(model, decoy)
        finally:
            service.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("policy,status", _CASES)
@pytest.mark.parametrize("with_controller", (False, True))
def test_control_charge_recovery_republishes_receipt_even_without_admission(
    policy: str,
    status: str,
    with_controller: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-OBS-013, PORT-PERF-006: handshake reinstall is independent of admission."""

    async def scenario() -> None:
        service, config, profile, authority = await _bootstrap(policy, status)
        model = f"control-charge-recovery-{policy}-{status}-{with_controller}"
        metrics = _RecordingMetrics(model)
        try:
            _seal(service, config, profile, authority)
            service.install_metrics(metrics)
            if not with_controller:
                # Exercise the service-only handshake branch while retaining
                # the configuration required to schedule autonomous recovery.
                service._admission_controller = None
                monkeypatch.setattr(service, "_install_admission_controller", lambda: None)
            # A worker snapshot is not the authority for this status.
            service._stage_client.snapshot_overrides["service_profile_control_charge_status"] = (
                "proven_zero" if status != "proven_zero" else "unproven"
            )
            decoy = "proven_zero" if status != "proven_zero" else "unproven"
            metrics.metrics.observe_persistent_state_control_charge("0", "0", decoy)
            _assert_charge(model, decoy)
            metrics.charge_calls.clear()
            service._demote()
            await _wait_until(lambda: service.ready)
            assert metrics.charge_calls == [("0", "0", status)]
            assert service.compiled_service_profile is profile
            if not with_controller:
                assert service._admission_controller is None
            _assert_charge(model, status)
        finally:
            service.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("policy,status", _CASES)
def test_sealed_receipt_exports_nothing_with_statistics_disabled(policy: str, status: str) -> None:
    """@spec PORT-OBS-013, PORT-OBS-014: disabled statistics produce no receipt series."""

    async def scenario() -> None:
        service, config, profile, authority = await _bootstrap(policy, status)
        model = f"control-charge-disabled-{policy}-{status}"
        try:
            service.install_metrics(OmniStreamingMetrics(model_name=model, log_stats=False))
            _seal(service, config, profile, authority)
            service._demote()
            await _wait_until(lambda: service.ready)
            assert _samples(_CHARGE, model) == []
            assert _samples(_ADVISORY, model) == []
        finally:
            service.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "invalid,identity_present", (("missing_receipt", True), ("missing_receipt", False), ("unknown", True))
)
def test_hard_cap_rejects_invalid_receipt_before_seal_or_publication(invalid: str, identity_present: bool) -> None:
    """@spec PORT-PERF-006, PORT-OBS-013, PORT-OBS-014: invalid authority publishes nothing."""

    async def scenario() -> None:
        service, config, profile, authority = await _bootstrap("hard_cap", "proven_zero")
        model = f"control-charge-invalid-{invalid}-{identity_present}"
        metrics = _RecordingMetrics(model)
        try:
            service.install_metrics(metrics)
            if invalid == "missing_receipt":
                profile = None
                if not identity_present:
                    authority = replace(authority, service_profile_identity=None)
            else:
                profile = replace(profile, receipt=replace(profile.receipt, control_charge_status="unknown"))
            with pytest.raises(ValueError):
                _seal(service, config, profile, authority)
            assert not service.ready
            assert not service._startup_profile_sealed
            assert service.compiled_service_profile is None
            assert metrics.charge_calls == []
            assert _samples(_CHARGE, model) == []
            assert _samples(_ADVISORY, model) == []
        finally:
            service.shutdown()

    asyncio.run(scenario())


def test_control_charge_observer_sets_all_three_states_in_one_call() -> None:
    """@spec PORT-OBS-013: every observation resets the complete bounded state vector."""
    model = "control-charge-observer"
    metrics = OmniStreamingMetrics(model_name=model, log_stats=True)
    for status in _STATUSES:
        metrics.observe_persistent_state_control_charge("0", "0", status)
        _assert_charge(model, status)


@pytest.mark.parametrize("status", _STATUSES)
def test_control_charge_observer_is_inert_with_statistics_disabled(status: str) -> None:
    """@spec PORT-OBS-013: the observer itself must respect disabled statistics."""
    model = f"control-charge-disabled-observer-{status}"
    metrics = OmniStreamingMetrics(model_name=model, log_stats=False)
    metrics.observe_persistent_state_control_charge("0", "0", status)
    assert _samples(_CHARGE, model) == []


def test_control_charge_definitions_publish_the_bounded_metric_contract() -> None:
    """@spec PORT-OBS-013: shared definitions expose the public name and status domain."""
    assert defs.PERSISTENT_STATE_CONTROL_CHARGE == _CHARGE
    assert defs.PERSISTENT_STATE_CONTROL_CHARGE_STATUSES == _STATUSES
