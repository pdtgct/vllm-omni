# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery contract: admission closures recover autonomously.

PORT-STATE-022/023/024, from the live incident where a client storm
latched admission closed until restart. Every fail-closed edge except
engine-epoch invalidation demotes the service to un-handshaked and arms one
service-owned recovery authority. Health probes join that authority rather
than driving a second handshake. The handshake reconciles
orphaned engine bindings; horizon exhaustion is per-operation retryable
backpressure, never a service latch.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.engine.persistent_state_service import (
    PersistentStateBackpressure,
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
    "resident_state_scatter_warmup_complete": True,
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
        self.release_operation_ids: list[str] = []
        self.release_failures_remaining = 0
        self.resident = 0
        self.revision = 0
        self.bindings: list[dict[str, Any]] = []
        self.snapshot_calls = 0
        self.engine_epoch = "epoch-a"
        self.snapshot_gate: asyncio.Event | None = None

    def _snapshot(self) -> dict[str, Any]:
        snapshot = dict(_SNAPSHOT_BASE)
        snapshot["engine_epoch"] = self.engine_epoch
        snapshot["manager_revision"] = self.revision
        snapshot["resident_count"] = self.resident
        snapshot["bindings"] = [dict(b) for b in self.bindings]
        return snapshot

    async def call_utility_async(self, name: str, *args: Any) -> dict[str, Any]:
        if name == "persistent_state_snapshot":
            self.snapshot_calls += 1
            if self.snapshot_gate is not None:
                await self.snapshot_gate.wait()
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
            self.release_operation_ids.append(operation_id)
            if self.release_failures_remaining:
                self.release_failures_remaining -= 1
                raise RuntimeError("claimed lease is still running")
            self.resident = max(0, self.resident - 1)
            self.revision += 1
            self.bindings = [b for b in self.bindings if b["binding_token"] != lease["binding_token"]]
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


def _recovering_service(
    stage: _RecoveryStage,
    clock: _Clock,
    *,
    fatal_errors: list[BaseException] | None = None,
    **overrides: Any,
) -> PersistentStateService:
    """Build the Phase-5 recovery shape with explicit, non-ENV defaults."""

    required = {"admission_config", "host_fatal_callback"}
    parameters = set(inspect.signature(PersistentStateService).parameters)
    if missing := required.difference(parameters):
        pytest.fail(
            f"PORT-STATE-014 / PORT-STATE-022 missing autonomous recovery constructor seams: {sorted(missing)}",
            pytrace=False,
        )
    errors = fatal_errors if fatal_errors is not None else []
    try:
        from vllm_omni.engine.persistent_state_admission import (
            AdmissionControllerConfig,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.fail(
            "PORT-STATE-027 missing AdmissionControllerConfig",
            pytrace=False,
        )
    admission_config = AdmissionControllerConfig(
        waiter_capacity=4,
        max_inflight_reserves=2,
        dispatch_budget=1,
        aging_threshold_ns=10_000_000,
        admission_wait_timeout_s=0.25,
        retry_floor_ms=10,
        retry_jitter_ms=0,
        recovery_backoff_s=(0.001, 0.002),
        release_convergence_timeout_s=0.03,
        supported_intervals_ms=(80, 160, 320, 560, 1120),
    )
    return _service(
        stage,
        clock,
        admission_config=admission_config,
        host_fatal_callback=errors.append,
        **overrides,
    )


async def _wait_until(
    predicate: Any,
    *,
    timeout_s: float = 0.25,
) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=timeout_s)


def _compiled_admission_profile(
    *,
    derating_factor: Fraction = Fraction(1, 2),
    maximum_population: int = 2,
    admitted_geometry_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> Any:
    from vllm_omni.engine.persistent_state_capacity import (
        ServiceExecutionTier,
        ServiceProfileContext,
        ServiceRoundExecution,
        compile_provisional_service_profile,
    )

    intervals = (80, 160, 320, 560, 1120)
    executions = tuple(
        ServiceRoundExecution(
            tier_id="all",
            active_population=maximum_population,
            # At the two-session fixture, the 1120-ms bucket admits one
            # under f=1/2 but fragmentation rejects two.  The c=4/J test
            # selects its own f=1 profile and retains the fast table.
            elapsed_ns=(300_000_000 if maximum_population == 2 and interval == 1120 else 50_000_000),
            service_interval_ms=interval,
            geometry_id=geometry,
            completed_legal_parks=maximum_population,
            completed_model_rows=None,
            post_jit=True,
            continuously_loaded=True,
            dummy_run=False,
            is_profile=False,
        )
        for geometry, interval in enumerate(intervals)
        if geometry in admitted_geometry_ids
    )
    return compile_provisional_service_profile(
        executions,
        execution_tiers=(ServiceExecutionTier("all", maximum_population),),
        max_population=maximum_population,
        reference_interval_ms=1120,
        reference_geometry_id=4,
        admitted_geometry_ids=admitted_geometry_ids,
        trailing_rounds=1,
        derating_factor=derating_factor,
        context=ServiceProfileContext(
            pre_override_physical_bound=4,
            allocated_pool=maximum_population,
            count_cap=maximum_population,
            execution_claim_ceiling=maximum_population,
            service_budget_source="measured_fallback",
            service_budget_coefficients=(Fraction(1),),
            derating_factor=derating_factor,
            slot_bytes=6_314_936,
            execution_environment_key="phase6-cpu-referee",
            precision_policy="fp32",
            state_profile="test-profile",
            compiler_version="test-v1",
            mixed_composition_policy="periodic_limited_preemption_edf",
        ),
        # Recovery tests isolate controller lifecycle with synthetic regular
        # rounds; role/control compilation has its own exact contract suite.
        control_dominance_sha256="d" * 64,
    )


def test_recovery_uses_one_explicit_convergence_clock_authority() -> None:
    """@spec PORT-STATE-014: no unread shadow clock can imply authority."""
    service = _recovering_service(_RecoveryStage(), _Clock())
    assert not hasattr(service, "_failed_release_started_at"), (
        "PORT-STATE-014 retains an unread shadow convergence clock"
    )
    service.shutdown()


def test_bootstrap_profile_seal_is_one_shot_and_opens_public_authority() -> None:
    """@spec ENV-MIG-012 / PORT-STATE-027 / PORT-PERF-006."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        service = _recovering_service(stage, _Clock())
        bootstrap = getattr(service, "bootstrap_handshake", None)
        seal = getattr(service, "seal_startup_profile", None)
        if not callable(bootstrap) or not callable(seal):
            pytest.fail(
                "ENV-MIG-012 missing bootstrap/profile-seal lifecycle",
                pytrace=False,
            )
        inventory = await bootstrap()
        assert inventory["engine_epoch"] == "epoch-a"
        assert not service.ready
        profile = _compiled_admission_profile()

        seal(compiled_service_profile=profile)

        assert service.ready
        assert service.compiled_service_profile is profile
        with pytest.raises(RuntimeError, match="already.*seal|one.*shot"):
            seal(compiled_service_profile=profile)
        reserve_for_priming = getattr(service, "reserve_for_priming", None)
        if not callable(reserve_for_priming):
            pytest.fail(
                "PORT-PERF-005 missing private bootstrap reserve",
                pytrace=False,
            )
        with pytest.raises(
            PersistentStateServiceUnavailable,
            match="bootstrap|seal|priming",
        ):
            await reserve_for_priming(**_lease_kwargs(99))
        service.shutdown()

    asyncio.run(scenario())


def test_service_projection_exports_the_exact_installed_derating() -> None:
    """@spec PORT-OBS-012: the budget gauge is the installed exact f."""

    class _MetricsSink:
        def __init__(self) -> None:
            self.capacity: list[dict[str, Any]] = []

        def observe_persistent_state_capacity(
            self,
            _stage: str,
            _replica: str,
            **values: Any,
        ) -> None:
            self.capacity.append(values)

        def observe_persistent_state_slots(
            self,
            _stage: str,
            _replica: str,
            _values: dict[str, int],
        ) -> None:
            pass

    async def scenario() -> None:
        service = _recovering_service(_RecoveryStage(), _Clock())
        await service.bootstrap_handshake()
        service.seal_startup_profile(
            compiled_service_profile=_compiled_admission_profile(
                derating_factor=Fraction(1, 2),
            )
        )
        metrics = _MetricsSink()

        service.install_metrics(metrics)

        assert metrics.capacity
        assert metrics.capacity[-1]["service_budget"] == 0.5
        service.shutdown()

    asyncio.run(scenario())


def test_profile_seal_installs_one_exact_noncontiguous_interval_authority() -> None:
    """@spec PORT-STATE-027: profile, counters, and queues share one subset."""

    async def scenario() -> None:
        from vllm_omni.engine.persistent_state_admission import (
            AdmissionControllerConfig,
        )

        served_ids = (0, 2, 3, 4)
        served_intervals = (80, 320, 560, 1120)
        config = AdmissionControllerConfig(
            waiter_capacity=4,
            max_inflight_reserves=2,
            dispatch_budget=1,
            aging_threshold_ns=10_000_000,
            admission_wait_timeout_s=0.25,
            retry_floor_ms=10,
            retry_jitter_ms=0,
            recovery_backoff_s=(0.001, 0.002),
            release_convergence_timeout_s=0.03,
            supported_intervals_ms=served_intervals,
        )
        service = _service(
            _RecoveryStage(),
            _Clock(),
            admission_config=config,
        )
        await service.bootstrap_handshake()
        profile = _compiled_admission_profile(
            admitted_geometry_ids=served_ids,
        )

        service.seal_startup_profile(compiled_service_profile=profile)

        assert tuple(service._resident_interval_counts) == served_intervals
        assert tuple(service._submitted_interval_counts) == served_intervals
        assert tuple(service._failed_release_interval_counts) == served_intervals
        assert tuple(service._admission_controller.pending_counts) == served_intervals
        service.shutdown()

    asyncio.run(scenario())


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


async def _reserve_with_interval(
    service: PersistentStateService,
    n: int,
) -> Any:
    if "service_interval_ms" not in inspect.signature(service.reserve).parameters:
        pytest.fail(
            "PORT-STATE-025 missing reserve-to-release service interval",
            pytrace=False,
        )
    return await service.reserve(
        **_lease_kwargs(n),
        service_interval_ms=560,
    )


# ---- PORT-STATE-022: demote and re-verify ----------------------------------


# @spec PORT-STATE-022
def test_probe_joins_recovery_after_an_initial_handshake_failure() -> None:
    """A probe observes, but never becomes, the recovery authority."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _service(stage, clock)
        await _open_service(service)
        stage.hang_reserve = True
        stage.snapshot_error = RuntimeError("engine not answering")
        with pytest.raises(PersistentStateIndeterminate):
            await service.reserve(**_lease_kwargs(1))
        assert not service.ready
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


# @spec PORT-STATE-022 / PORT-STATE-028
def test_indeterminate_reserve_recovers_without_an_external_health_probe() -> None:
    """Entering UNHANDSHAKED arms the one service-owned recovery driver."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _recovering_service(
            stage,
            clock,
            operation_timeout_s=0.005,
            reconciliation_timeout_s=0.005,
        )
        await _open_service(service)
        initial_snapshots = stage.snapshot_calls
        stage.hang_reserve = True
        with pytest.raises(PersistentStateIndeterminate):
            await service.reserve(**_lease_kwargs(1))
        assert not service.ready

        stage.hang_reserve = False
        stage._hang_release.set()
        await _wait_until(lambda: service.ready)

        assert stage.snapshot_calls > initial_snapshots
        result = await service.reserve(**_lease_kwargs(2))
        assert result.session_key == "session-2"
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-022
def test_health_probes_join_one_inflight_recovery_authority() -> None:
    """Health joins recovery without blocking or starting another handshake."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _recovering_service(
            stage,
            clock,
            operation_timeout_s=0.005,
            reconciliation_timeout_s=0.005,
        )
        await _open_service(service)
        stage.hang_reserve = True
        with pytest.raises(PersistentStateIndeterminate):
            await service.reserve(**_lease_kwargs(1))
        stage.hang_reserve = False
        stage._hang_release.set()
        stage.snapshot_gate = asyncio.Event()
        snapshots_before = stage.snapshot_calls
        await _wait_until(lambda: stage.snapshot_calls > snapshots_before)

        async def probe() -> None:
            with pytest.raises(
                PersistentStateServiceUnavailable,
                match="admission|unhandshaked|recovery|unavailable",
            ):
                await service.check_health()

        probe_tasks = [asyncio.create_task(probe()) for _ in range(8)]
        _, pending = await asyncio.wait(probe_tasks, timeout=0.05)
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            stage.snapshot_gate.set()
            pytest.fail(
                "PORT-STATE-030 health blocked on authority recovery",
                pytrace=False,
            )
        assert stage.snapshot_calls == snapshots_before + 1
        stage.snapshot_gate.set()
        await _wait_until(lambda: service.ready)
        await service.check_health()
        assert stage.snapshot_calls == snapshots_before + 1
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-022 / PORT-PERF-006
def test_recovery_reinstalls_the_cold_start_profile_without_remeasuring() -> None:
    """Recovery reuses the exact immutable compiled authority.

    The service receives a compiled profile, not a profiler callback. That
    type-level boundary prevents a recovering or partially loaded engine from
    silently replacing the startup capacity receipt.
    """

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        profile = SimpleNamespace(receipt_sha256="a" * 64)
        parameters = set(inspect.signature(PersistentStateService).parameters)
        if "compiled_service_profile" not in parameters:
            pytest.fail(
                "PORT-STATE-022 missing immutable compiled service profile on recovery",
                pytrace=False,
            )
        service = _recovering_service(
            stage,
            clock,
            compiled_service_profile=profile,
            operation_timeout_s=0.005,
            reconciliation_timeout_s=0.005,
        )
        await _open_service(service)
        assert service.compiled_service_profile == profile

        stage.hang_reserve = True
        with pytest.raises(PersistentStateIndeterminate):
            await service.reserve(**_lease_kwargs(1))
        stage.hang_reserve = False
        stage._hang_release.set()
        await _wait_until(lambda: service.ready)

        assert service.compiled_service_profile == profile
        assert service.inventory is not None
        assert service.inventory["service_profile_receipt_sha256"] == "a" * 64
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-004 / PORT-STATE-027 / PORT-STATE-028
def test_service_waits_before_reserve_and_release_wakes_the_oldest_head() -> None:
    """The serving adapter must reach the bounded controller, not bypass it."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _recovering_service(
            stage,
            clock,
            compiled_service_profile=_compiled_admission_profile(),
        )
        await _open_service(service)
        first = await service.reserve(
            **_lease_kwargs(1),
            service_interval_ms=1120,
            connection_id="connection-1",
        )
        second_task = asyncio.create_task(
            service.reserve(
                **_lease_kwargs(2),
                service_interval_ms=1120,
                connection_id="connection-2",
            )
        )
        await _wait_until(
            lambda: service.admission_snapshot is not None and service.admission_snapshot.waiter_count == 1
        )
        await asyncio.sleep(0)
        assert stage.reserve_calls == 1
        assert not second_task.done()

        await service.release(
            operation_id="release-1",
            lease=first,
            reason="test",
        )
        second = await asyncio.wait_for(second_task, timeout=0.25)

        assert stage.reserve_calls == 2
        assert second.session_key == "session-2"
        await service.release(
            operation_id="release-2",
            lease=second,
            reason="test",
        )
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-024 / PORT-STATE-027
def test_service_wait_deadline_sheds_without_submitting_to_engine() -> None:
    """A bounded waiting-room expiry is typed shed, never core work."""

    async def scenario() -> None:
        from vllm_omni.engine.persistent_state_admission import (
            AdmissionControllerConfig,
        )

        stage = _RecoveryStage()
        service = _service(
            stage,
            _Clock(),
            monotonic=time.monotonic,
            admission_config=AdmissionControllerConfig(
                waiter_capacity=4,
                max_inflight_reserves=1,
                dispatch_budget=1,
                aging_threshold_ns=1_000_000,
                admission_wait_timeout_s=0.01,
                retry_floor_ms=10,
                retry_jitter_ms=0,
                recovery_backoff_s=(0.001,),
                release_convergence_timeout_s=0.1,
                supported_intervals_ms=(80, 160, 320, 560, 1120),
            ),
            compiled_service_profile=_compiled_admission_profile(),
        )
        await _open_service(service)
        first = await service.reserve(
            **_lease_kwargs(1),
            service_interval_ms=1120,
        )

        with pytest.raises(PersistentStateBackpressure) as info:
            await asyncio.wait_for(
                service.reserve(
                    **_lease_kwargs(2),
                    service_interval_ms=1120,
                ),
                timeout=0.1,
            )

        assert info.value.cause == "wait_deadline"
        assert stage.reserve_calls == 1
        assert service.ready
        await service.release(
            operation_id="release-1",
            lease=first,
            reason="test",
        )
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-027 / PORT-PERF-008
def test_four_concurrent_streams_dispatch_in_bounded_fifo_waves() -> None:
    """c=4 proves J=2 while every hard authority permits four."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        stage.hang_reserve = True
        original_snapshot = stage._snapshot

        def four_slot_snapshot() -> dict[str, Any]:
            snapshot = original_snapshot()
            snapshot.update(
                physical_capacity=4,
                configured_limit=4,
                effective_capacity=4,
            )
            return snapshot

        stage._snapshot = four_slot_snapshot  # type: ignore[method-assign]
        service = _recovering_service(
            stage,
            _Clock(),
            compiled_service_profile=_compiled_admission_profile(
                derating_factor=Fraction(1),
                maximum_population=4,
            ),
        )
        await _open_service(service)
        tasks = [
            asyncio.create_task(
                service.reserve(
                    **_lease_kwargs(index),
                    service_interval_ms=1120,
                    connection_id=f"connection-{index}",
                )
            )
            for index in range(4)
        ]
        await _wait_until(
            lambda: service.admission_snapshot is not None and service.admission_snapshot.submitted_count == 2
        )
        await asyncio.sleep(0.01)
        assert stage.reserve_calls == 1
        assert service.admission_snapshot is not None
        assert service.admission_snapshot.submitted_count == 2
        assert not any(task.done() for task in tasks)

        stage.hang_reserve = False
        stage._hang_release.set()
        await _wait_until(lambda: tasks[0].done() and tasks[1].done())
        first_wave = await asyncio.gather(*tasks[:2])
        second_wave = await asyncio.wait_for(
            asyncio.gather(*tasks[2:]),
            timeout=0.25,
        )

        assert stage.reserve_calls == 4
        assert [lease.session_key for lease in second_wave] == [
            "session-2",
            "session-3",
        ]
        await asyncio.gather(
            *(
                service.release(
                    operation_id=f"release-final-{index}",
                    lease=lease,
                    reason="test",
                )
                for index, lease in enumerate((*first_wave, *second_wave))
            )
        )
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
        released_tokens = [lease["binding_token"] for _, lease in stage.release_calls]
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
        stage.reserve_error = RuntimeError("persistent_state_horizon_exhausted: retry after tombstone expiry")
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


# @spec PORT-STATE-013 / PORT-STATE-024
def test_full_reserve_bridge_is_retryable_shed() -> None:
    """@spec PORT-STATE-024: the J-bounded bridge is not a fault."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _service(
            stage,
            clock,
            reserve_queue_capacity=1,
            operation_timeout_s=5.0,
            reconciliation_timeout_s=5.0,
        )
        await _open_service(service)
        stage.hang_reserve = True
        first = asyncio.create_task(service.reserve(**_lease_kwargs(1)))
        while stage.reserve_calls < 1:
            await asyncio.sleep(0)
        second = asyncio.create_task(service.reserve(**_lease_kwargs(2)))
        while service.reserve_queue.qsize() < 1:
            await asyncio.sleep(0)

        with pytest.raises(Exception) as info:
            await service.reserve(**_lease_kwargs(3))

        assert getattr(info.value, "retryable", False), "PORT-STATE-024 requires queue-full to be typed shed"
        assert getattr(info.value, "cause", None) == "bridge_full"
        assert service.ready
        assert stage.reserve_calls == 1

        stage.hang_reserve = False
        stage._hang_release.set()
        await asyncio.gather(first, second)
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-014 / PORT-STATE-022 / PORT-STATE-023
def test_failed_release_stays_charged_and_exact_retries_after_terminality() -> None:
    """A failed release cannot become an orphan or lose its operation id."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        service = _service(stage, clock)
        await _open_service(service)
        lease = await _reserve_with_interval(service, 1)
        stage.bindings = [
            {
                "binding_token": lease.binding_token,
                "session_key": lease.session_key,
                "generation": lease.generation,
                "schema_id": lease.schema_id,
                "profile_id": lease.profile_id,
                "engine_epoch": lease.engine_epoch,
                "claim_expires_at": clock.now - 1.0,
                "terminal": False,
            }
        ]
        stage.release_failures_remaining = 1

        with pytest.raises(PersistentStateServiceUnavailable):
            await service.release(
                operation_id="release-1",
                lease=lease,
                reason="client_disconnect",
            )

        assert not service.ready
        assert service.inventory is not None
        assert service.inventory["resident_count"] == 1
        assert service.inventory["execution_claims"] == 1
        assert service.inventory["charged_demand"] > 0
        assert stage.release_operation_ids == ["release-1"]

        stage.bindings[0]["terminal"] = True
        await service.check_health()

        assert service.ready
        assert stage.release_operation_ids == ["release-1", "release-1"]
        assert service.inventory is not None
        assert service.inventory["resident_count"] == 0
        assert service.inventory["execution_claims"] == 0
        assert service.inventory["charged_demand"] == 0
        service.shutdown()

    asyncio.run(scenario())


# @spec PORT-STATE-014 / PORT-STATE-022
def test_nonconverging_release_requests_host_fatal_exit_exactly_once() -> None:
    """Repeated demotion keeps one clock and cannot postpone fatal exit."""

    async def scenario() -> None:
        stage = _RecoveryStage()
        clock = _Clock()
        fatal_errors: list[BaseException] = []
        service = _recovering_service(
            stage,
            clock,
            fatal_errors=fatal_errors,
            operation_timeout_s=0.005,
            reconciliation_timeout_s=0.005,
        )
        await _open_service(service)
        lease = await _reserve_with_interval(service, 1)
        stage.bindings = [
            {
                "binding_token": lease.binding_token,
                "session_key": lease.session_key,
                "generation": lease.generation,
                "schema_id": lease.schema_id,
                "profile_id": lease.profile_id,
                "engine_epoch": lease.engine_epoch,
                "claim_expires_at": clock.now - 1.0,
                "terminal": False,
            }
        ]
        stage.release_failures_remaining = 1_000
        stage.snapshot_gate = asyncio.Event()
        snapshots_before = stage.snapshot_calls

        with pytest.raises(PersistentStateServiceUnavailable):
            await service.release(
                operation_id="release-never-converges",
                lease=lease,
                reason="client_disconnect",
            )

        recovery = getattr(service, "_recovery_task", None)
        assert recovery is not None and not recovery.done()
        await _wait_until(lambda: stage.snapshot_calls > snapshots_before)
        service._demote()
        assert service._recovery_task is recovery, (
            "PORT-STATE-022 replaced the live recovery driver and reset PORT-STATE-014's convergence clock"
        )

        stage.snapshot_gate.set()
        await _wait_until(lambda: len(fatal_errors) == 1)
        await asyncio.sleep(0.05)
        assert len(fatal_errors) == 1
        assert "release" in str(fatal_errors[0]).lower()
        assert not service.ready
        with pytest.raises(PersistentStateServiceUnavailable):
            await service.check_health()
        assert len(fatal_errors) == 1
        service.shutdown()

    asyncio.run(scenario())
