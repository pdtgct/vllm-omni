# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for selected persistent-state startup composition."""

from __future__ import annotations

import asyncio
import importlib
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_STARTUP_MODULE = "vllm_omni.engine.persistent_state_startup"


def _startup_inventory() -> dict[str, Any]:
    return {
        "configured_limit": 4,
        "effective_capacity": 4,
        "engine_epoch": "epoch-a",
        "execution_claim_ceiling": 8,
        "execution_environment_key": "env-a",
        "physical_capacity": 5,
        "precision_policy": "torch.float32",
        "profile_id": "profile-a",
        "resident_state_scatter_warmup_complete": True,
        "schema_id": "schema-a",
        "slot_bytes": 6_314_936,
    }


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _startup_module() -> ModuleType:
    try:
        return importlib.import_module(_STARTUP_MODULE)
    except ModuleNotFoundError:
        _fail(f"PORT-INT-013 missing {_STARTUP_MODULE}")


def _startup_symbol(name: str) -> Any:
    try:
        return getattr(_startup_module(), name)
    except AttributeError:
        _fail(f"PORT-INT-013 missing {_STARTUP_MODULE}.{name}")


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        admission_waiter_capacity=8,
        admission_max_inflight_reserves=2,
        admission_dispatch_budget=1,
        admission_aging_threshold_s=0.05,
        admission_wait_timeout_s=0.1,
        admission_retry_floor_ms=10,
        admission_retry_jitter_ms=0,
        recovery_backoff_s=(0.01, 0.02),
        release_convergence_timeout_s=60.01,
        service_profile_trailing_rounds=3,
        startup_priming_timeout_s=0.5,
        service_profile_derating_factor=0.5,
        reserve_queue_capacity=8,
        priming_budget_descriptor=SimpleNamespace(
            sha256="budget-a",
            bootstrap_operation_budget=8,
        ),
        priming_budget_sha256="budget-a",
        bootstrap_operation_budget=8,
        runtime_tombstone_allowance=32,
        max_tombstones=40,
    )


class _Provider:
    def __init__(self, events: list[str], runtime: Any | None = None) -> None:
        self.events = events
        self.runtime = runtime

    def build_priming_plan(
        self,
        *,
        runtime_config: Any,
        inventory: Any,
    ) -> Any:
        if self.runtime is not None:
            assert runtime_config is self.runtime
        assert inventory["resident_state_scatter_warmup_complete"] is True
        self.events.append("plan")
        return SimpleNamespace(
            rounds=(SimpleNamespace(round_id="round-0"),),
            compile_kwargs={"trailing_rounds": 3},
        )

    async def execute_priming_round(
        self,
        *,
        engine_client: Any,
        round_spec: Any,
        leases: tuple[Any, ...],
    ) -> Any:
        del engine_client, round_spec, leases
        self.events.append("execute")
        return SimpleNamespace()


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_preparation_orders_bootstrap_priming_seal_and_installability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec ENV-MIG-012 / PORT-PERF-005/006 / PORT-INT-013."""
    module = _startup_module()
    prepare = _startup_symbol("prepare_persistent_state_service")
    events: list[str] = []
    runtime = _runtime()
    provider = _Provider(events, runtime)
    profile = object()

    class _Service:
        def __init__(self, stage_client: Any, **kwargs: Any) -> None:
            assert stage_client == "stage"
            assert kwargs["runtime_config"] is runtime
            events.append("construct")
            self.ready = False

        async def bootstrap_handshake(self) -> dict[str, Any]:
            events.append("handshake")
            return _startup_inventory()

        def seal_startup_profile(
            self,
            *,
            admission_config: Any,
            compiled_service_profile: Any,
        ) -> None:
            assert admission_config == "controller-config"
            assert compiled_service_profile is profile
            events.append("seal")
            self.ready = True

        def shutdown(self) -> None:
            events.append("shutdown")

    async def _prime(**kwargs: Any) -> Any:
        assert kwargs["service"].ready is False
        events.append("prime")
        return SimpleNamespace(round_id="round-0")

    monkeypatch.setattr(module, "PersistentStateService", _Service)
    monkeypatch.setattr(
        module,
        "derive_admission_controller_config",
        lambda value: events.append("validate") or "controller-config",
    )
    monkeypatch.setattr(module, "run_service_priming_round", _prime)
    monkeypatch.setattr(
        module,
        "compile_provisional_service_profile",
        lambda observations, **kwargs: (
            events.append("compile") or profile
        ),
    )

    service = await prepare(
        engine_client="engine",
        stage_client="stage",
        runtime_config=runtime,
        startup_provider=provider,
        host_fatal_callback=lambda error: None,
    )

    assert service.ready
    assert events == [
        "validate",
        "construct",
        "handshake",
        "plan",
        "prime",
        "compile",
        "seal",
    ]


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_preparation_rolls_back_when_warmup_attestation_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec ENV-MIG-012: API startup trusts an explicit worker attestation."""
    module = _startup_module()
    prepare = _startup_symbol("prepare_persistent_state_service")
    events: list[str] = []

    class _Service:
        def __init__(self, stage_client: Any, **kwargs: Any) -> None:
            del stage_client, kwargs

        async def bootstrap_handshake(self) -> dict[str, Any]:
            return {"resident_state_scatter_warmup_complete": False}

        def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr(module, "PersistentStateService", _Service)
    monkeypatch.setattr(
        module,
        "derive_admission_controller_config",
        lambda value: "controller-config",
    )

    with pytest.raises(RuntimeError, match="scatter.*warmup|warmup.*attest"):
        await prepare(
            engine_client="engine",
            stage_client="stage",
            runtime_config=_runtime(),
            startup_provider=_Provider(events),
            host_fatal_callback=lambda error: None,
        )

    assert events == ["shutdown"]


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_preparation_rejects_incomplete_startup_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-PERF-005/006: profile authority is fully fingerprinted."""
    module = _startup_module()
    prepare = _startup_symbol("prepare_persistent_state_service")
    events: list[str] = []

    class _Service:
        def __init__(self, stage_client: Any, **kwargs: Any) -> None:
            del stage_client, kwargs

        async def bootstrap_handshake(self) -> dict[str, Any]:
            inventory = _startup_inventory()
            del inventory["execution_environment_key"]
            return inventory

        def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr(module, "PersistentStateService", _Service)
    monkeypatch.setattr(
        module,
        "derive_admission_controller_config",
        lambda value: "controller-config",
    )

    with pytest.raises(RuntimeError, match="execution_environment_key"):
        await prepare(
            engine_client="engine",
            stage_client="stage",
            runtime_config=_runtime(),
            startup_provider=_Provider(events),
            host_fatal_callback=lambda error: None,
        )

    assert events == ["shutdown"]


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_preparation_timeout_is_primary_and_never_seals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec ENV-MIG-012 / PORT-PERF-005: the complete plan is bounded."""
    module = _startup_module()
    prepare = _startup_symbol("prepare_persistent_state_service")
    events: list[str] = []
    runtime = _runtime()
    runtime.startup_priming_timeout_s = 0.01

    class _Service:
        def __init__(self, stage_client: Any, **kwargs: Any) -> None:
            del stage_client, kwargs

        async def bootstrap_handshake(self) -> dict[str, Any]:
            return _startup_inventory()

        def seal_startup_profile(self, **kwargs: Any) -> None:
            del kwargs
            events.append("seal")

        def shutdown(self) -> None:
            events.append("shutdown")

    async def _blocked_prime(**kwargs: Any) -> Any:
        del kwargs
        await asyncio.sleep(0.05)
        pytest.fail(
            "PORT-PERF-005 startup priming exceeded its process bound",
            pytrace=False,
        )

    monkeypatch.setattr(module, "PersistentStateService", _Service)
    monkeypatch.setattr(
        module,
        "derive_admission_controller_config",
        lambda value: "controller-config",
    )
    monkeypatch.setattr(module, "run_service_priming_round", _blocked_prime)

    with pytest.raises(asyncio.TimeoutError):
        await prepare(
            engine_client="engine",
            stage_client="stage",
            runtime_config=runtime,
            startup_provider=_Provider(events),
            host_fatal_callback=lambda error: None,
        )

    assert events == ["plan", "shutdown"]


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_api_install_delegates_to_one_typed_preparation_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-INT-013: generic API code owns no partial construction."""
    from vllm.model_executor import model_loader

    from vllm_omni.engine import persistent_state_config
    from vllm_omni.entrypoints import async_omni as async_omni_module
    from vllm_omni.entrypoints.openai import api_server

    provider = object()

    class _PersistentModel:
        supports_persistent_state = True
        persistent_state_startup_provider = provider

    prepared = object()
    calls: list[dict[str, Any]] = []

    async def _prepare(**kwargs: Any) -> object:
        calls.append(kwargs)
        return prepared

    if not hasattr(api_server, "prepare_persistent_state_service"):
        _fail(
            "PORT-INT-013 API server does not expose the typed PORT "
            "preparation join"
        )
    monkeypatch.setattr(model_loader, "get_model_cls", lambda config: _PersistentModel)
    monkeypatch.setattr(api_server, "prepare_persistent_state_service", _prepare)
    engine = object.__new__(async_omni_module.AsyncOmni)
    engine.engine = SimpleNamespace(stage_clients=["stage"])
    engine._persistent_state_service = None
    runtime = object()
    monkeypatch.setattr(
        persistent_state_config.PersistentStateRuntimeConfig,
        "from_vllm_config",
        classmethod(
            lambda cls, config, *, startup_provider: (
                runtime if startup_provider is provider else None
            )
        ),
    )

    def fatal(error: BaseException) -> None:
        del error

    service = await api_server._install_persistent_state_service(
        engine,
        SimpleNamespace(model_config=object()),
        host_fatal_callback=fatal,
    )

    assert service is prepared
    assert engine.get_persistent_state_service() is prepared
    assert calls == [
        {
            "engine_client": engine,
            "stage_client": "stage",
            "runtime_config": runtime,
            "startup_provider": provider,
            "host_fatal_callback": fatal,
        }
    ]


def test_host_fatal_records_engine_state_before_termination_supervision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-STATE-014 / PORT-INT-013."""
    from vllm_omni.entrypoints.openai import api_server

    if not hasattr(api_server, "_persistent_state_host_fatal_callback"):
        _fail("PORT-STATE-014 missing API host-fatal supervision adapter")
    events: list[str] = []
    engine = SimpleNamespace(
        report_persistent_state_fatal=lambda error: events.append("report")
    )
    server = object()
    state = SimpleNamespace(server=server)
    monkeypatch.setattr(
        api_server,
        "terminate_if_errored",
        lambda *, server, engine: events.append("terminate"),
    )

    callback = api_server._persistent_state_host_fatal_callback(
        state=state,
        engine_client=engine,
    )
    callback(RuntimeError("release recovery exhausted"))

    assert events == ["report", "terminate"]


def test_async_omni_fatal_state_is_visible_to_launcher_supervision() -> None:
    """@spec PORT-STATE-014: the callback changes real engine properties."""
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    error = RuntimeError("release recovery exhausted")
    engine = object.__new__(AsyncOmni)
    engine._persistent_state_fatal = None
    engine.engine = SimpleNamespace(is_alive=lambda: True)
    engine.final_output_task = SimpleNamespace(done=lambda: False)

    engine.report_persistent_state_fatal(error)

    assert engine.errored
    assert not engine.is_running
    assert engine.dead_error is error


def test_nemotron_provider_covers_single_and_bulk_eager_shapes() -> None:
    """@spec PORT-PERF-005/006: the first consumer supplies a finite plan."""
    from vllm_omni.model_executor.models.nemotron_asr.startup import (
        NEMOTRON_PERSISTENT_STATE_STARTUP,
    )

    runtime = SimpleNamespace(
        service_profile_trailing_rounds=3,
        service_profile_derating_factor=0.5,
        priming_budget_descriptor=(
            NEMOTRON_PERSISTENT_STATE_STARTUP.build_priming_budget_descriptor(
                configured_population_ceiling=8,
                trailing_rounds=3,
            )
        ),
    )
    inventory = {
        "physical_capacity": 5,
        "effective_capacity": 4,
        "configured_limit": 4,
        "execution_claim_ceiling": 8,
        "slot_bytes": 6_314_936,
        "execution_environment_key": "env-a",
        "precision_policy": "torch.float32",
        "schema_id": "schema-a",
        "profile_id": "profile-a",
    }

    plan = NEMOTRON_PERSISTENT_STATE_STARTUP.build_priming_plan(
        runtime_config=runtime,
        inventory=inventory,
    )

    tiers = plan.compile_kwargs["execution_tiers"]
    assert [(tier.tier_id, tier.max_active_population) for tier in tiers] == [
        ("single", 1),
        ("eager-bulk", 4),
    ]
    assert len(plan.rounds) == 5 * 2 * 4
    assert plan.actual_operation_count == 200
    assert len(plan.actual_plan_sha256) == 64
    assert plan.compile_kwargs["startup_priming_receipt"] == {
        "budget_sha256": runtime.priming_budget_descriptor.sha256,
        "bootstrap_operation_budget": 360,
        "actual_plan_sha256": plan.actual_plan_sha256,
        "actual_operation_count": 200,
    }
    assert sum(not round_spec.post_jit for round_spec in plan.rounds) == 10
    assert {
        (round_spec.geometry_id, round_spec.tier_id)
        for round_spec in plan.rounds
    } == {
        (geometry_id, tier_id)
        for geometry_id in range(5)
        for tier_id in ("single", "eager-bulk")
    }


def test_nemotron_budget_uses_configured_not_inventory_population() -> None:
    """@spec PORT-INT-013 / ENV-MIG-012: pre-control-plane budget covers P."""
    from vllm_omni.model_executor.models.nemotron_asr.startup import (
        NEMOTRON_PERSISTENT_STATE_STARTUP,
    )

    budget = NEMOTRON_PERSISTENT_STATE_STARTUP.build_priming_budget_descriptor(
        configured_population_ceiling=8,
        trailing_rounds=3,
    )

    assert budget.configured_population_ceiling == 8
    assert budget.bootstrap_operation_budget == 360
    assert {
        (cell.geometry_id, cell.tier_id, cell.max_active_population)
        for cell in budget.cells
    } == {
        (geometry_id, tier_id, population)
        for geometry_id in range(5)
        for tier_id, population in (("single", 1), ("eager-bulk", 8))
    }


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_nemotron_provider_executes_bound_requests_to_legal_park(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-PERF-005: model priming uses ordinary bound sessions."""
    from vllm_omni.entrypoints import nemotron_session as binding_module
    from vllm_omni.model_executor.models.nemotron_asr import (
        session as session_module,
    )
    from vllm_omni.model_executor.models.nemotron_asr.startup import (
        NEMOTRON_PERSISTENT_STATE_STARTUP,
    )

    events: list[tuple[str, str]] = []

    class _Bound:
        def __init__(self, **kwargs: Any) -> None:
            state_lease = kwargs["state_lease"]
            assert kwargs["request_id"] == state_lease.session_key
            events.append(("construct", state_lease.session_key))
            self.request_id = state_lease.session_key

        async def feed(self, samples: Any) -> list[str]:
            assert samples.shape == (1_280,)
            events.append(("feed", self.request_id))
            return [""]

        async def abort(self) -> None:
            events.append(("abort", self.request_id))

    monkeypatch.setattr(binding_module, "NemotronSessionLease", _Bound)
    monkeypatch.setattr(
        session_module.NemotronRealtimeSession,
        "from_model_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    engine = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(prompt_dictionary={"en-US": 0})
        )
    )
    leases = tuple(
        SimpleNamespace(
            session_key=f"session-{index}",
            engine_epoch="epoch-a",
            generation=index + 1,
        )
        for index in range(2)
    )
    round_spec = SimpleNamespace(service_interval_ms=80)

    result = await NEMOTRON_PERSISTENT_STATE_STARTUP.execute_priming_round(
        engine_client=engine,
        round_spec=round_spec,
        leases=leases,
    )

    assert result.completed_legal_parks == 2
    assert set(events) == {
        ("construct", "session-0"),
        ("construct", "session-1"),
        ("feed", "session-0"),
        ("feed", "session-1"),
        ("abort", "session-0"),
        ("abort", "session-1"),
    }
