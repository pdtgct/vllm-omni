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
        admission_policy="profile",
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
        startup_priming_round_timeout_s=0.25,
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


@pytest.mark.parametrize("policy", ("profile", "hard_cap"))
def test_controller_config_carries_selected_admission_policy(
    policy: str,
) -> None:
    """@spec PORT-STATE-027 / ENV-MIG-011: startup carries policy once."""
    runtime = _runtime()
    runtime.admission_policy = policy

    config = _startup_symbol("derive_admission_controller_config")(
        runtime,
        supported_intervals_ms=(80, 320, 560, 1_120),
    )

    assert config.admission_policy == policy


class _Provider:
    def __init__(
        self,
        events: list[str],
        runtime: Any | None = None,
        expected_model_config: Any | None = None,
    ) -> None:
        self.events = events
        self.runtime = runtime
        self.expected_model_config = expected_model_config

    def build_priming_plan(
        self,
        *,
        runtime_config: Any,
        inventory: Any,
        model_config: Any | None = None,
    ) -> Any:
        if self.runtime is not None:
            assert runtime_config is self.runtime
        if self.expected_model_config is not None:
            assert model_config is self.expected_model_config
        assert inventory["resident_state_scatter_warmup_complete"] is True
        self.events.append("plan")
        return SimpleNamespace(
            rounds=(SimpleNamespace(round_id="round-0"),),
            compile_kwargs={"trailing_rounds": 3},
            served_intervals_ms=(80, 320, 560, 1120),
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
    model_config = object()
    engine = SimpleNamespace(model_config=model_config)
    provider = _Provider(events, runtime, model_config)
    profile = SimpleNamespace(compiled_demand=SimpleNamespace(intervals_ms=(80, 320, 560, 1120)))

    class _Service:
        def __init__(self, stage_client: Any, **kwargs: Any) -> None:
            assert stage_client == "stage"
            assert kwargs["runtime_config"] is runtime
            events.append("construct")
            self.ready = False

        async def bootstrap_handshake(self) -> dict[str, Any]:
            events.append("handshake")
            return _startup_inventory()

        def configure_bootstrap_intervals(
            self,
            intervals_ms: tuple[int, ...],
        ) -> None:
            assert intervals_ms == (80, 320, 560, 1120)
            events.append("configure")

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

    def _derive_config(
        value: Any,
        *,
        supported_intervals_ms: tuple[int, ...],
    ) -> str:
        assert value is runtime
        assert supported_intervals_ms == (80, 320, 560, 1120)
        events.append("validate")
        return "controller-config"

    monkeypatch.setattr(
        module,
        "derive_admission_controller_config",
        _derive_config,
    )
    monkeypatch.setattr(module, "run_service_priming_round", _prime)

    def _compile(observations: Any, **kwargs: Any) -> Any:
        del observations
        assert kwargs["admission_policy"] == runtime.admission_policy
        events.append("compile")
        return profile

    monkeypatch.setattr(
        module,
        "compile_provisional_service_profile",
        _compile,
    )

    service = await prepare(
        engine_client=engine,
        stage_client="stage",
        runtime_config=runtime,
        startup_provider=provider,
        host_fatal_callback=lambda error: None,
    )

    assert service.ready
    assert events == [
        "construct",
        "handshake",
        "plan",
        "configure",
        "prime",
        "compile",
        "validate",
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

        def configure_bootstrap_intervals(
            self,
            intervals_ms: tuple[int, ...],
        ) -> None:
            assert intervals_ms == (80, 320, 560, 1120)

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
            engine_client=SimpleNamespace(model_config=object()),
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
        _fail("PORT-INT-013 API server does not expose the typed PORT preparation join")
    monkeypatch.setattr(model_loader, "get_model_cls", lambda config: _PersistentModel)
    monkeypatch.setattr(api_server, "prepare_persistent_state_service", _prepare)
    engine = object.__new__(async_omni_module.AsyncOmni)
    engine.engine = SimpleNamespace(stage_clients=["stage"])
    engine._persistent_state_service = None
    runtime = object()
    monkeypatch.setattr(
        persistent_state_config.PersistentStateRuntimeConfig,
        "from_vllm_config",
        classmethod(lambda cls, config, *, startup_provider: runtime if startup_provider is provider else None),
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
    engine = SimpleNamespace(report_persistent_state_fatal=lambda error: events.append("report"))
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


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_hard_cap_preparation_attests_and_seals_without_profile_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec ENV-MIG-012 / PORT-PERF-005/006 / PORT-INT-013."""
    module = _startup_module()
    prepare = _startup_symbol("prepare_persistent_state_service")
    events: list[str] = []
    forbidden_profile_fields = frozenset(
        {
            "service_profile_trailing_rounds",
            "startup_priming_timeout_s",
            "service_profile_derating_factor",
            "priming_budget_descriptor",
            "priming_budget_sha256",
            "priming_configured_population_ceiling",
        }
    )

    class _HardCapRuntime(SimpleNamespace):
        def __getattr__(self, name: str) -> Any:
            if name in forbidden_profile_fields:
                _fail(f"PORT-PERF-006 hard_cap read profile-only field {name}")
            raise AttributeError(name)

    common = vars(_runtime()).copy()
    for name in forbidden_profile_fields:
        common.pop(name, None)
    runtime = _HardCapRuntime(**common)
    runtime.admission_policy = "hard_cap"
    runtime.bootstrap_operation_budget = 0
    runtime.service_profile_identity = None
    runtime.profile_status = "not_measured"

    class _HardCapProvider:
        def served_intervals_ms(self, *, model_config: Any) -> tuple[int, ...]:
            assert model_config is engine.model_config
            events.append("intervals")
            return (80, 320, 560, 1_120)

        def build_priming_plan(self, **kwargs: Any) -> NoReturn:
            del kwargs
            _fail("PORT-PERF-005 hard_cap constructed a priming plan")

    class _Service:
        def __init__(self, stage_client: Any, **kwargs: Any) -> None:
            assert stage_client == "stage"
            assert kwargs["runtime_config"] is runtime
            events.append("construct")
            self.ready = False

        async def bootstrap_handshake(self) -> dict[str, Any]:
            events.append("handshake")
            return _startup_inventory()

        def configure_bootstrap_intervals(self, intervals_ms: tuple[int, ...]) -> None:
            assert intervals_ms == (80, 320, 560, 1_120)
            events.append("configure")

        def seal_startup_authority(
            self,
            *,
            admission_config: Any,
            authority: Any,
        ) -> None:
            assert admission_config == "hard-controller"
            assert authority.policy == "hard_cap"
            assert authority.service_profile_identity is None
            assert authority.model_profile_id == "profile-a"
            assert authority.served_intervals_ms == (80, 320, 560, 1_120)
            events.append("seal-hard")
            self.ready = True

        def shutdown(self) -> None:
            events.append("shutdown")

    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        _fail("PORT-PERF-006 hard_cap invoked the service-profile compiler")

    engine = SimpleNamespace(model_config=object())
    monkeypatch.setattr(module, "PersistentStateService", _Service)
    monkeypatch.setattr(module, "run_service_priming_round", forbidden)
    monkeypatch.setattr(module, "compile_provisional_service_profile", forbidden)
    monkeypatch.setattr(
        module,
        "derive_admission_controller_config",
        lambda value, **kwargs: (
            "hard-controller"
            if value is runtime and kwargs["supported_intervals_ms"] == (80, 320, 560, 1_120)
            else _fail("PORT-STATE-027 hard-cap controller authority mismatch")
        ),
    )

    service = await prepare(
        engine_client=engine,
        stage_client="stage",
        runtime_config=runtime,
        startup_provider=_HardCapProvider(),
        host_fatal_callback=lambda error: None,
    )

    assert service.ready
    assert events == [
        "construct",
        "handshake",
        "intervals",
        "configure",
        "seal-hard",
    ]


def test_nemotron_provider_resolves_hard_cap_intervals_without_a_priming_plan() -> None:
    """@spec PORT-PERF-005 / PORT-STATE-026: model facts, no synthetic work."""
    from vllm_omni.model_executor.models.nemotron_asr.startup import (
        NEMOTRON_PERSISTENT_STATE_STARTUP,
    )

    resolver = getattr(
        NEMOTRON_PERSISTENT_STATE_STARTUP,
        "served_intervals_ms",
        None,
    )
    if not callable(resolver):
        _fail("PORT-PERF-005 missing dependency-light served-interval resolver")
    model_config = SimpleNamespace(hf_config=SimpleNamespace(supported_num_lookahead_tokens=[3, 0, 6, 13]))

    intervals = resolver(model_config=model_config)

    assert intervals == (80, 320, 560, 1_120)
    assert 160 not in intervals


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
        model_config=SimpleNamespace(hf_config=SimpleNamespace(supported_num_lookahead_tokens=[3, 0, 6, 13])),
    )

    tiers = plan.compile_kwargs["execution_tiers"]
    assert [(tier.tier_id, tier.max_active_population) for tier in tiers] == [
        ("single", 1),
        ("eager-bulk", 4),
    ]
    assert len(plan.rounds) == 4 * 2 * 4 * 3
    assert plan.actual_operation_count == 480
    assert len(plan.actual_plan_sha256) == 64
    assert plan.compile_kwargs["startup_priming_receipt"] == {
        "budget_sha256": runtime.priming_budget_descriptor.sha256,
        "bootstrap_operation_budget": 3_840,
        "actual_plan_sha256": plan.actual_plan_sha256,
        "actual_operation_count": 480,
    }
    assert sum(not round_spec.post_jit for round_spec in plan.rounds) == 24
    assert {(round_spec.geometry_id, round_spec.tier_id, round_spec.scenario_id) for round_spec in plan.rounds} == {
        (geometry_id, tier_id, scenario_id)
        for geometry_id in (0, 2, 3, 4)
        for tier_id in ("single", "eager-bulk")
        for scenario_id in (
            "ordinary",
            "forced_eou_then_chunk",
            "final_tail_then_flush",
        )
    }
    assert all(
        round_spec.expected_legal_parks_per_lease == (1 if round_spec.scenario_id == "ordinary" else 2)
        for round_spec in plan.rounds
    )
    assert plan.compile_kwargs["admitted_geometry_ids"] == (0, 2, 3, 4)
    assert plan.compile_kwargs["reference_geometry_id"] == 4
    assert plan.compile_kwargs["reference_interval_ms"] == 1120
    assert plan.served_intervals_ms == (80, 320, 560, 1120)
    assert plan.compile_kwargs.get("control_upper_ns_by_window_and_population") is None
    assert plan.compile_kwargs.get("control_dominance_sha256") is None


def test_nemotron_provider_keeps_all_manifest_geometries_without_declaration() -> None:
    """@spec PORT-PERF-005: absence preserves the full manifest table."""
    from vllm_omni.model_executor.models.nemotron_asr.startup import (
        NEMOTRON_PERSISTENT_STATE_STARTUP,
    )

    runtime = SimpleNamespace(
        service_profile_trailing_rounds=1,
        service_profile_derating_factor=0.5,
        priming_budget_descriptor=(
            NEMOTRON_PERSISTENT_STATE_STARTUP.build_priming_budget_descriptor(
                configured_population_ceiling=1,
                trailing_rounds=1,
            )
        ),
    )
    plan = NEMOTRON_PERSISTENT_STATE_STARTUP.build_priming_plan(
        runtime_config=runtime,
        inventory={
            "physical_capacity": 1,
            "effective_capacity": 1,
            "configured_limit": 1,
            "execution_claim_ceiling": 1,
            "slot_bytes": 6_314_936,
            "execution_environment_key": "env-a",
            "precision_policy": "torch.float32",
            "schema_id": "schema-a",
            "profile_id": "profile-a",
        },
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )

    assert plan.compile_kwargs["admitted_geometry_ids"] == (0, 1, 2, 3, 4)
    assert plan.compile_kwargs["reference_geometry_id"] == 4
    assert plan.compile_kwargs["reference_interval_ms"] == 1120
    assert plan.served_intervals_ms == (80, 160, 320, 560, 1120)


def test_nemotron_provider_rejects_empty_served_geometry_before_plan() -> None:
    """@spec PORT-PERF-005: unsupported declarations cannot mint a plan."""
    from vllm_omni.model_executor.models.nemotron_asr.startup import (
        NEMOTRON_PERSISTENT_STATE_STARTUP,
    )

    runtime = SimpleNamespace(
        service_profile_trailing_rounds=1,
        service_profile_derating_factor=0.5,
        priming_budget_descriptor=(
            NEMOTRON_PERSISTENT_STATE_STARTUP.build_priming_budget_descriptor(
                configured_population_ceiling=1,
                trailing_rounds=1,
            )
        ),
    )
    inventory = {
        "physical_capacity": 1,
        "effective_capacity": 1,
        "configured_limit": 1,
        "execution_claim_ceiling": 1,
        "slot_bytes": 6_314_936,
        "execution_environment_key": "env-a",
        "precision_policy": "torch.float32",
        "schema_id": "schema-a",
        "profile_id": "profile-a",
    }

    with pytest.raises(ValueError, match="no supported manifest geometry"):
        NEMOTRON_PERSISTENT_STATE_STARTUP.build_priming_plan(
            runtime_config=runtime,
            inventory=inventory,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(supported_num_lookahead_tokens=[999])),
        )


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
    assert budget.bootstrap_operation_budget == 3_840
    assert {
        (cell.geometry_id, cell.tier_id, cell.scenario_id, cell.max_active_population) for cell in budget.cells
    } == {
        (geometry_id, tier_id, scenario_id, population)
        for geometry_id in range(5)
        for tier_id, population in (("single", 1), ("eager-bulk", 8))
        for scenario_id in (
            "ordinary",
            "forced_eou_then_chunk",
            "final_tail_then_flush",
        )
    }

    single = NEMOTRON_PERSISTENT_STATE_STARTUP.build_priming_budget_descriptor(
        configured_population_ceiling=1,
        trailing_rounds=3,
    )
    target = NEMOTRON_PERSISTENT_STATE_STARTUP.build_priming_budget_descriptor(
        configured_population_ceiling=1_000,
        trailing_rounds=3,
    )
    assert single.bootstrap_operation_budget == 120
    assert target.bootstrap_operation_budget == 1_000_000


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

        async def force_segment(self) -> None:
            events.append(("force", self.request_id))

        async def flush(self) -> SimpleNamespace:
            events.append(("flush", self.request_id))
            return SimpleNamespace(text="")

        async def abort(self) -> None:
            events.append(("abort", self.request_id))

    monkeypatch.setattr(binding_module, "NemotronSessionLease", _Bound)
    monkeypatch.setattr(
        session_module.NemotronRealtimeSession,
        "from_model_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(prompt_dictionary={"en-US": 0})))
    leases = tuple(
        SimpleNamespace(
            session_key=f"session-{index}",
            engine_epoch="epoch-a",
            generation=index + 1,
        )
        for index in range(2)
    )
    round_spec = SimpleNamespace(
        service_interval_ms=80,
        scenario_id="ordinary",
    )

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


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
@pytest.mark.parametrize(
    ("scenario_id", "expected_action_order"),
    [
        ("forced_eou_then_chunk", ("setup_feed", "force", "feed")),
        ("final_tail_then_flush", ("feed", "flush")),
    ],
)
async def test_nemotron_provider_executes_exceptional_control_chain(
    monkeypatch: pytest.MonkeyPatch,
    scenario_id: str,
    expected_action_order: tuple[str, ...],
) -> None:
    """@spec PORT-PERF-005/006: canaries traverse the real lease API."""
    from vllm_omni.entrypoints import nemotron_session as binding_module
    from vllm_omni.model_executor.models.nemotron_asr import (
        session as session_module,
    )
    from vllm_omni.model_executor.models.nemotron_asr import startup as startup_module
    from vllm_omni.model_executor.models.nemotron_asr.startup import (
        NEMOTRON_PERSISTENT_STATE_STARTUP,
    )

    events: list[tuple[str, str]] = []

    class _Bound:
        def __init__(self, **kwargs: Any) -> None:
            self.request_id = str(kwargs["request_id"])
            self._first_chunk_registered = False

        async def feed(self, samples: Any) -> list[str]:
            expected_samples = 1_279 if scenario_id == "final_tail_then_flush" else 1_280
            assert samples.shape == (expected_samples,)
            action = (
                "setup_feed" if scenario_id == "forced_eou_then_chunk" and not self._first_chunk_registered else "feed"
            )
            self._first_chunk_registered = True
            events.append((action, self.request_id))
            return [""]

        async def force_segment(self) -> None:
            assert self._first_chunk_registered, "forced EOU must follow the first CHUNK"
            events.append(("force", self.request_id))

        async def flush(self) -> SimpleNamespace:
            events.append(("flush", self.request_id))
            return SimpleNamespace(text="")

        async def abort(self) -> None:
            events.append(("abort", self.request_id))

    monkeypatch.setattr(binding_module, "NemotronSessionLease", _Bound)
    monkeypatch.setattr(
        session_module.NemotronRealtimeSession,
        "from_model_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(prompt_dictionary={"en-US": 0})))
    leases = (
        SimpleNamespace(
            session_key="session-0",
            engine_epoch="epoch-a",
            generation=1,
        ),
    )
    round_spec = SimpleNamespace(
        service_interval_ms=80,
        scenario_id=scenario_id,
    )
    if scenario_id == "forced_eou_then_chunk":
        clock = iter((100, 200, 230))
        monkeypatch.setattr(startup_module.time, "monotonic_ns", lambda: next(clock))

    result = await NEMOTRON_PERSISTENT_STATE_STARTUP.execute_priming_round(
        engine_client=engine,
        round_spec=round_spec,
        leases=leases,
    )

    assert result.completed_legal_parks == 2
    assert tuple(action for action, _ in events[:-1]) == expected_action_order
    assert events[-1] == ("abort", "session-0")
    if scenario_id == "forced_eou_then_chunk":
        assert result.elapsed_ns == 30
