# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests-first contract for selected persistent-state startup composition."""

from __future__ import annotations

import asyncio
import importlib
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn
from unittest.mock import Mock

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
        startup_priming_round_timeout_s=2.0,
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


@pytest.mark.parametrize("policy", ("profile", "hard_cap"))
@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_preparation_orders_bootstrap_priming_seal_and_installability(
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    """@spec ENV-MIG-012 / PORT-PERF-005/006 / PORT-INT-013."""
    module = _startup_module()
    prepare = _startup_symbol("prepare_persistent_state_service")
    events: list[str] = []
    runtime = _runtime()
    runtime.admission_policy = policy
    model_config = object()
    engine = SimpleNamespace(model_config=model_config)
    provider = _Provider(events, runtime, model_config)
    profile = SimpleNamespace(
        compiled_demand=SimpleNamespace(intervals_ms=(80, 320, 560, 1120)),
        receipt=SimpleNamespace(receipt_sha256="advisory-profile"),
    )

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

        def seal_startup_authority(
            self, *, admission_config: Any, authority: Any, compiled_service_profile: Any
        ) -> None:
            assert authority.service_profile_identity == "advisory-profile"
            assert compiled_service_profile is profile
            self.seal_startup_profile(
                admission_config=admission_config, compiled_service_profile=compiled_service_profile
            )

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
        hard_cap_capacity: int | None = None,
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


@pytest.mark.parametrize("policy", ("profile", "hard_cap"))
@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_preparation_timeout_is_primary_and_never_seals(
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    """@spec ENV-MIG-012 / PORT-PERF-005: the complete plan is bounded."""
    module = _startup_module()
    prepare = _startup_symbol("prepare_persistent_state_service")
    events: list[str] = []
    runtime = _runtime()
    runtime.admission_policy = policy
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

    assert plan.compile_kwargs["context"].compiler_version == "persistent-state-service-v2"
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
        round_id="geometry-0-bulk-ordinary-repeat-1",
        geometry_id=0,
        service_interval_ms=80,
        scenario_id="ordinary",
        active_population=2,
        tier_id="bulk",
        post_jit=True,
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
        round_id=f"geometry-0-single-{scenario_id}-repeat-1",
        geometry_id=0,
        service_interval_ms=80,
        scenario_id=scenario_id,
        active_population=1,
        tier_id="single",
        post_jit=True,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ("profile", "hard_cap"))
@pytest.mark.parametrize("unsupported", (False, True))
async def test_preparation_handles_failed_valid_canary_by_policy(monkeypatch, policy, unsupported):
    """@spec PORT-PERF-006, PORT-STATE-026, PORT-STATE-027: preparation seals unsupported cadences with one warning."""
    from dataclasses import replace
    from fractions import Fraction
    from unittest.mock import Mock

    from vllm_omni.engine.persistent_state_capacity import (
        ServiceExecutionTier,
        ServiceProfileContext,
        ServiceRoundExecution,
    )

    module = _startup_module()
    # Spy on the compiler bound at startup, including across module reloads.
    logger = module.compile_provisional_service_profile.__globals__["logger"]
    warning = Mock(wraps=logger.warning)
    monkeypatch.setattr(logger, "warning", warning)
    runtime = _runtime()
    runtime.admission_policy = policy
    events = []
    from tests.engine.persistent_state_helpers import canary_timing_kwargs

    ordinary = ServiceRoundExecution(
        tier_id="single",
        active_population=1,
        elapsed_ns=100,
        service_interval_ms=80,
        geometry_id=0,
        completed_legal_parks=1,
        completed_model_rows=None,
        post_jit=True,
        continuously_loaded=True,
        dummy_run=False,
        is_profile=False,
    )
    rounds = (
        ordinary,
        *(
            replace(
                ordinary,
                scenario_id=scenario,
                elapsed_ns=201,
                completed_legal_parks=2,
                **canary_timing_kwargs(scenario),
            )
            for scenario in ("forced_eou_then_chunk", "final_tail_then_flush")
        ),
    )

    if unsupported:
        rounds = tuple(
            replace(s, elapsed_ns=50_000_000 if s.scenario_id == "ordinary" else 100_000_001) for s in rounds
        ) + tuple(
            replace(
                s,
                geometry_id=1,
                service_interval_ms=160,
                elapsed_ns=100 if s.scenario_id == "ordinary" else 200,
                **canary_timing_kwargs(s.scenario_id, interval_ms=160),
            )
            for s in rounds
        )
    intervals = (80, 160) if unsupported else (80,)

    class Provider:
        def build_priming_plan(self, **kwargs):
            return SimpleNamespace(
                rounds=rounds,
                served_intervals_ms=intervals,
                compile_kwargs=dict(
                    execution_tiers=(ServiceExecutionTier("single", 1),),
                    max_population=1,
                    reference_interval_ms=80,
                    reference_geometry_id=0,
                    admitted_geometry_ids=tuple(range(len(intervals))),
                    trailing_rounds=1,
                    derating_factor=Fraction(1, 2),
                    context=ServiceProfileContext(
                        pre_override_physical_bound=5,
                        allocated_pool=4,
                        count_cap=4,
                        execution_claim_ceiling=8,
                        service_budget_source="measured_fallback",
                        service_budget_coefficients=(),
                        derating_factor=Fraction(1, 2),
                        slot_bytes=6_314_936,
                        execution_environment_key="env-a",
                        precision_policy="torch.float32",
                        state_profile="profile-a",
                        compiler_version="test-v1",
                        mixed_composition_policy="periodic_limited_preemption_edf",
                    ),
                ),
            )

    class Service:
        def __init__(self, *args, **kwargs):
            self.ready = False

        async def bootstrap_handshake(self):
            return _startup_inventory()

        def configure_bootstrap_intervals(self, intervals):
            assert intervals == ((80, 160) if unsupported else (80,))

        def seal_startup_profile(self, *, admission_config, compiled_service_profile):
            self.seal_startup_authority(
                authority=SimpleNamespace(service_profile_identity=compiled_service_profile.receipt.receipt_sha256),
                admission_config=admission_config,
                compiled_service_profile=compiled_service_profile,
            )

        def seal_startup_authority(self, *, authority, admission_config, compiled_service_profile):
            assert authority.service_profile_identity == compiled_service_profile.receipt.receipt_sha256
            receipt = compiled_service_profile.receipt
            assert receipt.control_charge_status == ("proven_zero" if policy == "profile" else "unproven")
            if unsupported and policy == "profile":
                assert set(receipt.measured_unsupported_by_geometry) == {0}
                assert receipt.measured_unsupported_by_geometry[0].control_ns == 0
                cells = receipt.control_dominance_evidence["cells"]
                failed = [cell for cell in cells if not cell["dominance_passed"]]
                assert len(failed) == 2
                assert all(cell["geometry_id"] == 0 and not cell["admission_relevant"] for cell in failed)
            self.ready = True
            events.append("seal")

        def shutdown(self):
            events.append("shutdown")

    async def prime(**kwargs):
        assert not kwargs["service"].ready
        return kwargs["round_spec"]

    monkeypatch.setattr(module, "PersistentStateService", Service)
    monkeypatch.setattr(module, "run_service_priming_round", prime)
    kwargs = dict(
        engine_client=SimpleNamespace(model_config=None),
        stage_client=None,
        runtime_config=runtime,
        startup_provider=Provider(),
        host_fatal_callback=lambda error: None,
    )
    if policy == "profile" and not unsupported:
        with pytest.raises(ValueError, match="control-dominance failed"):
            await module.prepare_persistent_state_service(**kwargs)
        assert events == ["shutdown"]
    else:
        service = await module.prepare_persistent_state_service(**kwargs)
        assert service.ready and events == ["seal"]
        assert warning.call_count == 1
        if policy == "profile":
            assert "measured-unsupported" in warning.call_args.args[0]


@pytest.fixture
def startup_correlation_probe(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Exercise the provider with existing lease APIs and fake NVTX, without CUDA."""
    from vllm.logger import init_logger

    from vllm_omni.entrypoints import nemotron_session as binding_module
    from vllm_omni.model_executor.models.nemotron_asr import profiling, session, startup

    logger = init_logger(startup.__name__)
    warning = Mock(wraps=logger.warning)
    monkeypatch.setattr(logger, "warning", warning)
    events: list[tuple[str, Any]] = []
    failures: dict[str, BaseException] = {}
    clock_calls = 0

    def range_start(name: str) -> int:
        events.append(("start", name))
        if "start" in failures:
            raise failures["start"]
        return 0

    def range_end(handle: int) -> None:
        events.append(("end", handle))
        if "end" in failures:
            raise failures["end"]

    def clock() -> int:
        nonlocal clock_calls
        clock_calls += 1
        events.append(("clock", clock_calls))
        if f"clock-{clock_calls}" in failures:
            raise failures[f"clock-{clock_calls}"]
        return (100, 200, 230, 270)[clock_calls - 1]

    class Bound:
        def __init__(self, **kwargs: Any) -> None:
            self.request_id = kwargs["request_id"]
            events.append(("construct", self.request_id))
            self.session = SimpleNamespace(accepted_audio=SimpleNamespace(prior_actual_submission_ns=220))

        async def feed(self, samples: Any) -> list[str]:
            events.append(("feed", (self.request_id, len(samples))))
            if "feed" in failures:
                raise failures["feed"]
            return [""]

        async def wait_for_legal_park(self) -> None:
            events.append(("park", self.request_id))

        def stage_forced_eou_then_chunk(self, samples: Any) -> SimpleNamespace:
            events.append(("stage", self.request_id))
            return SimpleNamespace(true_ready_ns=100, prior_actual_submission_ns=100, cadence_ns=10, eligible_ns=190)

        async def dispatch_staged_priming_chain(self) -> None:
            events.append(("dispatch", self.request_id))

        async def flush(self) -> SimpleNamespace:
            events.append(("flush", self.request_id))
            return SimpleNamespace(text="")

        async def abort(self) -> None:
            events.append(("abort", self.request_id))

    monkeypatch.setattr(binding_module, "NemotronSessionLease", Bound)
    monkeypatch.setattr(session.NemotronRealtimeSession, "from_model_config", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(profiling, "_ENABLED", True)
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range_start", range_start)
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range_end", range_end)
    monkeypatch.setattr(startup.time, "monotonic_ns", clock)

    async def run(
        geometry: int = 0, repeat: int = 1, scenario: str = "ordinary", tier: str = "single", enabled: bool = True
    ) -> Any:
        monkeypatch.setattr(profiling, "_ENABLED", enabled)
        population = 2 if tier == "bulk" else 1
        return await startup.NEMOTRON_PERSISTENT_STATE_STARTUP.execute_priming_round(
            engine_client=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=SimpleNamespace(prompt_dictionary={"en-US": 0}))
            ),
            round_spec=SimpleNamespace(
                round_id=f"geometry-{geometry}-{tier}-{scenario}-repeat-{repeat}",
                geometry_id=geometry,
                service_interval_ms=80 if geometry == 0 else 160,
                scenario_id=scenario,
                active_population=population,
                tier_id=tier,
                post_jit=repeat > 0,
            ),
            leases=tuple(
                SimpleNamespace(session_key=f"session-{index}", engine_epoch="epoch-a", generation=index + 1)
                for index in range(population)
            ),
        )

    return SimpleNamespace(run=run, events=events, failures=failures, warning=warning)


# @spec PORT-PERF-005, PORT-PERF-006, PORT-PERF-007
@pytest.mark.asyncio
@pytest.mark.parametrize("geometry", (0, 1))
@pytest.mark.parametrize("repeat", range(4))
async def test_startup_correlation_actual_provider_preserves_ordinary_timer(
    startup_correlation_probe: Any, geometry: int, repeat: int
) -> None:
    probe = startup_correlation_probe
    result = await probe.run(geometry=geometry, repeat=repeat)
    assert probe.events == [
        ("construct", "session-0"),
        ("start", f"port.startup.measurement.geometry-{geometry}-single-ordinary-repeat-{repeat}"),
        ("clock", 1),
        ("feed", ("session-0", 1_280 if geometry == 0 else 2_560)),
        ("clock", 2),
        ("end", 0),
        ("abort", "session-0"),
    ]
    assert result.elapsed_ns == 100
    assert result.completed_legal_parks == 1
    assert (result.canary_wait_ns, result.canary_wait_start_ns, result.canary_timer_start_ns) == (0, None, None)
    assert result.following_chunk_timings == ()
    assert probe.warning.call_count == 0
    assert (result.completed_model_rows, result.dummy_run, result.is_profile) == (None, False, False)


# @spec PORT-PERF-005, PORT-PERF-006, PORT-PERF-007
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "tier", "actions", "elapsed", "parks"),
    [
        ("final_tail_then_flush", "single", ("clock", "feed", "flush", "clock", "abort"), 100, 2),
        (
            "forced_eou_then_chunk",
            "single",
            ("clock", "feed", "park", "stage", "clock", "clock", "dispatch", "clock", "abort"),
            40,
            2,
        ),
        ("ordinary", "bulk", ("clock", "feed", "feed", "clock", "abort", "abort"), 100, 2),
    ],
)
async def test_startup_correlation_excludes_canaries_and_bulk(
    startup_correlation_probe: Any, scenario: str, tier: str, actions: tuple[str, ...], elapsed: int, parks: int
) -> None:
    probe = startup_correlation_probe
    result = await probe.run(scenario=scenario, tier=tier)
    construction = ("construct",) * (2 if tier == "bulk" else 1)
    assert tuple(action for action, _ in probe.events) == construction + actions
    assert probe.warning.call_count == 0
    assert (result.elapsed_ns, result.completed_legal_parks) == (elapsed, parks)
    if scenario == "forced_eou_then_chunk":
        assert (result.canary_wait_start_ns, result.canary_timer_start_ns, result.canary_wait_ns) == (200, 230, 30)
        assert len(result.following_chunk_timings) == 1
        assert result.following_chunk_timings[0].submitted_ns == 220


# @spec PORT-PERF-005, PORT-PERF-006, PORT-PERF-007
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_point", "cleanup_end_fails"),
    [(point, cleanup) for point in ("start", "clock-1", "clock-2", "feed", "cancel") for cleanup in (False, True)]
    + [("normal-end", False)],
)
async def test_startup_correlation_failures_close_once_and_always_abort(
    startup_correlation_probe: Any,
    failure_point: str,
    cleanup_end_fails: bool,
) -> None:
    probe = startup_correlation_probe
    primary: BaseException = (
        asyncio.CancelledError("feed cancellation")
        if failure_point == "cancel"
        else ValueError(f"primary {failure_point}")
    )
    point = "feed" if failure_point == "cancel" else "end" if failure_point == "normal-end" else failure_point
    probe.failures[point] = primary
    if cleanup_end_fails and failure_point != "normal-end":
        probe.failures["end"] = RuntimeError("secondary NVTX end")
    with pytest.raises(type(primary)) as caught:
        await probe.run()
    assert caught.value is primary
    actions = [action for action, _ in probe.events]
    assert probe.events[:2] == [
        ("construct", "session-0"),
        ("start", "port.startup.measurement.geometry-0-single-ordinary-repeat-1"),
    ]
    assert probe.events[-1] == ("abort", "session-0")
    assert actions.count("abort") == 1
    assert actions.count("end") == (0 if failure_point == "start" else 1)
    if "end" in actions:
        assert actions.index("end") < actions.index("abort")
    if failure_point in ("start", "clock-1"):
        assert "feed" not in actions
    if failure_point == "start":
        assert "clock" not in actions
    expected_warning_count = int(cleanup_end_fails and failure_point not in ("start", "normal-end"))
    assert probe.warning.call_count == expected_warning_count
    if expected_warning_count:
        assert "nvtx" in str(probe.warning.call_args).lower()


# @spec PORT-PERF-005, PORT-PERF-006, PORT-PERF-007
@pytest.mark.asyncio
async def test_startup_correlation_actual_provider_disabled_preserves_ordinary_timer(
    startup_correlation_probe: Any,
) -> None:
    probe = startup_correlation_probe
    result = await probe.run(enabled=False)
    assert probe.events == [
        ("construct", "session-0"),
        ("clock", 1),
        ("feed", ("session-0", 1_280)),
        ("clock", 2),
        ("abort", "session-0"),
    ]
    assert result.elapsed_ns == 100
    assert result.completed_legal_parks == 1
    assert probe.warning.call_count == 0
