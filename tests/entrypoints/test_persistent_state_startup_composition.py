# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for selected persistent-state startup composition."""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_STARTUP_MODULE = "vllm_omni.engine.persistent_state_startup"


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
        service_profile_trailing_rounds=2,
        service_profile_derating_factor=0.5,
        reserve_queue_capacity=8,
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
            compile_kwargs={"trailing_rounds": 2},
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
            return {
                "resident_state_scatter_warmup_complete": True,
                "engine_epoch": "epoch-a",
            }

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
        classmethod(lambda cls, config: runtime),
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
