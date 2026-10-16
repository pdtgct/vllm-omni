# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-specific startup evidence checks; no CUDA/numerical claim from these mocks."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import torch

from tests.model_executor.models.nemotron_asr import test_native_burst as native
from tests.model_executor.models.nemotron_asr.test_outer_tensor_fusion import (
    CONSTANTS,
    FAMILIES,
    _post_inputs,
    _pre_inputs,
    _symbol,
)
from vllm_omni.model_executor.models.nemotron_asr import startup

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _ready_receipt() -> dict[str, Any]:
    declared = []
    for family in FAMILIES:
        geometry = int(family[-1]) if family.startswith("post-g") else None
        tensors = _pre_inputs() if family == "pre" else _post_inputs(geometry)
        declared.append(
            {
                "id": family,
                "region": "pre" if family == "pre" else "post",
                "geometry": geometry,
                "inputs": [
                    {
                        "name": name,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype).removeprefix("torch."),
                        "device": "cuda:0",
                        "stride": list(t.stride()),
                        "storage_offset_bytes": 0,
                        "requires_grad": False,
                        "object_group": i,
                        "storage_group": i,
                    }
                    for i, (name, t) in enumerate(tensors.items())
                ],
            }
        )
    return {
        "schema": "nemotron-execution-profile/1",
        "emission": {"lane": "native_burst", "profile": "fp32-sync-single-device-five-cadence-v1"},
        "encoder": {"arm": "eager-graphed", "ready": True},
        "decode": {"arm": "dense-graphed", "ready": True, "captured_keys": [[g, 1] for g in range(5)]},
        "outer_tensor_fusion": {
            "schema": "nemotron-outer-tensor-fusion/1",
            "arm": "compiled-static",
            "profile": "fp32-sync-single-device-five-cadence-v1",
            "constants": dict(CONSTANTS),
            "compiler": {
                "backend": "inductor",
                "torch": "2.13.0+cu130",
                "triton": "3.7.1",
                "fullgraph": True,
                "dynamic": False,
                "options": {"triton.cudagraphs": False},
            },
            "context": {
                "default_dtype": "float32",
                "deterministic_algorithms": False,
                "deterministic_warn_only": False,
                "grad_enabled": False,
                "inference_mode": True,
                "autocast_cpu_enabled": False,
                "autocast_cpu_dtype": "bfloat16",
                "autocast_cuda_enabled": False,
                "autocast_cuda_dtype": "float16",
                "dispatch_modes": [],
            },
            "declared": declared,
            "prepared": list(FAMILIES),
            "real_profile_covered": list(FAMILIES),
            "warmup_complete": True,
            "sealed": True,
            "ready": True,
            "frames": {
                "accepted": dict.fromkeys(FAMILIES, 1),
                "at_seal": dict.fromkeys(FAMILIES, 1),
                "backend_attempts": 7,
                "attempts_at_seal": 7,
                "post_seal_attempts": 0,
                "undeclared_attempts": 0,
            },
            "dispatch": {
                r: {"fused_calls": 40, "eager_calls": 0, "fused_at_seal": 0, "eager_at_seal": 0}
                for r in ("pre", "post")
            },
        },
    }


class _Engine:
    def __init__(self, enabled, events, *, pid=42):
        hf = native._native_config(profile=native.FIVE_CADENCE_PROFILE, lookaheads=native.LOOKAHEADS)
        hf.experimental_outer_tensor_fusion = enabled
        hf.encoder_execution_arm = "eager-graphed"
        hf.decode_dispatch_arm = "dense-graphed"
        hf.num_asr_labels = 12
        hf.eos_token_id, hf.audio_chunk_token_id, hf.eou_token_id = 9000, 9001, 9002
        hf.hidden_size = 17927
        self.model_config = SimpleNamespace(hf_config=hf, dtype=torch.float32)
        self.events = events
        self.receipt = _ready_receipt()
        self.pid = pid
        self.warmup_complete = True
        self.fail_read = False
        self.after_abort = False

    async def collective_rpc(self, method, *args, **kwargs):
        assert method == "nemotron_execution_profile_receipt"
        self.events.append("rpc")
        if self.fail_read:
            raise RuntimeError("receipt read failed")
        # AsyncOmni preserves the stage/replica list and each worker-result list.
        return [
            [
                {
                    "worker_pid": self.pid,
                    "persistent_state_warmup_complete": self.warmup_complete,
                    "execution_profile": deepcopy(self.receipt),
                }
            ]
        ]

    def execute(self):
        block = self.receipt["outer_tensor_fusion"]
        for region in ("pre", "post"):
            block["dispatch"][region]["fused_calls"] += 1


def _plan(provider, engine, *, trailing_rounds=1, admission_policy="profile"):
    runtime = SimpleNamespace(
        service_profile_trailing_rounds=trailing_rounds,
        admission_policy=admission_policy,
        service_profile_derating_factor=0.5,
        priming_budget_descriptor=provider.build_priming_budget_descriptor(
            configured_population_ceiling=1,
            trailing_rounds=trailing_rounds,
        ),
    )
    inventory = {
        "physical_capacity": 1,
        "effective_capacity": 1,
        "configured_limit": 1,
        "execution_claim_ceiling": 1,
        "slot_bytes": 1000,
        "execution_environment_key": "env-a",
        "precision_policy": "torch.float32",
        "schema_id": "schema-a",
        "profile_id": "profile-a",
        "engine_epoch": "epoch-a",
        "stage": 0,
        "replica": 0,
    }
    return provider.build_priming_plan(runtime_config=runtime, inventory=inventory, model_config=engine.model_config)


def _bound_fakes(monkeypatch, engine, *, no_op=False, poison=None):
    from vllm_omni.entrypoints import nemotron_session
    from vllm_omni.model_executor.models.nemotron_asr import profiling, session

    class Bound:
        def __init__(self, **kwargs):
            engine.events.append("construct")
            self.session = SimpleNamespace(accepted_audio=SimpleNamespace(prior_actual_submission_ns=150))

        async def feed(self, samples):
            engine.events.append("feed")
            if not no_op:
                engine.execute()

        async def flush(self):
            engine.events.append("flush")
            if not no_op:
                engine.execute()

        async def wait_for_legal_park(self):
            engine.events.append("park")

        def stage_forced_eou_then_chunk(self, samples):
            return SimpleNamespace(eligible_ns=1, true_ready_ns=0, prior_actual_submission_ns=0, cadence_ns=80_000_000)

        async def dispatch_staged_priming_chain(self):
            engine.events.append("dispatch-chain")
            if not no_op:
                engine.execute()

        async def abort(self):
            engine.events.append("abort")
            engine.after_abort = True
            if poison is not None:
                poison(engine)

    monkeypatch.setattr(nemotron_session, "NemotronSessionLease", Bound)
    monkeypatch.setattr(session.NemotronRealtimeSession, "from_model_config", lambda *a, **kw: SimpleNamespace())
    ticks = iter(range(100, 100000, 100))

    def now():
        engine.events.append("clock")
        return next(ticks)

    monkeypatch.setattr(startup.time, "monotonic_ns", now)
    monkeypatch.setattr(profiling, "start_startup_round", lambda _: None)
    monkeypatch.setattr(profiling, "end_startup_round", lambda _: None)


async def _boundary(provider, engine, plan, phase, context=None):
    return await _symbol(provider, "validate_priming_boundary")(
        engine_client=engine,
        plan=plan,
        phase=phase,
        boundary_context=context,
    )


def _leases():
    return (SimpleNamespace(session_key="session-1", engine_epoch="epoch-a", generation=1),)


@pytest.mark.asyncio
async def test_actual_orchestrator_worker_result_shape_reaches_provider(monkeypatch):
    """@spec PORT-PERF-025: consume the actual stage/replica aggregation shape."""
    from vllm_omni.engine.messages import CollectiveRPCRequestMessage
    from vllm_omni.engine.orchestrator import Orchestrator
    from vllm_omni.engine.stage_pool import StagePool

    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(True, [])
    plan = _plan(provider, engine)
    worker = {
        "worker_pid": engine.pid,
        "persistent_state_warmup_complete": True,
        "execution_profile": deepcopy(engine.receipt),
    }
    # The core executor returns one list item per worker. Exercise the real pool
    # forwarding and orchestrator aggregation rather than duplicating that logic.
    client_rpc = AsyncMock(return_value=[worker])
    pool = object.__new__(StagePool)
    pool.stage_id = 0
    pool.clients = [SimpleNamespace(stage_type="llm", collective_rpc_async=client_rpc)]
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [pool]
    orchestrator.num_stages = 1
    orchestrator.rpc_async_queue = asyncio.Queue()

    async def collective_rpc(method, **kwargs):
        await orchestrator._handle_collective_rpc(
            CollectiveRPCRequestMessage(rpc_id="receipt", method=method, args=(), kwargs={}, stage_ids=None)
        )
        result = orchestrator.rpc_async_queue.get_nowait()
        assert result.stage_ids == [0]
        assert result.results == [[worker]]
        return result.results

    monkeypatch.setattr(engine, "collective_rpc", collective_rpc)
    context = await _boundary(provider, engine, plan, "before")
    assert context.worker_pid == engine.pid
    client_rpc.assert_awaited_once_with(method="nemotron_execution_profile_receipt", timeout=None, args=(), kwargs={})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ("non-sequence", "exactly one stage/replica"),
        ("empty-outer", "exactly one stage/replica"),
        ("multiple-replicas", "exactly one stage/replica"),
        ("flat", "exactly one expected worker"),
        ("empty-inner", "exactly one expected worker"),
        ("multiple-workers", "exactly one expected worker"),
        ("over-nested", "worker receipt envelope is malformed"),
        ("scalar-worker", "worker receipt envelope is malformed"),
        ("stage-error", "exactly one expected worker"),
        ("worker-error", "worker receipt envelope is malformed"),
    ],
)
async def test_receipt_rpc_shape_rejects_malformed_membership(monkeypatch, shape, message):
    """@spec PORT-PERF-025: exactly one stage/replica and one worker are required."""
    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(True, [])
    plan = _plan(provider, engine)
    worker = (await engine.collective_rpc("nemotron_execution_profile_receipt"))[0][0]
    responses: object = {
        "non-sequence": None,
        "empty-outer": [],
        "multiple-replicas": [[worker], [worker]],
        "flat": [worker],
        "empty-inner": [[]],
        "multiple-workers": [[worker, worker]],
        "over-nested": [[[worker]]],
        "scalar-worker": [[None]],
        "stage-error": [{"supported": False, "error": "unavailable"}],
        "worker-error": [[{"error": "receipt read failed"}]],
    }[shape]
    monkeypatch.setattr(engine, "collective_rpc", AsyncMock(return_value=responses))
    with pytest.raises(RuntimeError, match=message):
        await _boundary(provider, engine, plan, "before")


@pytest.mark.asyncio
@pytest.mark.parametrize("warmup", [False, None, 1, "true"])
@pytest.mark.parametrize("snapshot", ["pre-plan", "before-round", "after-round", "final-boundary"])
async def test_worker_warmup_requires_true_at_every_snapshot(monkeypatch, warmup, snapshot):
    """@spec PORT-PERF-025: unwrapping never bypasses the worker readiness flag."""
    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(True, [])
    plan = _plan(provider, engine)
    if snapshot == "pre-plan":
        engine.warmup_complete = warmup
        with pytest.raises(RuntimeError, match="worker warmup is incomplete"):
            await _boundary(provider, engine, plan, "before")
        return
    context = await _boundary(provider, engine, plan, "before")
    if snapshot == "final-boundary":
        engine.execute()
        engine.warmup_complete = warmup
        with pytest.raises(RuntimeError, match="worker warmup is incomplete"):
            await _boundary(provider, engine, plan, "after", context)
        return

    def poison(current):
        current.warmup_complete = warmup

    _bound_fakes(monkeypatch, engine, poison=poison if snapshot == "after-round" else None)
    if snapshot == "before-round":
        engine.warmup_complete = warmup
    with pytest.raises(RuntimeError, match="worker warmup is incomplete"):
        await provider.execute_priming_round(
            engine_client=engine, round_spec=plan.rounds[0], leases=_leases(), boundary_context=context
        )
    if snapshot == "before-round":
        assert "feed" not in engine.events
    else:
        assert engine.events[-2:] == ["abort", "rpc"]


@pytest.mark.asyncio
async def test_final_boundary_rejects_worker_pid_change():
    """@spec PORT-PERF-025: final cleanup cannot replace the expected worker."""
    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(True, [])
    plan = _plan(provider, engine)
    context = await _boundary(provider, engine, plan, "before")
    engine.execute()
    engine.pid += 1
    with pytest.raises(RuntimeError, match="worker identity/counters changed"):
        await _boundary(provider, engine, plan, "after", context)


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_chunk_off", [False, True])
async def test_default_off_round_preserves_exact_rpc_clock_and_cleanup_behavior(monkeypatch, explicit_chunk_off):
    """@spec PORT-PERF-012, PORT-PERF-025, PORT-PERF-005, PORT-PERF-007: no disabled-path evidence RPC."""
    events: list[str] = []
    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(False, events)
    if explicit_chunk_off:
        engine.model_config.hf_config.experimental_chunk_bucket_graph_cells = []
        engine.model_config.hf_config.experimental_chunk_graph_startup_attestation = False
    plan = _plan(provider, engine)
    context = await _boundary(provider, engine, plan, "before")
    assert context is None and events == []
    _bound_fakes(monkeypatch, engine)
    result = await provider.execute_priming_round(
        engine_client=engine,
        round_spec=plan.rounds[0],
        leases=_leases(),
        boundary_context=context,
    )
    await _boundary(provider, engine, plan, "after", context)
    assert result.elapsed_ns == 100 and result.completed_legal_parks == 1
    assert events == ["construct", "clock", "feed", "clock", "abort"]


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["all-noop", "only-pre", "only-post", "eager-gap", "backend-gap", "frame-gap"])
async def test_whole_plan_completion_rejects_missing_execution_and_final_gap_growth(defect):
    """@spec PORT-PERF-025: final check covers the gap after last round/lease release."""
    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(True, [])
    plan = _plan(provider, engine)
    context = await _boundary(provider, engine, plan, "before")
    block = engine.receipt["outer_tensor_fusion"]
    if defect in {"only-pre", "only-post"}:
        block["dispatch"][defect.removeprefix("only-")]["fused_calls"] += 1
    elif defect != "all-noop":
        engine.execute()
        if defect == "eager-gap":
            block["dispatch"]["post"]["eager_calls"] += 1
        elif defect == "backend-gap":
            block["frames"]["backend_attempts"] += 1
        else:
            block["frames"]["accepted"]["post-control"] += 1
    with pytest.raises((ValueError, RuntimeError)):
        await _boundary(provider, engine, plan, "after", context)


def _assert_immutable_context(value):
    if isinstance(value, Mapping):
        with pytest.raises(TypeError):
            cast(Any, value)["caller-mutation"] = True
        for item in value.values():
            _assert_immutable_context(item)
    elif is_dataclass(value):
        for field in fields(value):
            with pytest.raises((AttributeError, TypeError)):
                setattr(value, field.name, None)
            _assert_immutable_context(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            _assert_immutable_context(item)
    else:
        assert value is None or isinstance(value, (str, bytes, bool, int, float)), type(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["ordinary", "forced_eou_then_chunk", "final_tail_then_flush"])
@pytest.mark.parametrize("post_jit", [False, True])
async def test_enabled_round_readbacks_bracket_every_scenario_outside_u_after_abort(monkeypatch, scenario, post_jit):
    """@spec PORT-PERF-025, PORT-PERF-005, PORT-PERF-006: leading rounds and canaries are included."""
    events: list[str] = []
    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(True, events)
    plan = _plan(provider, engine)
    context = await _boundary(provider, engine, plan, "before")
    # Freeze launch expectation in plan/round authority; later mutable config is irrelevant.
    engine.model_config.hf_config.experimental_outer_tensor_fusion = False
    _bound_fakes(monkeypatch, engine)
    spec = next(s for s in plan.rounds if s.scenario_id == scenario and s.post_jit == post_jit)
    start = len(events)
    result = await provider.execute_priming_round(
        engine_client=engine,
        round_spec=spec,
        leases=_leases(),
        boundary_context=context,
    )
    assert result.elapsed_ns > 0
    sequence = events[start:]
    assert sequence.count("rpc") == 2
    assert sequence.index("rpc") < sequence.index("clock")
    assert sequence[-2:] == ["abort", "rpc"]
    assert "rpc" not in sequence[sequence.index("clock") : sequence.index("abort")]


@pytest.mark.asyncio
async def test_legitimate_control_noop_has_no_fabricated_fused_delta(monkeypatch):
    """@spec PORT-PERF-025: no-op controls retain existing park-unit semantics."""
    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(True, [])
    plan = _plan(provider, engine)
    context = await _boundary(provider, engine, plan, "before")
    _bound_fakes(monkeypatch, engine, no_op=True)
    spec = next(s for s in plan.rounds if s.scenario_id == "final_tail_then_flush")
    before = deepcopy(engine.receipt)
    result = await provider.execute_priming_round(
        engine_client=engine,
        round_spec=spec,
        leases=_leases(),
        boundary_context=context,
    )
    assert result.completed_legal_parks == 2 and engine.receipt == before


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["eager", "backend", "frame", "identity", "pid", "reset", "missing"])
async def test_between_round_gap_and_worker_identity_corruption_fail_before_work(monkeypatch, defect):
    """@spec PORT-PERF-020, PORT-PERF-022, PORT-PERF-025: seal-relative checks cover unsampled gaps."""
    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(True, [])
    plan = _plan(provider, engine)
    context = await _boundary(provider, engine, plan, "before")
    block = engine.receipt["outer_tensor_fusion"]
    if defect == "eager":
        block["dispatch"]["pre"]["eager_calls"] += 1
    elif defect == "backend":
        block["frames"]["backend_attempts"] += 1
    elif defect == "frame":
        block["frames"]["accepted"]["pre"] += 1
    elif defect == "identity":
        block["constants"]["blank_id"] += 1
    elif defect == "pid":
        engine.pid += 1
    elif defect == "reset":
        block["dispatch"]["pre"]["fused_calls"] = 0
    else:
        del engine.receipt["outer_tensor_fusion"]
    _bound_fakes(monkeypatch, engine)
    with pytest.raises((ValueError, RuntimeError)):
        await provider.execute_priming_round(
            engine_client=engine,
            round_spec=plan.rounds[1],
            leases=_leases(),
            boundary_context=context,
        )
    assert "feed" not in engine.events
    if "construct" in engine.events:
        assert "abort" in engine.events


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["read-before", "read-after", "eager-after", "compile-after", "pid-after"])
async def test_receipt_failures_preserve_abort_and_reject_round(monkeypatch, failure):
    """@spec PORT-PERF-021, PORT-PERF-025, PORT-STATE-008: no retry after selected-call failure."""
    provider = startup.NemotronPersistentStateStartupProvider()
    engine = _Engine(True, [])
    plan = _plan(provider, engine)
    context = await _boundary(provider, engine, plan, "before")

    def poison(current):
        block = current.receipt["outer_tensor_fusion"]
        if failure == "read-after":
            current.fail_read = True
        elif failure == "eager-after":
            block["dispatch"]["post"]["eager_calls"] += 1
        elif failure == "compile-after":
            block["frames"]["backend_attempts"] += 1
        elif failure == "pid-after":
            current.pid += 1

    _bound_fakes(monkeypatch, engine, poison=poison)
    if failure == "read-before":
        engine.fail_read = True
    with pytest.raises((ValueError, RuntimeError)):
        await provider.execute_priming_round(
            engine_client=engine,
            round_spec=plan.rounds[0],
            leases=_leases(),
            boundary_context=context,
        )
    if "construct" in engine.events:
        assert engine.events.count("abort") == 1
    assert engine.events.count("feed") <= 1


@pytest.mark.asyncio
async def test_shared_provider_contexts_are_engine_local_detached_and_worker_checked(monkeypatch):
    """@spec PORT-PERF-020, PORT-PERF-022, PORT-PERF-025: singleton provider must not retain cross-plan state."""
    provider = startup.NEMOTRON_PERSISTENT_STATE_STARTUP
    first, second = _Engine(True, [], pid=42), _Engine(True, [], pid=84)
    plan_a, plan_b = _plan(provider, first), _plan(provider, second)
    context_a = await _boundary(provider, first, plan_a, "before")
    context_b = await _boundary(provider, second, plan_b, "before")
    assert context_a is not context_b
    _assert_immutable_context(context_a)
    _assert_immutable_context(context_b)
    frozen_a = repr(context_a)
    first.execute()
    second.execute()
    await _boundary(provider, first, plan_a, "after", context_a)
    await _boundary(provider, second, plan_b, "after", context_b)
    with pytest.raises((ValueError, RuntimeError)):
        await _boundary(provider, second, plan_b, "after", context_a)
    first.receipt["outer_tensor_fusion"]["constants"]["blank_id"] = -1
    assert repr(context_a) == frozen_a
    with pytest.raises((ValueError, RuntimeError)):
        await _boundary(provider, first, plan_a, "after", context_a)


CHUNK_LOG_PREFIX = "Nemotron CHUNK startup attestation "


def _chunk_engine(*, selected=True, observe=True):
    engine = _Engine(False, [])
    hf = engine.model_config.hf_config
    hf.experimental_chunk_bucket_graph_cells = [[0, 1]] if selected else []
    hf.experimental_chunk_graph_startup_attestation = observe
    domain = [[g, 1] for g in range(5)]
    engine.receipt.pop("outer_tensor_fusion")
    engine.receipt["encoder"].update(
        warmup_geometries=list(range(5)),
        warmup_populations=[1],
        warmup_cells=deepcopy(domain),
        captured_keys=deepcopy(domain[1:] if selected else domain),
    )
    if selected:
        engine.receipt["encoder"].update(chunk_graph_keys=[[0, 1]], reserved_chunk_graph_keys=[[0, 1]])
        engine.receipt["chunk"] = {
            "ready": True,
            "worker_pid": engine.pid,
            "instance_id": "42:12345",
            "source_sha256": {
                name: "a" * 64
                for name in (
                    "advance.py",
                    "encoder.py",
                    "encoder_execution.py",
                    "decode_graph.py",
                    "rnnt.py",
                    "chunk_bucket_graph.py",
                    "nemotron_asr.py",
                    "profile_execution.py",
                )
            },
            "cells": [{"geometry": 0, "encoder_population": 1, "decoder_tier": 1, "successful_replays": 7}],
            "fallbacks": [],
        }

    def execute():
        if selected:
            engine.receipt["chunk"]["cells"][0]["successful_replays"] += 1

    setattr(engine, "execute", execute)
    return engine


def _chunk_log_data(caplog):
    import hashlib
    import json

    messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith(CHUNK_LOG_PREFIX)]
    assert len(messages) == 1, "requested three-sample CHUNK attestation log is missing or duplicated"
    envelope = json.loads(messages[0][len(CHUNK_LOG_PREFIX) :])
    data = envelope["data"]
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    assert envelope["data_sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(raw) <= 131072
    return data


@pytest.mark.parametrize("value", [None, 0, "", [], {}])
def test_chunk_analysis_present_nonboolean_rejected_during_plan_build(value):
    """@spec PORT-PERF-033: malformed falsey flags cannot quietly disable observation."""
    engine = _chunk_engine(observe=value)
    with pytest.raises(ValueError, match="(?i)boolean"):
        _plan(startup.NemotronPersistentStateStartupProvider(), engine, trailing_rounds=3)


@pytest.mark.parametrize("trailing_rounds", [1, 2, 4])
def test_chunk_analysis_requires_exact_three_round_roster_at_plan_build(trailing_rounds):
    """@spec PORT-PERF-033, PORT-PERF-034: no vacuous zero/short witness."""
    with pytest.raises(ValueError, match="(?i)three|3|trailing|roster"):
        _plan(startup.NemotronPersistentStateStartupProvider(), _chunk_engine(), trailing_rounds=trailing_rounds)


@pytest.mark.parametrize(
    "field,value",
    [
        ("supported_num_lookahead_tokens", [1]),
        ("experimental_chunk_bucket_graph_cells", [[1, 1]]),
    ],
)
def test_chunk_analysis_rejects_wrong_domain_at_plan_build(field, value):
    """@spec PORT-PERF-033: analysis checks happen at provider plan construction."""
    engine = _chunk_engine()
    setattr(engine.model_config.hf_config, field, value)
    with pytest.raises(ValueError, match="(?i)chunk|attestation|geometry|lookahead|native"):
        _plan(startup.NemotronPersistentStateStartupProvider(), engine, trailing_rounds=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("admission_policy", ["profile", "hard_cap"])
async def test_chunk_analysis_brackets_only_final_three_and_emits_canonical_owned_log(
    monkeypatch, caplog, selected, admission_policy
):
    """@spec PORT-PERF-033, PORT-PERF-034, PORT-PERF-035: exact sample witness, not setup credit."""
    import hashlib
    import json

    caplog.set_level("INFO")
    import os

    from vllm_omni.model_executor.models.nemotron_asr.batch_invariance import installed_batch_invariant_mode

    provider, engine = startup.NemotronPersistentStateStartupProvider(), _chunk_engine(selected=selected)
    plan = _plan(provider, engine, trailing_rounds=3, admission_policy=admission_policy)
    # Derive expectations from independently supplied config/plan inputs before mutation,
    # never from a worker receipt or the finalized witness. hard_cap is observable even
    # though the separate experimental harness must reject it for the support gate.
    hf = engine.model_config.hf_config
    expected_config = {
        "geometry_cells": [list(cell) for cell in sorted({(r.geometry_id, r.active_population) for r in plan.rounds})],
        "chunk_cells": deepcopy(hf.experimental_chunk_bucket_graph_cells),
        "encoder_arm": hf.encoder_execution_arm,
        "decode_arm": hf.decode_dispatch_arm,
        "dtype": str(engine.model_config.dtype).removeprefix("torch."),
        "batch_invariant_mode": "on" if installed_batch_invariant_mode().enabled else "off",
        "emission_lane": "native_burst" if hf.experimental_native_burst else "compatibility_replay",
        "emission_profile": hf.experimental_native_burst_profile,
        "max_num_seqs": plan.compile_kwargs["max_population"],
        "admission_policy": admission_policy,
        "derating_numerator": plan.compile_kwargs["derating_factor"].numerator,
        "derating_denominator": plan.compile_kwargs["derating_factor"].denominator,
        "trailing_rounds": plan.compile_kwargs["trailing_rounds"],
    }
    expected_api_pid, expected_worker_pid = os.getpid(), engine.pid
    before_identity = (plan.actual_plan_sha256, plan.actual_operation_count, deepcopy(plan.compile_kwargs))
    context = await _boundary(provider, engine, plan, "before")
    _bound_fakes(monkeypatch, engine)
    from vllm_omni.model_executor.models.nemotron_asr import profiling

    monkeypatch.setattr(profiling, "start_startup_round", lambda _: engine.events.append("range-start"))
    monkeypatch.setattr(profiling, "end_startup_round", lambda _: engine.events.append("range-end"))
    expected = []
    # Frozen plan controls observation even if the HF config is changed afterward.
    engine.model_config.hf_config.experimental_chunk_graph_startup_attestation = False
    for ordinal, spec in enumerate(plan.rounds):
        start = len(engine.events)
        result = await provider.execute_priming_round(
            engine_client=engine,
            round_spec=spec,
            leases=_leases(),
            boundary_context=context,
        )
        events = engine.events[start:]
        measured = spec.geometry_id == 0 and spec.scenario_id == "ordinary" and spec.post_jit
        if measured:
            assert events.count("rpc") == 2, "timed g0 sample lacks its own BEFORE/AFTER pair"
            assert events.index("rpc") < events.index("range-start")
            assert (
                events.index("range-end")
                < max(i for i, event in enumerate(events) if event == "rpc")
                < events.index("abort")
            )
            assert "rpc" not in events[events.index("range-start") : events.index("range-end")]
            assert result.elapsed_ns == 100, "observer time leaked into U"
            expected.append((ordinal, spec.round_id, result.elapsed_ns))
        else:
            assert events.count("rpc") == 0, "leading/canary/other geometry was needlessly observed"
    await _boundary(provider, engine, plan, "after", context)
    assert (plan.actual_plan_sha256, plan.actual_operation_count, plan.compile_kwargs) == before_identity
    assert "chunk_graph_attestation" not in plan.compile_kwargs["startup_priming_receipt"]
    data = _chunk_log_data(caplog)
    assert data["schema"] == "nemotron-chunk-startup-attestation/1"
    assert data["expected"] == expected_config
    assert data["api_pid"] == expected_api_pid
    assert data["worker_pid"] == expected_worker_pid == engine.pid
    assert data["verdict"] == "PASS"
    assert data["plan_sha256"] == plan.actual_plan_sha256
    assert [(r["plan_ordinal"], r["round_id"], r["elapsed_ns"]) for r in data["samples"]] == expected
    assert len(expected) == 3
    identity_keys = (
        "plan_ordinal",
        "round_id",
        "geometry_id",
        "tier_id",
        "active_population",
        "scenario_id",
        "post_jit",
        "repeat",
    )
    roster = [{key: record[key] for key in identity_keys} for record in data["samples"]]
    assert (
        data["roster_sha256"]
        == hashlib.sha256(
            json.dumps(
                roster,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    for record in data["samples"]:
        assert record["completed_legal_parks"] == 1
        assert (
            record["before_response_ns"]
            <= record["timer_start_ns"]
            <= record["timer_end_ns"]
            <= record["after_request_ns"]
        )
        assert record["elapsed_ns"] == record["timer_end_ns"] - record["timer_start_ns"]
        before = record["before"][0][0]["execution_profile"]
        after = record["after"][0][0]["execution_profile"]
        if selected:
            assert after["chunk"]["cells"][0]["successful_replays"] > before["chunk"]["cells"][0]["successful_replays"]
        else:
            assert "chunk" not in before and "chunk" not in after


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect", ["no-replay", "fallback", "changed-instance", "nonfinite", "oversized", "inventory", "source"]
)
async def test_chunk_analysis_invalid_interval_fails_startup_with_cleanup(monkeypatch, caplog, defect):
    """@spec PORT-PERF-034, PORT-PERF-035: provider enforces, not merely reports invalid evidence."""
    provider, engine = startup.NemotronPersistentStateStartupProvider(), _chunk_engine()
    plan = _plan(provider, engine, trailing_rounds=3)
    context = await _boundary(provider, engine, plan, "before")
    original = engine.execute

    def execute():
        if defect != "no-replay":
            original()
        block = engine.receipt["chunk"]
        if defect == "fallback":
            block["fallbacks"] = [{"geometry": 0, "encoder_population": 1, "successful_calls": 1}]
        elif defect == "changed-instance":
            block["instance_id"] = "foreign-instance"
        elif defect == "inventory":
            engine.receipt["encoder"]["captured_keys"].append([0, 1])
        elif defect == "source":
            block["source_sha256"]["chunk_bucket_graph.py"] = "b" * 64
        elif defect in {"nonfinite", "oversized"}:
            engine.receipt["encoder"]["memory_diagnostics"] = [
                {
                    "stage": "x" * 17000 if defect == "oversized" else "after-capture",
                    "allocated_bytes": 0 if defect == "oversized" else float("nan"),
                }
            ]

    engine.execute = execute
    _bound_fakes(monkeypatch, engine)
    spec = next(s for s in plan.rounds if s.geometry_id == 0 and s.scenario_id == "ordinary" and s.post_jit)
    with pytest.raises((ValueError, RuntimeError)):
        await provider.execute_priming_round(
            engine_client=engine, round_spec=spec, leases=_leases(), boundary_context=context
        )
    assert "abort" in engine.events
    assert not any(r.getMessage().startswith(CHUNK_LOG_PREFIX) for r in caplog.records)


@pytest.mark.asyncio
async def test_chunk_analysis_execution_error_keeps_primary_and_skips_after(monkeypatch):
    """@spec PORT-PERF-034: the failed operation cannot mint replay or be masked by readback."""
    provider, engine = startup.NemotronPersistentStateStartupProvider(), _chunk_engine()
    plan = _plan(provider, engine, trailing_rounds=3)
    context = await _boundary(provider, engine, plan, "before")
    primary = RuntimeError("primary model failure")

    def execute():
        engine.fail_read = True  # an accidental AFTER would obscure the primary error
        raise primary

    engine.execute = execute
    _bound_fakes(monkeypatch, engine)
    spec = next(s for s in plan.rounds if s.geometry_id == 0 and s.scenario_id == "ordinary" and s.post_jit)
    start = len(engine.events)
    with pytest.raises(RuntimeError, match="primary model failure") as caught:
        await provider.execute_priming_round(
            engine_client=engine, round_spec=spec, leases=_leases(), boundary_context=context
        )
    assert caught.value is primary
    events = engine.events[start:]
    assert events.count("rpc") == 1 and events[-1] == "abort"


def test_chunk_analysis_plan_samples_compile_to_exact_maximum_without_receipt_join():
    """@spec PORT-PERF-037: actual provider plan -> converter -> real compiler, source/test linkage only."""
    from tests.engine.persistent_state_helpers import canary_timing_kwargs
    from vllm_omni.engine.persistent_state_capacity import compile_provisional_service_profile
    from vllm_omni.engine.persistent_state_priming import ServicePrimingObservation
    from vllm_omni.engine.persistent_state_startup import _service_executions

    provider = startup.NemotronPersistentStateStartupProvider()
    plan = _plan(provider, _chunk_engine(observe=False), trailing_rounds=3)
    independent_plan = _plan(provider, _chunk_engine(observe=False), trailing_rounds=3)
    assert independent_plan.actual_plan_sha256 == plan.actual_plan_sha256
    assert independent_plan.rounds == plan.rounds
    observations = []
    selected = []
    for ordinal, spec in enumerate(plan.rounds):
        repeat = int(spec.round_id.rsplit("-", 1)[1])
        elapsed = (9999, 17, 31, 23)[repeat] if spec.scenario_id == "ordinary" else 10
        observations.append(
            ServicePrimingObservation(
                round_id=spec.round_id,
                service_interval_ms=spec.service_interval_ms,
                geometry_id=spec.geometry_id,
                active_population=spec.active_population,
                elapsed_ns=elapsed,
                completed_legal_parks=spec.expected_legal_parks_per_lease,
                completed_model_rows=None,
                tier_id=spec.tier_id,
                post_jit=spec.post_jit,
                scenario_id=spec.scenario_id,
                **canary_timing_kwargs(spec.scenario_id, interval_ms=spec.service_interval_ms),
            )
        )
        if spec.geometry_id == 0 and spec.scenario_id == "ordinary" and spec.post_jit:
            selected.append((ordinal, repeat, elapsed))
    assert selected == [(1, 1, 17), (2, 2, 31), (3, 3, 23)]
    converted = _service_executions(observations)
    assert [sample.elapsed_ns for sample in converted] == [sample.elapsed_ns for sample in observations]
    result = compile_provisional_service_profile(converted, admission_policy="profile", **plan.compile_kwargs)
    assert result.upper_duration_ns_by_geometry_and_population[0] == (max(x[2] for x in selected),)
    assert result.receipt.fragmentation_duration_ns_by_geometry_and_population[0] == (31,)
    assert result.receipt.control_charge_status == "proven_zero"
    assert result.receipt.derating_factor.numerator == 1
    assert result.receipt.derating_factor.denominator == 2
    assert result.receipt.count_cap == 1
    assert result.service_interval_ms_by_geometry[0] == 80
