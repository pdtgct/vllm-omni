# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Execution inventory must coexist with the final no-resident memory profile."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr import profile_execution as profile
from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import NemotronASRForRNNT
from vllm_omni.model_executor.models.nemotron_asr.precision import (
    BF16_COMPUTE,
    FP16_ENCODER_EXPERIMENT,
    FP32_BRINGUP,
)
from vllm_omni.worker import base as worker_module
from vllm_omni.worker import gpu_ar_model_runner_v2 as runner_module
from vllm_omni.worker import persistent_state as worker_state

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(monkeypatch, *, graphs=True, fail=None, policy=FP32_BRINGUP):
    model = object.__new__(NemotronASRForRNNT)
    events = []
    parameter = torch.zeros(1, dtype=policy.dtype_for("encoder_compute"))
    execution = SimpleNamespace(arm="compiled-static" if graphs else "eager", ready=not graphs)
    execution.ready_receipt = lambda: {"ready": execution.ready}
    binding = SimpleNamespace(captured_keys=()) if graphs else None

    def encoder_warmup(value, *, device):
        assert value is model
        assert torch.is_inference_mode_enabled()
        events.append("encoder")
        if fail == "encoder":
            raise RuntimeError("encoder failed")
        execution.ready = True

    def decoder_warmup(device, dtype):
        assert binding is not None
        assert device == parameter.device and dtype == policy.dtype_for("activations")
        assert torch.is_inference_mode_enabled()
        events.append("decoder")
        if fail == "decoder":
            raise RuntimeError("decoder failed")
        binding.captured_keys = ((0, 1),) if fail == "partial" else ((0, 1), (0, 2))

    if binding is not None:
        binding.warmup = decoder_warmup

    def final_profile(value, *, num_rows, device, geometry_id, ready_domain=False):
        assert value is model
        assert num_rows == 2 and device == parameter.device
        assert geometry_id == 0
        assert torch.is_inference_mode_enabled()
        assert ready_domain and execution.ready
        assert binding is None or binding.captured_keys == ((0, 1), (0, 2))
        events.append("final-profile")
        if fail == "final":
            raise RuntimeError("final failed")

    def no_resident():
        pytest.fail("memory preparation touched resident state")

    for name, value in {
        "config": SimpleNamespace(
            supported_num_lookahead_tokens=[0], decode_dispatch_arm="dense-graphed" if graphs else "dense-eager"
        ),
        "core": SimpleNamespace(encoder=SimpleNamespace(parameters=lambda: iter([parameter])), policy=policy),
        "_max_num_seqs": 2,
        "_encoder_execution": execution,
        "_decode_graph_binding": binding,
        "_emission_selection": ("compatibility_replay", None),
        "_execution_memory_prepared": False,
        "_execution_memory_failed": False,
        "_state_pools": no_resident,
    }.items():
        object.__setattr__(model, name, value)
    monkeypatch.setattr(profile, "warmup_static_encoder_execution", encoder_warmup)
    monkeypatch.setattr(profile, "run_persistent_state_profile", final_profile)
    return model, events, execution, binding


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize(
    "policy", [FP32_BRINGUP, BF16_COMPUTE, FP16_ENCODER_EXPERIMENT], ids=["fp32", "bf16", "fp16-encoder"]
)
def test_preparation_retains_complete_inventory_for_final_profile(monkeypatch, graphs, policy):
    """@spec PORT-PREC-001, PORT-STATE-003, PORT-PERF-009, PORT-PERF-011."""
    model, events, execution, binding = _model(monkeypatch, graphs=graphs, policy=policy)
    model.prepare_execution_memory()
    assert events == (["encoder", "decoder", "final-profile"] if graphs else ["encoder", "final-profile"])
    assert model._encoder_execution is execution
    assert model._decode_graph_binding is binding
    assert model._execution_memory_prepared
    model.prepare_execution_memory()
    assert events.count("final-profile") == 1


@pytest.mark.parametrize("failure", ["encoder", "decoder", "partial", "final"])
def test_preparation_failure_never_allows_resident_warmup_or_retry(monkeypatch, failure):
    """@spec PORT-STATE-003, PORT-ADV-003, PORT-PERF-011."""
    model, events, _, _ = _model(monkeypatch, fail=failure)
    with pytest.raises(RuntimeError, match="failed|inventory"):
        model.prepare_execution_memory()
    assert not model._execution_memory_prepared
    before = list(events)
    with pytest.raises(RuntimeError, match="failed|prepared|inventory"):
        model.warmup_resident_state()
    with pytest.raises(RuntimeError, match="failed"):
        model.prepare_execution_memory()
    assert events == before


def test_resident_warmup_requires_prebuilt_inventory_and_never_recaptures(monkeypatch):
    """@spec PORT-ADV-003, PORT-PERF-011."""
    model, events, _, binding = _model(monkeypatch)
    with pytest.raises(RuntimeError, match="prepared|inventory"):
        model.warmup_resident_state()
    model.prepare_execution_memory()
    before = list(events)
    resident_reads = []

    def pools():
        resident_reads.append(True)
        return SimpleNamespace(predictor_h=torch.zeros(1))

    object.__setattr__(model, "_state_pools", pools)
    model.warmup_resident_state()
    assert resident_reads == [True]
    assert events == before
    binding.captured_keys = ((0, 1),)
    with pytest.raises(RuntimeError, match="inventory"):
        model.warmup_resident_state()
    assert resident_reads == [True]


def test_runner_prepares_after_core_profile_inside_worker_measurement(monkeypatch):
    """@spec PORT-MIG-005, PORT-STATE-003: no nested core forward or allocation."""
    events = []
    window = {"open": False, "forward": False}

    def core_profile(self):
        assert window["open"]
        window["forward"] = True
        events.append("core-profile")
        window["forward"] = False

    def prepare():
        assert window["open"] and not window["forward"]
        assert torch.is_inference_mode_enabled()
        events.append("inventory-and-final-profile")

    monkeypatch.setattr(runner_module.GPUModelRunner, "profile_run", core_profile)
    monkeypatch.setattr(runner_module, "discover_persistent_state_specs", lambda _: {})
    runner = object.__new__(runner_module.GPUARModelRunnerV2)
    runner.vllm_config = object()
    runner.model = SimpleNamespace(prepare_execution_memory=prepare)
    runner.model_memory_usage = 100
    runner.get_kv_cache_spec = lambda: {}
    monkeypatch.setattr(worker_state, "discover_persistent_state_specs", lambda _: {})

    @contextmanager
    def measurement(*args, **kwargs):
        window["open"] = True
        events.append("measure")
        yield SimpleNamespace(
            non_torch_increase=0,
            transient_peak_headroom=100,
            total_consumed=400,
            non_kv_cache_memory=500,
        )
        events.append("measured")
        window["open"] = False

    monkeypatch.setattr(worker_module, "memory_profiling", measurement)
    worker = object.__new__(worker_module.OmniGPUWorkerBase)
    worker.model_runner = runner
    worker.vllm_config = runner.vllm_config
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
    worker.init_snapshot = object()
    worker.requested_memory = 1_000
    worker.local_rank = 0
    assert worker.determine_available_memory() == 500
    assert not window["open"]
    assert events == ["measure", "core-profile", "inventory-and-final-profile", "measured"]


def test_runner_propagates_inventory_failure(monkeypatch):
    """@spec PORT-STATE-003: worker cannot size a pool after failed preparation."""
    monkeypatch.setattr(runner_module.GPUModelRunner, "profile_run", lambda _: None)
    monkeypatch.setattr(runner_module, "discover_persistent_state_specs", lambda _: {})
    runner = object.__new__(runner_module.GPUARModelRunnerV2)
    runner.vllm_config = object()

    def fail():
        raise RuntimeError("inventory failed")

    runner.model = SimpleNamespace(prepare_execution_memory=fail)
    with pytest.raises(RuntimeError, match="inventory failed"):
        runner.profile_run()


def _fusion_inventory_model(monkeypatch, *, failure=None):
    model, events, execution, decoder = _model(monkeypatch)
    fusion = SimpleNamespace(ready=False, sealed=False, warmup_calls=0)
    fusion.receipt = lambda: {"ready": fusion.ready, "sealed": fusion.sealed}
    object.__setattr__(model, "_outer_tensor_execution", fusion)

    def warm_outer(value, *, device):
        assert value is model and device.type == "cpu"
        assert events == ["encoder", "decoder"]
        assert torch.is_inference_mode_enabled()
        fusion.warmup_calls += 1
        events.append("outer-preparation-and-real-profile-coverage")
        if failure == "warmup":
            raise RuntimeError("outer compilation failed")
        if failure == "coverage":
            raise RuntimeError("outer canonical coverage incomplete")
        fusion.ready = True
        fusion.sealed = True

    monkeypatch.setattr(profile, "warmup_outer_tensor_execution", warm_outer, raising=False)
    original_profile = profile.run_persistent_state_profile

    def final_profile(*args, **kwargs):
        assert fusion.ready and fusion.sealed, "final memory profile preceded fusion inventory"
        return original_profile(*args, **kwargs)

    monkeypatch.setattr(profile, "run_persistent_state_profile", final_profile)
    return model, events, fusion


def test_outer_inventory_precedes_final_coexistence_profile_and_does_not_reprepare(monkeypatch):
    """@spec PORT-PERF-018, PORT-PERF-020, PORT-STATE-003: model lifecycle, not CPU compiler proof."""
    model, events, fusion = _fusion_inventory_model(monkeypatch)
    model.prepare_execution_memory()
    assert events == ["encoder", "decoder", "outer-preparation-and-real-profile-coverage", "final-profile"]
    model.config.experimental_outer_tensor_fusion = False
    model.prepare_execution_memory()
    assert fusion.warmup_calls == 1
    assert model._outer_tensor_execution is fusion


@pytest.mark.parametrize("failure", ["warmup", "coverage"])
def test_outer_preparation_failure_latches_no_retry_or_resident_access(monkeypatch, failure):
    """@spec PORT-PERF-019, PORT-PERF-020, PORT-STATE-003: failed preparation requires fresh worker."""
    model, events, fusion = _fusion_inventory_model(monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match="outer"):
        model.prepare_execution_memory()
    assert model._execution_memory_failed and not model._execution_memory_prepared
    before = list(events)
    for attempt in (model.prepare_execution_memory, model.warmup_resident_state):
        with pytest.raises(RuntimeError, match="failed|prepared"):
            attempt()
    assert events == before and fusion.warmup_calls == 1


def test_outer_inventory_cannot_hide_behind_decoder_early_return(monkeypatch):
    """@spec PORT-PERF-018: every execution inventory gates readiness."""
    model, _, _, _ = _model(monkeypatch, graphs=False)
    object.__setattr__(model, "_outer_tensor_execution", SimpleNamespace(ready=False))
    with pytest.raises(RuntimeError, match="outer|fusion|inventory"):
        model._require_execution_inventory_ready()
