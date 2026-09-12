# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Execution inventory must coexist with the final no-resident memory profile."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr import profile_execution as profile
from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import NemotronASRForRNNT
from vllm_omni.worker import base as worker_module
from vllm_omni.worker import gpu_ar_model_runner_v2 as runner_module
from vllm_omni.worker import persistent_state as worker_state

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(monkeypatch, *, graphs=True, fail=None):
    model = object.__new__(NemotronASRForRNNT)
    events = []
    parameter = torch.zeros(1)
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
        assert device == parameter.device and dtype == parameter.dtype
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
        "core": SimpleNamespace(encoder=SimpleNamespace(parameters=lambda: iter([parameter]))),
        "_max_num_seqs": 2,
        "_encoder_execution": execution,
        "_decode_graph_binding": binding,
        "_execution_memory_prepared": False,
        "_execution_memory_failed": False,
        "_state_pools": no_resident,
    }.items():
        object.__setattr__(model, name, value)
    monkeypatch.setattr(profile, "warmup_static_encoder_execution", encoder_warmup)
    monkeypatch.setattr(profile, "run_persistent_state_profile", final_profile)
    return model, events, execution, binding


@pytest.mark.parametrize("graphs", [False, True])
def test_preparation_retains_complete_inventory_for_final_profile(monkeypatch, graphs):
    """@spec PORT-STATE-003, PORT-PERF-009, PORT-PERF-011."""
    model, events, execution, binding = _model(monkeypatch, graphs=graphs)
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
