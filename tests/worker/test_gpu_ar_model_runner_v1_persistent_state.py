# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the MRV1 persistent-state differential lane."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_v1_runner_remains_a_thin_core_runner_subclass() -> None:
    # @spec PORT-MIG-002 / PORT-MIG-006
    assert issubclass(GPUARModelRunner, GPUModelRunner)


def test_v1_discovers_one_persistent_group_without_mamba_identity() -> None:
    # @spec PORT-STATE-002 / PORT-MIG-006
    source = inspect.getsource(GPUARModelRunner.get_kv_cache_spec)

    assert "super().get_kv_cache_spec" in source
    assert "PersistentStateLayerBase" in source
    assert "PersistentStateSpec" in source
    assert "MambaSpec" not in source


def test_v1_initialization_partitions_persistent_state_before_core_dispatch() -> None:
    # @spec PORT-STATE-002 / PORT-MIG-006
    source = inspect.getsource(GPUARModelRunner.initialize_kv_cache)

    partition = source.index("PersistentStateSpec")
    delegate = source.index("super().initialize_kv_cache")
    assert partition < delegate
    assert "prepare_kernel_block_sizes" not in source[:delegate]
    assert "MambaSpec" not in source


def test_v1_reshape_handles_persistent_state_and_delegates_other_groups() -> None:
    # @spec PORT-STATE-002 / PORT-MIG-006
    source = inspect.getsource(GPUARModelRunner._reshape_kv_cache_tensors)

    assert "PersistentStateSpec" in source
    assert "super()._reshape_kv_cache_tensors" in source
    assert "MambaSpec" not in source


def test_v1_and_v2_use_the_same_runner_neutral_batch_builder() -> None:
    # @spec PORT-STATE-007 / PORT-MIG-005
    try:
        from vllm_omni.worker.gpu_ar_model_runner_v2 import GPUARModelRunnerV2
    except ModuleNotFoundError:
        pytest.fail("PORT-MIG-005 missing GPUARModelRunnerV2 module", pytrace=False)

    v1_source = inspect.getsource(GPUARModelRunner._build_persistent_state_batch)
    v2_source = inspect.getsource(GPUARModelRunnerV2._build_persistent_state_batch)

    for source in (v1_source, v2_source):
        assert "build_persistent_state_batch" in source
        assert "StateLease" not in source
        assert "PersistentStateService" not in source


def test_v1_runner_never_owns_physical_lifecycle() -> None:
    # @spec PORT-STATE-014 / PORT-MIG-006
    source = inspect.getsource(GPUARModelRunner)

    for forbidden in (
        "PersistentStateService",
        "persistent_state_reserve",
        "persistent_state_release",
        "free_slot",
    ):
        assert forbidden not in source


@pytest.mark.parametrize("enabled", [True, False])
def test_v1_prefix_cache_guard_runs_before_initial_profile(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    # @spec PORT-INT-007 / PORT-STATE-002
    if "profile_run" not in GPUARModelRunner.__dict__:
        pytest.fail(
            "PORT-INT-007 missing V1 pre-profile persistent-state guard",
            pytrace=False,
        )
    runner: Any = object.__new__(GPUARModelRunner)
    runner.vllm_config = object()
    runner.cache_config = SimpleNamespace(enable_prefix_caching=enabled)
    profile_calls: list[str] = []
    module = importlib.import_module("vllm_omni.worker.gpu_ar_model_runner")
    monkeypatch.setattr(
        module,
        "discover_persistent_state_specs",
        lambda config: {"persistent": object()},
    )
    monkeypatch.setattr(
        OmniGPUModelRunner,
        "profile_run",
        lambda self: profile_calls.append("profile"),
        raising=False,
    )

    if enabled:
        with pytest.raises(ValueError, match="prefix caching"):
            GPUARModelRunner.profile_run(runner)
        assert profile_calls == []
    else:
        GPUARModelRunner.profile_run(runner)
        assert profile_calls == ["profile"]
