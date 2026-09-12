# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Executable runner-dispatch checks for the custom state group."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2

from tests.model_executor.persistent_state._helpers import make_generic_spec
from vllm_omni.model_executor.persistent_state import PersistentStateLayerBase
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_model_runner_v2 import GPUARModelRunnerV2
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _state_config(spec: Any) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * spec.page_size_bytes,
                layers=["persistent_state"],
                layer_stride=4 * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["persistent_state"],
                kv_cache_spec=spec,
            )
        ],
    )


def _runner_config(spec: Any) -> Any:
    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    PersistentStateLayerBase(
        spec,
        prefix="persistent_state",
        vllm_config=config,
    )
    return config


@pytest.mark.parametrize(
    ("runner_cls", "base_cls"),
    (
        (GPUARModelRunner, OmniGPUModelRunner),
        (GPUARModelRunnerV2, GPUModelRunnerV2),
    ),
)
def test_runner_initialization_partitions_and_binds_state_storage(
    monkeypatch: pytest.MonkeyPatch,
    runner_cls: type[Any],
    base_cls: type[Any],
) -> None:
    """@spec PORT-STATE-002 / PORT-MIG-006."""

    from vllm_omni.model_executor import persistent_state

    spec = make_generic_spec(persistent_state)
    config = _runner_config(spec)
    runner = object.__new__(runner_cls)
    runner.vllm_config = config
    runner.device = "cpu"
    delegated: list[KVCacheConfig] = []

    allocation_context = nullcontext()

    def initialize(
        self: Any,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        del self
        assert is_profiling is False
        assert kv_cache_allocation_context is allocation_context
        delegated.append(kv_cache_config)

    monkeypatch.setattr(base_cls, "initialize_kv_cache", initialize)
    runner.initialize_kv_cache(_state_config(spec), kv_cache_allocation_context=allocation_context)

    assert len(delegated) == 1
    assert delegated[0].kv_cache_groups == []
    assert delegated[0].kv_cache_tensors == []
    storage = runner._persistent_state_storage
    assert storage.spec == spec
    assert storage.raw.numel() == 4 * spec.page_size_bytes
    layer = config.compilation_config.static_forward_context["persistent_state"]
    assert layer.persistent_state_storage is storage
