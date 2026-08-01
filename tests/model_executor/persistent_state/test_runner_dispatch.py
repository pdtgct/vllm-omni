# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executable runner-dispatch checks for the custom state group."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

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


def _state_config(spec: Any) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * spec.page_size_bytes,
                shared_by=["persistent_state"],
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
    ("runner_cls", "base_cls", "accepts_profiling"),
    (
        (GPUARModelRunner, OmniGPUModelRunner, True),
        (GPUARModelRunnerV2, GPUModelRunnerV2, False),
    ),
)
def test_runner_initialization_partitions_and_binds_state_storage(
    monkeypatch: pytest.MonkeyPatch,
    runner_cls: type[Any],
    base_cls: type[Any],
    accepts_profiling: bool,
) -> None:
    """@spec PORT-STATE-002 / PORT-MIG-006."""

    from vllm_omni.model_executor import persistent_state

    spec = make_generic_spec(persistent_state)
    config = _runner_config(spec)
    runner = object.__new__(runner_cls)
    runner.vllm_config = config
    runner.device = "cpu"
    delegated: list[KVCacheConfig] = []

    def initialize_v1(
        self: Any,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        del self
        assert is_profiling is False
        delegated.append(kv_cache_config)

    def initialize_v2(
        self: Any,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        del self
        delegated.append(kv_cache_config)

    initialize = cast(
        Callable[..., None],
        initialize_v1 if accepts_profiling else initialize_v2,
    )
    monkeypatch.setattr(base_cls, "initialize_kv_cache", initialize)
    runner.initialize_kv_cache(_state_config(spec))

    assert len(delegated) == 1
    assert delegated[0].kv_cache_groups == []
    assert delegated[0].kv_cache_tensors == []
    storage = runner._persistent_state_storage
    assert storage.spec == spec
    assert storage.raw.numel() == 4 * spec.page_size_bytes
    layer = config.compilation_config.static_forward_context["persistent_state"]
    assert layer.persistent_state_storage is storage
