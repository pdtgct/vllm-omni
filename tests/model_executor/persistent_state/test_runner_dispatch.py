# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Executable runner-dispatch checks for the custom state group."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.utils import allocate_kv_cache

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


def _runner_config(spec: Any, layout: KVCacheLayout = KVCacheLayout.LBHNC) -> Any:
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: layout),
        model_config=SimpleNamespace(enable_sleep_mode=False, enable_cumem_allocator=False),
    )
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
@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_runner_initialization_partitions_and_binds_state_storage(
    monkeypatch: pytest.MonkeyPatch,
    runner_cls: type[Any],
    base_cls: type[Any],
    layout: KVCacheLayout,
) -> None:
    """@spec PORT-STATE-002, PORT-MIG-006: context ownership and byte aliasing."""

    from vllm_omni.model_executor import persistent_state

    spec = make_generic_spec(persistent_state)
    config = _runner_config(spec, layout)
    runner = object.__new__(runner_cls)
    runner.vllm_config = config
    runner.device = "cpu"
    delegated: list[KVCacheConfig] = []

    from vllm_omni.worker import persistent_state as integration

    events: list[str] = []
    allocated: list[torch.Tensor] = []

    @contextmanager
    def allocation_scope():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    allocation_context = allocation_scope()

    def allocate(*args, **kwargs):
        assert events == ["enter"]
        assert args[2] is layout
        result = allocate_kv_cache(*args, **kwargs)
        allocated.append(result["persistent_state"])
        events.append("allocate")
        return result

    monkeypatch.setattr(integration, "allocate_kv_cache", allocate, raising=False)

    def initialize(
        self: Any,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        del self
        assert is_profiling is False
        assert kv_cache_allocation_context is None
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

    assert events == ["enter", "allocate", "exit"]
    assert len(allocated) == 1
    assert storage.raw.untyped_storage().data_ptr() == allocated[0].untyped_storage().data_ptr()
    storage.initialize_fresh_state_slot(1, generation=1)
    for index, descriptor in enumerate(spec.descriptors):
        view = storage.views[descriptor.name]
        assert view.untyped_storage().data_ptr() == storage.raw.untyped_storage().data_ptr()
        assert view.data_ptr() - storage.raw.data_ptr() == descriptor.offset_bytes
        assert view.dtype == descriptor.dtype
        assert tuple(view.shape) == (4, *descriptor.shape)
        assert view.stride(0) == spec.page_size_bytes // descriptor.element_size_bytes
        view[1].fill_(index + 3)
        raw_start = spec.page_size_bytes + descriptor.offset_bytes
        observed = storage.raw[raw_start : raw_start + descriptor.size_bytes].view(descriptor.dtype)
        assert torch.all(observed == index + 3)


@pytest.mark.parametrize(
    "runner_cls,base_cls",
    [
        (GPUARModelRunner, OmniGPUModelRunner),
        (GPUARModelRunnerV2, GPUModelRunnerV2),
    ],
)
def test_ordinary_cache_keeps_allocation_context_delegation(monkeypatch, runner_cls, base_cls):
    """@spec PORT-MIG-006: preserve ordinary core initialization ownership."""
    runner = object.__new__(runner_cls)
    context = nullcontext()
    config = KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])
    delegated = []

    def initialize(self, kv_cache_config, is_profiling=False, kv_cache_allocation_context=None):
        delegated.append((kv_cache_config, is_profiling, kv_cache_allocation_context))

    monkeypatch.setattr(base_cls, "initialize_kv_cache", initialize)
    runner.initialize_kv_cache(config, is_profiling=True, kv_cache_allocation_context=context)
    assert delegated == [(config, True, context)]
    assert runner._persistent_state_storage is None


@pytest.mark.parametrize(
    "runner_cls,base_cls",
    [
        (GPUARModelRunner, OmniGPUModelRunner),
        (GPUARModelRunnerV2, GPUModelRunnerV2),
    ],
)
def test_failed_native_allocation_exits_context_without_binding(monkeypatch, runner_cls, base_cls):
    """@spec PORT-STATE-002: failed allocation never binds partial storage."""
    from vllm_omni.model_executor import persistent_state
    from vllm_omni.worker import persistent_state as integration

    spec = make_generic_spec(persistent_state)
    runner = object.__new__(runner_cls)
    runner.vllm_config = _runner_config(spec)
    runner.device = "cpu"
    events = []

    @contextmanager
    def allocation_scope():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def allocate(*args, **kwargs):
        assert events == ["enter"]
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(base_cls, "initialize_kv_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(integration, "allocate_kv_cache", allocate, raising=False)
    with pytest.raises(RuntimeError, match="allocation failed"):
        runner.initialize_kv_cache(_state_config(spec), kv_cache_allocation_context=allocation_scope())
    assert events == ["enter", "exit"]
    assert getattr(runner, "_persistent_state_storage", None) is None
    layer = runner.vllm_config.compilation_config.static_forward_context["persistent_state"]
    with pytest.raises(RuntimeError, match="not bound"):
        _ = layer.persistent_state_storage


@pytest.mark.parametrize(
    "runner_cls,base_cls",
    [(GPUARModelRunner, OmniGPUModelRunner), (GPUARModelRunnerV2, GPUModelRunnerV2)],
)
@pytest.mark.parametrize("flag", ["enable_sleep_mode", "enable_cumem_allocator"])
def test_native_allocation_rejects_unqualified_memory_mode_before_context_or_binding(
    monkeypatch, runner_cls, base_cls, flag
):
    """@spec PORT-STATE-011, PORT-MIG-006: reject before entering either allocator path."""
    from vllm_omni.model_executor import persistent_state
    from vllm_omni.worker import persistent_state as integration

    spec = make_generic_spec(persistent_state)
    runner = object.__new__(runner_cls)
    runner.vllm_config = _runner_config(spec)
    runner.device = "cpu"
    setattr(runner.vllm_config.model_config, flag, True)
    calls = []

    @contextmanager
    def forbidden_context():
        calls.append("context")
        yield

    def forbidden_allocate(*args, **kwargs):
        calls.append("allocate")
        raise AssertionError("allocation must not run")

    def ordinary_initialize(self, kv_cache_config, is_profiling=False, kv_cache_allocation_context=None):
        # A delegated context would be entered by core even for an empty partition.
        if kv_cache_allocation_context is not None:
            with kv_cache_allocation_context:
                pass

    monkeypatch.setattr(base_cls, "initialize_kv_cache", ordinary_initialize)
    monkeypatch.setattr(integration, "allocate_kv_cache", forbidden_allocate, raising=False)
    with pytest.raises(ValueError, match=flag):
        runner.initialize_kv_cache(_state_config(spec), kv_cache_allocation_context=forbidden_context())
    assert calls == []
    assert getattr(runner, "_persistent_state_storage", None) is None
    layer = runner.vllm_config.compilation_config.static_forward_context["persistent_state"]
    with pytest.raises(RuntimeError, match="not bound"):
        _ = layer.persistent_state_storage


@pytest.mark.parametrize(
    "runner_cls,base_cls",
    [(GPUARModelRunner, OmniGPUModelRunner), (GPUARModelRunnerV2, GPUModelRunnerV2)],
)
@pytest.mark.parametrize("context", [None, nullcontext()], ids=["absent", "nullcontext"])
def test_native_allocation_accepts_disabled_modes_with_optional_context(monkeypatch, runner_cls, base_cls, context):
    """@spec PORT-STATE-002, PORT-MIG-006: absent and supplied null contexts are valid."""
    from vllm_omni.model_executor import persistent_state
    from vllm_omni.worker import persistent_state as integration

    spec = make_generic_spec(persistent_state)
    runner = object.__new__(runner_cls)
    runner.vllm_config = _runner_config(spec)
    runner.device = "cpu"
    allocated = []

    def allocate(*args, **kwargs):
        result = allocate_kv_cache(*args, **kwargs)
        allocated.append(result["persistent_state"])
        return result

    monkeypatch.setattr(base_cls, "initialize_kv_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(integration, "allocate_kv_cache", allocate, raising=False)
    runner.initialize_kv_cache(_state_config(spec), kv_cache_allocation_context=context)
    storage = runner._persistent_state_storage
    assert len(allocated) == 1, "the persistent backing must come from core allocation"
    assert storage.raw.numel() == 4 * spec.page_size_bytes
    assert storage.raw.untyped_storage().data_ptr() == allocated[0].untyped_storage().data_ptr()
    assert (
        runner.vllm_config.compilation_config.static_forward_context["persistent_state"].persistent_state_storage
        is storage
    )


@pytest.mark.parametrize(
    "runner_cls,base_cls",
    [(GPUARModelRunner, OmniGPUModelRunner), (GPUARModelRunnerV2, GPUModelRunnerV2)],
)
@pytest.mark.parametrize("invalid", ["byte_shape", "typed_view_alignment"])
def test_invalid_native_bytes_never_publish_or_fall_back(monkeypatch, runner_cls, base_cls, invalid):
    """@spec PORT-STATE-002, PORT-MIG-006: successful allocation is not binding publication."""
    from vllm_omni.model_executor import persistent_state
    from vllm_omni.worker import persistent_state as integration

    spec = make_generic_spec(persistent_state)
    runner = object.__new__(runner_cls)
    runner.vllm_config = _runner_config(spec)
    runner.device = "cpu"
    events = []
    fallback_calls = []
    unaligned = torch.empty(4 * spec.page_size_bytes + 1, dtype=torch.uint8)[1:]
    if invalid == "typed_view_alignment":
        # This is a real invalid dtype view, not a mocked formatter exception.
        with pytest.raises(RuntimeError, match="storage_offset"):
            persistent_state.persistent_state_storage_from_raw(spec, unaligned)

    @contextmanager
    def allocation_scope():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def allocate(*args, **kwargs):
        result = allocate_kv_cache(*args, **kwargs)
        raw = result["persistent_state"]
        if invalid == "byte_shape":
            result["persistent_state"] = raw.view(-1)
            assert result["persistent_state"].shape != raw.shape
        else:
            # Correct byte shape/extent/contiguity, but a real dtype view cannot
            # interpret fp32/int64 descriptors starting at byte offset one.
            result["persistent_state"] = unaligned.view(raw.shape)
            assert result["persistent_state"].is_contiguous()
            assert result["persistent_state"].storage_offset() == 1
        events.append("allocate")
        return result

    def independent_allocate(*args, **kwargs):
        fallback_calls.append("independent")
        return persistent_state.allocate_persistent_state_storage(*args, **kwargs)

    monkeypatch.setattr(base_cls, "initialize_kv_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(integration, "allocate_kv_cache", allocate, raising=False)
    monkeypatch.setattr(integration, "allocate_persistent_state_storage", independent_allocate, raising=False)
    with pytest.raises((ValueError, RuntimeError)):
        runner.initialize_kv_cache(_state_config(spec), kv_cache_allocation_context=allocation_scope())
    assert events == ["enter", "allocate", "exit"]
    assert fallback_calls == []
    assert getattr(runner, "_persistent_state_storage", None) is None
    layer = runner.vllm_config.compilation_config.static_forward_context["persistent_state"]
    with pytest.raises(RuntimeError, match="not bound"):
        _ = layer.persistent_state_storage
