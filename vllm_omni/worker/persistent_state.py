# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-neutral integration helpers for model-defined persistent state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_omni.model_executor.persistent_state import (
    PersistentStateBatch,
    PersistentStateLayerBase,
    PersistentStateSpec,
    PersistentStateStorage,
    allocate_persistent_state_storage,
    persistent_state_storage_from_raw,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class PersistentStatePartition:
    """Ordinary core configuration plus one purpose-named state group."""

    ordinary_config: KVCacheConfig
    state_group: KVCacheGroupSpec | None
    state_tensor: KVCacheTensor | None


def resolve_persistent_state_available_memory(
    vllm_config: Any,
    *,
    available_memory_bytes: int,
    cache_specs: Mapping[str, Any],
) -> int:
    """Cap a persistent-only pool from the real post-profile memory bound.

    vLLM consumes the returned byte count when it constructs the cache
    configuration.  Therefore the physical bound remains independent of an
    operator count, while a smaller count can still return memory to the
    deployment.  An explicit block override is validated against, never used
    to manufacture, the post-profile bound.
    """

    persistent_specs = discover_persistent_state_specs(vllm_config)
    if not persistent_specs:
        return available_memory_bytes
    if len(persistent_specs) != 1:
        raise ValueError("persistent-state allocation requires one aggregate spec")
    if set(cache_specs) != set(persistent_specs):
        raise ValueError("persistent-state byte allocation requires a persistent-only cache layout")
    spec = next(iter(persistent_specs.values()))
    from vllm_omni.engine.persistent_state_capacity import (
        resolve_pool_capacity,
    )
    from vllm_omni.engine.persistent_state_config import (
        PersistentStateRuntimeConfig,
    )

    runtime = PersistentStateRuntimeConfig.from_vllm_config(vllm_config)
    profiled_block_bound = available_memory_bytes // spec.page_size_bytes
    resolution = resolve_pool_capacity(
        profiled_block_bound=profiled_block_bound,
        safety_reserve_slots=runtime.safety_reserve_slots,
        max_resident_sessions=runtime.max_resident_sessions,
        num_gpu_blocks_override=getattr(
            vllm_config.cache_config,
            "num_gpu_blocks_override",
            None,
        ),
        page_size_bytes=spec.page_size_bytes,
    )
    if resolution.count_was_clamped:
        logger.warning(
            "Clamping requested persistent-state capacity from %d to %d slots against the post-profile physical bound",
            resolution.requested_count_limit,
            resolution.resolved_count_limit,
        )
    return resolution.allocated_total_blocks * int(spec.page_size_bytes)


def discover_persistent_state_specs(vllm_config: Any) -> dict[str, PersistentStateSpec]:
    """Discover purpose-named state layers outside attention/Mamba dispatch."""

    layers = get_layers_from_vllm_config(vllm_config, PersistentStateLayerBase)
    discovered: dict[str, PersistentStateSpec] = {}
    for layer_name, layer in layers.items():
        spec = layer.get_kv_cache_spec(vllm_config)
        if not isinstance(spec, PersistentStateSpec):
            raise TypeError("persistent-state layer returned an unsupported spec")
        discovered[layer_name] = spec
    return discovered


def partition_persistent_state_config(
    kv_cache_config: KVCacheConfig,
) -> PersistentStatePartition:
    """Remove the one state group before core attention initialization."""

    state_groups = [
        group for group in kv_cache_config.kv_cache_groups if isinstance(group.kv_cache_spec, PersistentStateSpec)
    ]
    if len(state_groups) > 1:
        raise ValueError("persistent-state profile permits exactly one state group")
    if not state_groups:
        return PersistentStatePartition(kv_cache_config, None, None)

    state_group = state_groups[0]
    if len(state_group.layer_names) != 1:
        raise ValueError("persistent-state group must contain exactly one aggregate layer")
    state_layer_names = set(state_group.layer_names)
    ordinary_groups = [group for group in kv_cache_config.kv_cache_groups if group is not state_group]
    ordinary_layer_names = {layer_name for group in ordinary_groups for layer_name in group.layer_names}

    state_tensors: list[KVCacheTensor] = []
    ordinary_tensors: list[KVCacheTensor] = []
    for tensor in kv_cache_config.kv_cache_tensors:
        shared_by = set(tensor.shared_by)
        if shared_by & state_layer_names:
            if shared_by - state_layer_names:
                raise ValueError("persistent-state and ordinary layers cannot share an allocation")
            state_tensors.append(tensor)
        else:
            if not shared_by <= ordinary_layer_names:
                raise ValueError("KV-cache tensor is not owned by a declared group")
            ordinary_tensors.append(tensor)
    if len(state_tensors) != 1:
        raise ValueError("persistent-state group requires one aggregate allocation")

    ordinary_config = KVCacheConfig(
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_tensors=ordinary_tensors,
        kv_cache_groups=ordinary_groups,
    )
    return PersistentStatePartition(
        ordinary_config=ordinary_config,
        state_group=state_group,
        state_tensor=state_tensors[0],
    )


def allocate_runner_persistent_state(
    runner: Any,
    partition: PersistentStatePartition,
) -> PersistentStateStorage | None:
    """Allocate and bind the state group after ordinary core initialization."""

    if partition.state_group is None:
        return None
    assert partition.state_tensor is not None
    spec = partition.state_group.kv_cache_spec
    assert isinstance(spec, PersistentStateSpec)
    expected_size = partition.ordinary_config.num_blocks * spec.page_size_bytes
    if partition.state_tensor.size != expected_size:
        raise ValueError("persistent-state tensor size disagrees with slot geometry")
    storage = allocate_persistent_state_storage(
        spec,
        partition.ordinary_config.num_blocks,
        runner.device,
    )
    layers = get_layers_from_vllm_config(
        runner.vllm_config,
        PersistentStateLayerBase,
        partition.state_group.layer_names,
    )
    if set(layers) != set(partition.state_group.layer_names):
        raise RuntimeError("persistent-state layer disappeared before allocation")
    for layer in layers.values():
        layer.bind_persistent_state_storage(storage)
    return storage


def reshape_runner_persistent_state(
    spec: PersistentStateSpec,
    raw: torch.Tensor,
) -> PersistentStateStorage:
    """Create typed state views for a core-provided raw byte allocation."""

    return persistent_state_storage_from_raw(spec, raw)


def build_persistent_state_batch(rows: Any) -> PersistentStateBatch:
    """Build the common row projection without acquiring physical lifetime."""

    if isinstance(rows, PersistentStateBatch):
        return rows
    return PersistentStateBatch(rows)
