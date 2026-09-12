# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real core layout and manager contracts; no accelerator is required."""

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    compute_layer_kv_cache_shape_bytes,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

from tests.model_executor.persistent_state._helpers import make_generic_spec
from vllm_omni.model_executor import persistent_state
from vllm_omni.worker.persistent_state import partition_persistent_state_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _config():
    spec = make_generic_spec(persistent_state)
    tensor = KVCacheTensor(size=256, layers=["state"], layer_stride=256, block_stride=64)
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["state"], kv_cache_spec=spec)],
        prefix_cache_retention_interval=17,
        kv_cache_layout="LBHNC",
    )


def test_fixed_state_uses_core_layout_without_a_token_history():
    """@spec PORT-STATE-002, PORT-STATE-011."""
    spec = make_generic_spec(persistent_state)
    assert not spec.prefix_cacheable
    assert spec.tokens_per_state == -1
    assert spec.num_states == 1
    assert spec.max_num_blocks_per_req(None, 1_000_000) == 1
    assert compute_layer_kv_cache_shape_bytes(spec, 4) == (4, 1, 1, 64)
    for layout in KVCacheLayout:
        strides = compute_layout_strides(spec, 4, 1, layout)
        assert strides[1] == spec.page_size_bytes
        assert strides[-1] == 1


def test_partition_preserves_core_config_metadata_and_original():
    config = _config()
    original_tensor = config.kv_cache_tensors[0]
    result = partition_persistent_state_config(config)
    assert result.state_tensor is original_tensor
    assert result.ordinary_config.kv_cache_groups == []
    assert result.ordinary_config.kv_cache_tensors == []
    assert result.ordinary_config.prefix_cache_retention_interval == 17
    assert result.ordinary_config.kv_cache_layout == "LBHNC"
    assert config.kv_cache_tensors == [original_tensor]


@pytest.mark.parametrize(
    "changes",
    [
        {"offset": 64},
        {"block_stride": 128},
        {"size": 512},
        {"layers": ["state", "state"]},
        {"layers": ["state", "attention"]},
    ],
)
def test_partition_rejects_noncanonical_state_placement(changes):
    config = _config()
    config.kv_cache_tensors[0] = replace(config.kv_cache_tensors[0], **changes)
    with pytest.raises(ValueError):
        partition_persistent_state_config(config)


def test_manager_accepts_core_zeroing_and_local_compute_arguments():
    """@spec PORT-STATE-004, PORT-STATE-011: one slot and exact release."""
    spec = make_generic_spec(persistent_state)
    pool = BlockPool(num_gpu_blocks=4, enable_caching=False, hash_block_size=1)
    manager = persistent_state.PersistentStateManager(
        spec,
        pool,
        False,
        0,
        1,
        needs_kv_cache_zeroing=True,
    )
    assert not manager.records_new_block_ids
    kwargs = dict(
        request_id="a",
        num_tokens=1000,
        new_computed_blocks=[],
        total_computed_tokens=900,
        num_local_computed_tokens=900,
        num_tokens_main_model=1000,
    )
    assert manager.get_num_blocks_to_allocate(**kwargs) == 1
    blocks = manager.allocate_new_blocks("a", 1000, 1000)
    first = manager.get_state_binding("a")
    assert len(blocks) == 1
    assert manager.get_num_blocks_to_allocate(**kwargs) == 0
    assert manager.take_new_block_ids() == []
    manager.free("a")
    assert manager.get_state_binding("a") == first
    manager.drop_lease("a")
    manager.allocate_new_blocks("b", 1, 1)
    assert manager.get_state_binding("b").generation > first.generation


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_core_constructed_pool_has_exact_aggregate_byte_placement(layout):
    """@spec PORT-STATE-002: exercise the core allocation constructor itself."""
    config = _config()
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: layout,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        )
    )
    built = get_kv_cache_config_from_groups(
        vllm_config,
        config.kv_cache_groups,
        available_memory=4 * 64,
    )
    result = partition_persistent_state_config(built)
    assert result.state_tensor.size == 256
    assert result.state_tensor.offset == 0
    assert result.state_tensor.block_stride == 64
    assert result.state_tensor.layers == ["state"]
    assert result.ordinary_config.num_blocks == 4
    views = allocate_kv_cache(built, "cpu", layout)
    state = views["state"]
    assert tuple(state.shape) == (4, 1, 1, 64)
    assert state.stride(0) == 64
    assert state.untyped_storage().nbytes() == 256


def test_manager_overrides_accept_core_keyword_contracts():
    for name in (
        "get_num_blocks_to_allocate",
        "allocate_new_blocks",
        "find_longest_cache_hit",
        "remove_skipped_blocks",
        "cache_blocks",
        "add_local_computed_blocks",
        "allocate_external_computed_blocks",
    ):
        core = inspect.signature(getattr(SingleTypeKVCacheManager, name))
        override = inspect.signature(getattr(persistent_state.PersistentStateManager, name))
        override.bind(**{name: None for name in core.parameters})


def test_partition_rejects_mixed_groups_sharing_core_backing_allocation():
    config = _config()
    # Ordinary-group type is irrelevant: no second group may share state bytes.
    config.kv_cache_groups.append(KVCacheGroupSpec(layer_names=["other"], kv_cache_spec=KVCacheSpec(block_size=1)))
    with pytest.raises(ValueError, match="persistent-only"):
        partition_persistent_state_config(config)
