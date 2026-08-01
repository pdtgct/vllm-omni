# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mechanical contracts for the aggregate persistent-state specification."""

from __future__ import annotations

import pytest
import torch
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.core.kv_cache_utils import (
    create_kv_cache_group_specs,
    is_kv_cache_type_attention_free,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

from tests.model_executor.persistent_state._helpers import (
    descriptor_field,
    make_generic_spec,
    require_persistent_state_module,
    require_symbol,
    spec_field,
)


def test_persistent_spec_is_direct_immutable_core_kv_spec() -> None:
    """@spec PORT-STATE-002: the aggregate is a direct immutable KV spec."""

    module = require_persistent_state_module()
    spec_cls = require_symbol(module, "PersistentStateSpec")
    spec = make_generic_spec(module)

    assert issubclass(spec_cls, KVCacheSpec)
    assert not issubclass(spec_cls, AttentionSpec)
    assert not issubclass(spec_cls, MambaSpec)
    assert spec.block_size == 1

    descriptors = spec_field(spec, "descriptors", "tensor_descriptors", "layout")
    assert isinstance(descriptors, tuple)
    assert len(descriptors) == 2
    for descriptor in descriptors:
        shape = tuple(descriptor_field(descriptor, "shape"))
        dtype = descriptor_field(descriptor, "dtype")
        offset = descriptor_field(descriptor, "offset_bytes", "byte_offset", "offset")
        alignment = descriptor_field(descriptor, "alignment_bytes", "alignment")
        initializer = descriptor_field(
            descriptor, "initializer", "fresh_initializer", "fresh_init"
        )
        assert shape and all(isinstance(dimension, int) and dimension > 0 for dimension in shape)
        assert isinstance(dtype, torch.dtype)
        assert isinstance(offset, int) and offset >= 0
        assert isinstance(alignment, int) and alignment > 0
        assert offset % alignment == 0
        assert initializer is not None

    assert spec.page_size_bytes == 64
    assert spec.page_size_bytes % max(
        descriptor_field(item, "alignment_bytes", "alignment") for item in descriptors
    ) == 0
    # KVCacheSpec requires this method to accept the core config parameter; a
    # persistent slot must not inspect max_model_len to size itself.
    assert spec.max_memory_usage_bytes(None) == spec.page_size_bytes

    with pytest.raises((AttributeError, TypeError, RuntimeError)):
        spec.block_size = 2

    with pytest.raises((AttributeError, TypeError, RuntimeError)):
        descriptors[0].shape = (99,)

    try:
        copied = spec.copy_with_new_block_size(2)
    except (AttributeError, TypeError, RuntimeError, ValueError, AssertionError):
        pass
    else:
        assert copied.block_size == 1


def test_spec_schema_fingerprint_covers_complete_generic_layout() -> None:
    """@spec PORT-STATE-009: schema identity follows the complete layout."""

    module = require_persistent_state_module()
    base = make_generic_spec(module)
    same_layout = make_generic_spec(module)
    shifted = make_generic_spec(module, variant="shifted")
    reshaped = make_generic_spec(module, variant="reshaped")

    schema_name = ("schema_id", "state_schema_id", "schema")
    base_schema = spec_field(base, *schema_name)
    assert base_schema == spec_field(same_layout, *schema_name)
    assert base_schema != spec_field(shifted, *schema_name)
    assert base_schema != spec_field(reshaped, *schema_name)

    # A schema is layout-derived, not a request or mutable-content identity.
    assert all(
        not hasattr(base, request_field)
        for request_field in ("request_id", "session_id", "generation", "slot_id")
    )


def test_core_one_entry_group_merge_keeps_concrete_persistent_spec() -> None:
    """@spec PORT-STATE-002: core grouping does not wrap one state entry."""

    module = require_persistent_state_module()
    spec = make_generic_spec(module)
    groups = create_kv_cache_group_specs({"persistent_state": spec}, [["persistent_state"]])

    assert not is_kv_cache_type_attention_free({"persistent_state": spec})
    assert len(groups) == 1
    assert groups[0].layer_names == ["persistent_state"]
    assert type(groups[0].kv_cache_spec) is type(spec)
    assert groups[0].kv_cache_spec is not spec
    assert groups[0].kv_cache_spec == spec


def test_persistent_layer_base_is_not_attention_or_mamba() -> None:
    """@spec PORT-STATE-002: the state layer has no false core type identity."""

    module = require_persistent_state_module()
    layer_base = require_symbol(module, "PersistentStateLayerBase")

    assert not issubclass(layer_base, AttentionLayerBase)
    assert not issubclass(layer_base, MambaBase)
