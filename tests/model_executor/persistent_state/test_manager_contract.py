# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mechanical contracts for the resident-only persistent-state manager."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from tests.model_executor.persistent_state._helpers import (
    make_manager,
    real_request,
    require_persistent_state_module,
    state_binding_pair,
)


def test_manager_allocates_one_block_independent_of_token_count() -> None:
    """@spec PORT-STATE-011: admission is one group block, not token-sized."""

    module = require_persistent_state_module()
    for token_count in (0, 1, 17, 10_000):
        manager, _, _ = make_manager(module)
        assert (
            manager.get_num_blocks_to_allocate(
                "request-1",
                token_count,
                [],
                token_count,
                token_count,
            )
            == 1
        )
        new_blocks = manager.allocate_new_blocks(
            "request-1", token_count, token_count
        )
        assert len(new_blocks) == 1
        assert len(manager.req_to_blocks["request-1"]) == 1


def test_continuations_retain_binding_and_free_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-STATE-011: continuations retain one binding and free once."""

    module = require_persistent_state_module()
    manager, block_pool, _ = make_manager(module)
    manager.allocate_new_blocks("request-1", 1, 1)
    first_binding = state_binding_pair(manager, "request-1")
    first_block_ids = tuple(block.block_id for block in manager.req_to_blocks["request-1"])

    for token_count in (1, 8, 1000):
        assert (
            manager.get_num_blocks_to_allocate(
                "request-1", token_count, [], token_count, token_count
            )
            == 0
        )
        assert manager.allocate_new_blocks("request-1", token_count, token_count) == []
        assert tuple(block.block_id for block in manager.req_to_blocks["request-1"]) == first_block_ids
        assert state_binding_pair(manager, "request-1") == first_binding

    freed: list[tuple[int, ...]] = []
    original_free_blocks = block_pool.free_blocks

    def record_free(blocks: Iterable[Any]) -> None:
        materialized_blocks = tuple(blocks)
        materialized = tuple(block.block_id for block in materialized_blocks)
        if materialized:
            freed.append(materialized)
        original_free_blocks(materialized_blocks)

    monkeypatch.setattr(block_pool, "free_blocks", record_free)
    manager.free("request-1")
    manager.free("request-1")
    assert freed == []

    manager.drop_lease("request-1")
    manager.drop_lease("request-1")

    assert len(freed) == 1


def test_reused_physical_slot_gets_a_fresh_generation() -> None:
    """@spec PORT-STATE-011: slot reuse cannot retain the old generation."""

    module = require_persistent_state_module()
    manager, _, _ = make_manager(module, num_gpu_blocks=2)
    manager.allocate_new_blocks("request-1", 1, 1)
    first_binding = state_binding_pair(manager, "request-1")
    manager.free("request-1")
    manager.drop_lease("request-1")

    manager.allocate_new_blocks("request-2", 10_000, 10_000)
    second_binding = state_binding_pair(manager, "request-2")

    assert second_binding[0] == first_binding[0]
    assert second_binding[1] != first_binding[1]


def test_manager_has_no_prefix_or_skipped_block_semantics() -> None:
    """@spec PORT-STATE-011: token-cache operations are inert for resident state."""

    module = require_persistent_state_module()
    manager, block_pool, _ = make_manager(module)
    manager.allocate_new_blocks("request-1", 1, 1)
    request = real_request()
    before_blocks = tuple(block.block_id for block in manager.req_to_blocks["request-1"])
    before_cache_map = len(block_pool.cached_block_hash_to_block)
    before_cached_counts = dict(manager.num_cached_block)

    cache_hits = type(manager).find_longest_cache_hit(
        [],
        max_length=1000,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=manager.kv_cache_spec,
        drop_eagle_block=False,
        alignment_tokens=1,
    )
    assert not any(cache_hits)
    assert manager.get_num_common_prefix_blocks("request-1") == 0

    manager.cache_blocks(request, num_tokens=10_000)
    manager.remove_skipped_blocks("request-1", num_computed_tokens=10_000)

    assert tuple(block.block_id for block in manager.req_to_blocks["request-1"]) == before_blocks
    assert len(block_pool.cached_block_hash_to_block) == before_cache_map
    assert manager.num_cached_block == before_cached_counts
    assert all(block.block_hash is None for block in manager.req_to_blocks["request-1"])


def test_external_computed_blocks_are_rejected_before_mutation() -> None:
    """@spec PORT-STATE-011: resident-only state rejects core external imports."""

    module = require_persistent_state_module()
    manager, _, _ = make_manager(module)
    manager.allocate_new_blocks("request-1", 1, 1)
    before = tuple(block.block_id for block in manager.req_to_blocks["request-1"])

    with pytest.raises(Exception) as exc_info:
        manager.allocate_external_computed_blocks(
            "request-1",
            num_local_computed_tokens=0,
            num_external_computed_tokens=1,
        )

    assert "unsupported" in str(exc_info.value).lower()
    assert tuple(block.block_id for block in manager.req_to_blocks["request-1"]) == before
