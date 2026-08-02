# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident-slot management for model-defined persistent state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.request import Request

from .spec import PersistentStateSpec

logger = init_logger(__name__)


@dataclass(frozen=True)
class StateBinding:
    """Manager-issued logical identity for one resident physical slot."""

    request_id: str
    slot_id: int
    generation: int
    schema_id: str
    profile_id: str
    stage: int
    replica: int
    fresh: bool
    engine_epoch: str = "unbound"


class PersistentStateManager(SingleTypeKVCacheManager):
    """Exactly-one-slot manager with no token-cache semantics."""

    def __init__(
        self,
        kv_cache_spec: PersistentStateSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        max_admission_blocks_per_request: int | None = None,
        *,
        safety_reserve_slots: int = 0,
        max_resident_sessions: int | None = None,
        profile_id: str = "default",
        stage: int = 0,
        replica: int = 0,
        engine_epoch: str = "unbound",
    ) -> None:
        # @spec PORT-STATE-004, PORT-STATE-009, PORT-STATE-011
        if not isinstance(kv_cache_spec, PersistentStateSpec):
            raise TypeError("PersistentStateManager requires PersistentStateSpec")
        if enable_caching:
            raise ValueError("persistent state does not support prefix caching")
        if dcp_world_size != 1 or pcp_world_size != 1:
            raise ValueError("persistent state supports DCP=PCP=1 only")
        if max_admission_blocks_per_request not in (None, 1):
            raise ValueError("persistent state admits exactly one block per request")
        if safety_reserve_slots < 0:
            raise ValueError("persistent-state safety reserve cannot be negative")

        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            scheduler_block_size,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            max_admission_blocks_per_request=1,
        )
        self.persistent_state_spec = kv_cache_spec
        self.physical_capacity = block_pool.num_gpu_blocks - 1
        self.safety_reserve_slots = safety_reserve_slots
        usable_capacity = self.physical_capacity - safety_reserve_slots
        if usable_capacity <= 0:
            raise ValueError("persistent-state reserve leaves zero usable capacity")
        configured = usable_capacity if max_resident_sessions is None else max_resident_sessions
        if configured <= 0:
            raise ValueError("persistent-state configured capacity must be positive")
        self.max_resident_sessions = configured
        self.configured_limit = configured
        self.effective_capacity = min(usable_capacity, configured)
        if configured > usable_capacity:
            logger.warning(
                "Clamping persistent-state capacity from %d to %d slots",
                configured,
                usable_capacity,
            )

        self.profile_id = profile_id
        self.stage = stage
        self.replica = replica
        self.engine_epoch = engine_epoch
        self._bindings: dict[str, StateBinding] = {}
        self._terminal_request_ids: set[str] = set()
        self._used_request_ids: set[str] = set()
        self._next_generation = 1
        self._capacity_configured = False

    def configure_capacity(
        self,
        *,
        safety_reserve_slots: int,
        max_resident_sessions: int,
    ) -> None:
        """Apply the fingerprinted operator limits before first admission."""

        if safety_reserve_slots < 0:
            raise ValueError("persistent-state safety reserve cannot be negative")
        if max_resident_sessions <= 0:
            raise ValueError("persistent-state configured capacity must be positive")
        if self._capacity_configured:
            if (
                self.safety_reserve_slots != safety_reserve_slots
                or self.max_resident_sessions != max_resident_sessions
            ):
                raise RuntimeError(
                    "persistent-state capacity configuration changed"
                )
            return
        if self._bindings:
            raise RuntimeError(
                "persistent-state capacity cannot change after admission"
            )

        usable_capacity = self.physical_capacity - safety_reserve_slots
        if usable_capacity <= 0:
            raise ValueError("persistent-state reserve leaves zero usable capacity")
        self.safety_reserve_slots = safety_reserve_slots
        self.max_resident_sessions = max_resident_sessions
        self.configured_limit = max_resident_sessions
        self.effective_capacity = min(usable_capacity, max_resident_sessions)
        if max_resident_sessions > usable_capacity:
            logger.warning(
                "Clamping persistent-state capacity from %d to %d slots",
                max_resident_sessions,
                usable_capacity,
            )
        self._capacity_configured = True

    # @spec PORT-STATE-011
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del (
            num_tokens,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        if new_computed_blocks:
            raise ValueError("unsupported persistent-state computed-block import")
        return 0 if request_id in self._bindings else 1

    # @spec PORT-STATE-004, PORT-STATE-011
    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        del num_tokens, num_tokens_main_model
        if request_id in self._bindings:
            return []
        if request_id in self._used_request_ids:
            raise ValueError("persistent-state request identity cannot be reused")
        if len(self._bindings) >= self.effective_capacity:
            raise ValueError("persistent-state capacity exhausted")

        blocks = self.block_pool.get_new_blocks(1)
        block = blocks[0]
        if block.is_null:
            raise RuntimeError("null block cannot back live persistent state")
        self.req_to_blocks[request_id].append(block)
        binding = StateBinding(
            request_id=request_id,
            slot_id=block.block_id,
            generation=self._next_generation,
            schema_id=self.persistent_state_spec.schema_id,
            profile_id=self.profile_id,
            stage=self.stage,
            replica=self.replica,
            fresh=True,
            engine_epoch=self.engine_epoch,
        )
        self._next_generation += 1
        self._bindings[request_id] = binding
        self._used_request_ids.add(request_id)
        return blocks

    def get_state_binding(self, request_id: str) -> StateBinding | None:
        return self._bindings.get(request_id)

    def is_terminal(self, request_id: str) -> bool:
        """Whether scheduler terminality fenced this resident binding."""

        return request_id in self._terminal_request_ids

    def mark_initialized(
        self,
        request_id: str,
        generation: int,
    ) -> StateBinding:
        """Mark one exact generation initialized without changing ownership."""

        binding = self._bindings.get(request_id)
        if binding is None or binding.generation != generation:
            raise ValueError("stale persistent-state initialization acknowledgement")
        if binding.fresh:
            binding = replace(binding, fresh=False)
            self._bindings[request_id] = binding
        return binding

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        if request_id in self._bindings:
            self._terminal_request_ids.add(request_id)
            return []
        return super().pop_blocks_for_free(request_id)

    def free(self, request_id: str) -> None:
        # Scheduler terminality never owns physical persistent-state cleanup.
        if request_id in self._bindings:
            self._terminal_request_ids.add(request_id)
            return
        super().free(request_id)

    def mark_terminal(self, request_id: str) -> None:
        """Record scheduler terminality without returning the slot."""
        if request_id in self._bindings:
            self._terminal_request_ids.add(request_id)

    def drop_lease(self, request_id: str) -> None:
        """Return one API-released lease's slot exactly once."""
        if self._bindings.pop(request_id, None) is None:
            return
        self._terminal_request_ids.discard(request_id)
        super().free(request_id)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: Any,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        del (
            block_hashes,
            max_length,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size,
            pcp_world_size,
        )
        return tuple([] for _ in kv_cache_group_ids)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        del running_request_id
        return 0

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        del request, num_tokens, retention_interval

    def remove_skipped_blocks(
        self,
        request_id: str,
        total_computed_tokens: int | None = None,
        num_prompt_tokens: int | None = None,
        *,
        num_computed_tokens: int | None = None,
    ) -> None:
        del request_id, total_computed_tokens, num_prompt_tokens, num_computed_tokens

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        del request_id, num_local_computed_tokens, num_external_computed_tokens
        if new_computed_blocks:
            raise ValueError("unsupported persistent-state computed-block import")

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        del request_id, num_local_computed_tokens, num_external_computed_tokens
        raise ValueError("unsupported persistent-state external computed blocks")
