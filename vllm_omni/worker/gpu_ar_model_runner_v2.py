# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRv2 AR runner substrate for Omni-owned persistent state."""

from __future__ import annotations

from typing import Any, cast

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_omni.worker.persistent_state import (
    allocate_runner_persistent_state,
    build_persistent_state_batch,
    discover_persistent_state_specs,
    partition_persistent_state_config,
)


class GPUARModelRunnerV2(GPUModelRunner):
    """Thin core-V2 subclass; lifecycle projection lands in the P5 slice."""

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Add the purpose-named group outside core attention discovery."""

        from vllm_omni.model_executor.persistent_state import PersistentStateSpec

        specs = cast(
            dict[str, KVCacheSpec],
            super().get_kv_cache_spec(),  # type: ignore[no-untyped-call]
        )
        persistent_specs = discover_persistent_state_specs(self.vllm_config)
        if any(not isinstance(spec, PersistentStateSpec) for spec in persistent_specs.values()):
            raise TypeError("persistent-state discovery returned an unsupported spec")
        if set(specs) & set(persistent_specs):
            raise RuntimeError("persistent-state layer collides with core cache discovery")
        specs.update(persistent_specs)
        return specs

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Keep the state group out of core attention backend initialization."""

        from vllm_omni.model_executor.persistent_state import PersistentStateSpec

        persistent_groups = [
            group for group in kv_cache_config.kv_cache_groups if isinstance(group.kv_cache_spec, PersistentStateSpec)
        ]
        partition = partition_persistent_state_config(kv_cache_config)
        if len(persistent_groups) > 1:
            raise ValueError("persistent-state profile permits exactly one state group")
        self._omni_full_kv_cache_config = kv_cache_config
        self._persistent_state_group_index = (
            kv_cache_config.kv_cache_groups.index(partition.state_group)
            if partition.state_group is not None
            else None
        )
        super().initialize_kv_cache(partition.ordinary_config)
        self._persistent_state_storage = allocate_runner_persistent_state(
            self,
            partition,
        )

    def _ordinary_block_ids(
        self,
        block_ids: tuple[list[int], ...],
    ) -> tuple[list[int], ...]:
        group_index = self._persistent_state_group_index
        if group_index is None:
            return block_ids
        if group_index >= len(block_ids):
            raise RuntimeError("scheduler block groups omit persistent state")
        return tuple(
            group
            for index, group in enumerate(block_ids)
            if index != group_index
        )

    def add_requests(self, scheduler_output: Any) -> None:
        """Keep the custom group out of core V2's ordinary block tables."""

        original = [data.block_ids for data in scheduler_output.scheduled_new_reqs]
        try:
            for data in scheduler_output.scheduled_new_reqs:
                data.block_ids = self._ordinary_block_ids(data.block_ids)
            super().add_requests(scheduler_output)
        finally:
            for data, block_ids in zip(
                scheduler_output.scheduled_new_reqs,
                original,
            ):
                data.block_ids = block_ids

    def _build_persistent_state_batch(self, rows: Any) -> Any:
        """Use the same model-neutral row projection as the MRV1 oracle."""

        return build_persistent_state_batch(rows)

    def finish_requests(self, scheduler_output: Any) -> None:
        """Preserve the scheduler's distinct terminal/preemption authorities."""
        self._omni_finished_req_ids = frozenset(
            scheduler_output.finished_req_ids
        )
        self._omni_preempted_req_ids = frozenset(
            scheduler_output.preempted_req_ids
        )
        super().finish_requests(scheduler_output)

    def update_requests(self, scheduler_output: Any) -> None:
        """Reconcile only after core has rebuilt the complete resident map."""
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        original = None if cached is None else cached.new_block_ids
        try:
            if cached is not None:
                cached.new_block_ids = [
                    None
                    if block_ids is None
                    else self._ordinary_block_ids(block_ids)
                    for block_ids in original
                ]
            super().update_requests(scheduler_output)
        finally:
            if cached is not None:
                cached.new_block_ids = original
        self.model_state.reconcile_omni_lifecycle(
            finished_req_ids=getattr(
                self, "_omni_finished_req_ids", frozenset()
            ),
            preempted_req_ids=getattr(
                self, "_omni_preempted_req_ids", frozenset()
            ),
            resident_req_ids=frozenset(self.req_states.req_id_to_index),
        )

    def execute_model(
        self,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
        *,
        skip_attn_for_dummy_run: bool = False,
        dummy_run: bool = False,
        is_profile: bool = False,
    ) -> Any:
        """Bracket the selected core runner without replacing its mechanics."""
        self.model_state.begin_omni_projection(
            scheduler_output,
            dummy_run=dummy_run,
            is_profile=is_profile,
        )
        try:
            result = super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                dummy_run=dummy_run,
                is_profile=is_profile,
            )
            commit = getattr(self.model_state, "commit_omni_projection", None)
            if not dummy_run and commit is not None:
                commit()
            return result
        finally:
            self.model_state.end_omni_projection()
            self._omni_finished_req_ids = frozenset()
            self._omni_preempted_req_ids = frozenset()
