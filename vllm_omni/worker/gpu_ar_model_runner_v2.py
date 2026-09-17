# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 AR runner substrate for Omni-owned persistent state."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, cast

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_omni.metrics import streaming_transport
from vllm_omni.outputs import OmniConnectorOutput
from vllm_omni.worker.persistent_state import (
    allocate_runner_persistent_state,
    build_persistent_state_batch,
    discover_persistent_state_specs,
    partition_persistent_state_config,
)


class _OmniProjectionState(Protocol):
    """Model-owned projection hooks used by the persistent-state runner."""

    def begin_omni_projection(self, scheduler_output: Any, *, dummy_run: bool, is_profile: bool) -> None: ...

    def end_omni_projection(self) -> None: ...

    def reconcile_omni_lifecycle(
        self, *, finished_req_ids: frozenset[str], preempted_req_ids: frozenset[str], resident_req_ids: frozenset[str]
    ) -> None: ...


class GPUARModelRunnerV2(GPUModelRunner):
    """Thin core-V2 subclass; lifecycle projection lands in the P5 slice."""

    def _dummy_sampler_run(self, hidden_states: Any) -> None:
        handoff = getattr(self.model, "_native_burst_handoff", None)
        if handoff is None:
            super()._dummy_sampler_run(hidden_states)
            return
        with handoff.dummy_sampling(park_id=int(self.model.config.eos_token_id)):
            super()._dummy_sampler_run(hidden_states)

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

    # @spec PORT-MIG-006
    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        """Keep the state group out of core attention backend initialization."""

        from vllm_omni.model_executor.persistent_state import PersistentStateSpec

        persistent_groups = [
            group for group in kv_cache_config.kv_cache_groups if isinstance(group.kv_cache_spec, PersistentStateSpec)
        ]
        partition = partition_persistent_state_config(kv_cache_config)
        if len(persistent_groups) > 1:
            raise ValueError("persistent-state profile permits exactly one state group")
        self._omni_full_kv_cache_config = kv_cache_config
        self._ordinary_kv_cache_group_count = len(partition.ordinary_config.kv_cache_groups)
        self._persistent_state_group_index = (
            kv_cache_config.kv_cache_groups.index(partition.state_group) if partition.state_group is not None else None
        )
        super().initialize_kv_cache(
            partition.ordinary_config,
            is_profiling=is_profiling,
            kv_cache_allocation_context=(kv_cache_allocation_context if partition.state_group is None else None),
        )
        self._persistent_state_storage = allocate_runner_persistent_state(
            self,
            partition,
            kv_cache_allocation_context=kv_cache_allocation_context,
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
        ordinary = tuple(group for index, group in enumerate(block_ids) if index != group_index)
        if len(ordinary) != self._ordinary_kv_cache_group_count:
            raise RuntimeError("ordinary block ids do not match initialized cache groups")
        return ordinary

    def add_requests(self, scheduler_output: Any) -> None:
        """Keep the custom group out of core V2's ordinary block tables."""

        validate_request = getattr(self.model_state, "validate_omni_request", None)
        if validate_request is not None:
            for data in scheduler_output.scheduled_new_reqs:
                validate_request(data)
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
        self._omni_finished_req_ids = frozenset(scheduler_output.finished_req_ids)
        self._omni_preempted_req_ids = frozenset(scheduler_output.preempted_req_ids)
        super().finish_requests(scheduler_output)

    def update_requests(self, scheduler_output: Any) -> None:
        """Reconcile only after core has rebuilt the complete resident map."""
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        original = None if cached is None else cached.new_block_ids
        try:
            if cached is not None:
                assert original is not None
                cached.new_block_ids = [
                    None if block_ids is None else self._ordinary_block_ids(block_ids) for block_ids in original
                ]
            super().update_requests(scheduler_output)
        finally:
            if cached is not None:
                cached.new_block_ids = original
        cast(_OmniProjectionState, self.model_state).reconcile_omni_lifecycle(
            finished_req_ids=getattr(self, "_omni_finished_req_ids", frozenset()),
            preempted_req_ids=getattr(self, "_omni_preempted_req_ids", frozenset()),
            resident_req_ids=frozenset(self.req_states.req_id_to_index),
        )

    def execute_model(
        self,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
    ) -> Any:
        """Bracket the selected core runner without replacing its mechanics."""
        if is_profile and not dummy_run:
            raise ValueError("persistent-state profile execution requires a dummy run")
        projection_state = cast(_OmniProjectionState, self.model_state)
        projection_state.begin_omni_projection(
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
                context_len=context_len,
            )
            commit = getattr(self.model_state, "commit_omni_projection", None)
            if not dummy_run and commit is not None:
                commit()
            return result
        finally:
            projection_state.end_omni_projection()
            self._omni_finished_req_ids = frozenset()
            self._omni_preempted_req_ids = frozenset()

    # @spec PORT-OBS-008, PORT-OBS-009
    def sample_tokens(self, grammar_output: Any) -> Any:
        """Forward committed model status and batch stats at the boundary."""

        output = super().sample_tokens(grammar_output)
        model_runner_output = getattr(output, "model_runner_output", output)
        consume_batch_stats = getattr(
            self.model,
            "consume_batch_stats",
            None,
        )
        if model_runner_output is not None and callable(consume_batch_stats):
            streaming_transport.drain_batch_stats_into_runner_output(
                self.model,
                model_runner_output,
            )
        collect = getattr(self.model, "collect_commit_status", None)
        if not callable(collect):
            return output
        model_status, failed_req_ids = collect()
        if not model_status and not failed_req_ids:
            return output

        if model_runner_output is None:
            raise RuntimeError("model transaction status has no model-runner output carrier")
        connector_output = getattr(
            model_runner_output,
            "omni_connector_output",
            None,
        )
        if connector_output is None:
            connector_output = OmniConnectorOutput()
        elif not isinstance(connector_output, OmniConnectorOutput):
            raise TypeError("model-runner output carries an invalid Omni result")
        if connector_output.model_status or connector_output.model_failed_req_ids:
            raise RuntimeError("model transaction status was attached twice")
        connector_output.model_status = model_status
        connector_output.model_failed_req_ids = failed_req_ids
        model_runner_output.omni_connector_output = connector_output
        return output

    # @spec PORT-INT-007, PORT-STATE-002
    def profile_run(self) -> None:
        """Reject prefix caching before persistent-state profiling."""

        persistent_specs = discover_persistent_state_specs(self.vllm_config)
        if persistent_specs and self.cache_config.enable_prefix_caching:
            raise ValueError("prefix caching is incompatible with persistent state")
        super().profile_run()
        prepare = getattr(getattr(self, "model", None), "prepare_execution_memory", None)
        if callable(prepare):
            import torch

            # The core dummy forward has returned, while the worker still
            # holds its memory-profiling window open.
            with torch.inference_mode():
                prepare()
