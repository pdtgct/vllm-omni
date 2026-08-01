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
        super().initialize_kv_cache(partition.ordinary_config)
        self._persistent_state_storage = allocate_runner_persistent_state(
            self,
            partition,
        )

    def _build_persistent_state_batch(self, rows: Any) -> Any:
        """Use the same model-neutral row projection as the MRV1 oracle."""

        return build_persistent_state_batch(rows)
