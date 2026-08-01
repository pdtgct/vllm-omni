# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative layout types for model-defined persistent state."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.v1.kv_cache_interface import KVCacheSpec

StateInitializer = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class PersistentStateDescriptor:
    """One typed component inside an aggregate persistent-state page."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset_bytes: int
    alignment_bytes: int
    initializer: StateInitializer

    def __post_init__(self) -> None:
        # @spec PORT-STATE-002, PORT-STATE-009
        if not self.name:
            raise ValueError("persistent-state descriptor name must be nonempty")
        if not self.shape or any(not isinstance(dimension, int) or dimension <= 0 for dimension in self.shape):
            raise ValueError(f"invalid persistent-state shape for {self.name!r}")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"invalid persistent-state dtype for {self.name!r}")
        if self.offset_bytes < 0:
            raise ValueError(f"negative persistent-state offset for {self.name!r}")
        if self.alignment_bytes <= 0:
            raise ValueError(f"invalid persistent-state alignment for {self.name!r}")
        if self.offset_bytes % self.alignment_bytes:
            raise ValueError(f"misaligned persistent-state offset for {self.name!r}")
        if self.offset_bytes % self.element_size_bytes:
            raise ValueError(f"dtype-misaligned persistent-state offset for {self.name!r}")
        if not callable(self.initializer):
            raise TypeError(f"persistent-state initializer for {self.name!r} must be callable")

    @property
    def element_size_bytes(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def size_bytes(self) -> int:
        return math.prod(self.shape) * self.element_size_bytes


@dataclass(frozen=True, init=False)
class PersistentStateSpec(KVCacheSpec):
    """One immutable allocation record for an atomic persistent-state bundle."""

    descriptors: tuple[PersistentStateDescriptor, ...]
    state_name: str
    persistence_class: str
    schema_id: str = field(compare=True)
    _page_size_bytes: int = field(compare=True, repr=False)

    def __init__(
        self,
        *,
        descriptors: Sequence[PersistentStateDescriptor],
        page_size_bytes: int,
        block_size: int = 1,
        state_name: str,
        persistence_class: str = "resident",
    ) -> None:
        # @spec PORT-STATE-002, PORT-STATE-009
        descriptor_tuple = tuple(descriptors)
        if block_size != 1:
            raise ValueError("persistent state requires block_size=1")
        if not state_name:
            raise ValueError("persistent-state name must be nonempty")
        if persistence_class != "resident":
            raise ValueError("initial persistent-state capability is resident-only")
        self._validate_layout(descriptor_tuple, page_size_bytes)

        object.__setattr__(self, "block_size", 1)
        object.__setattr__(self, "descriptors", descriptor_tuple)
        object.__setattr__(self, "state_name", state_name)
        object.__setattr__(self, "persistence_class", persistence_class)
        object.__setattr__(self, "_page_size_bytes", page_size_bytes)
        object.__setattr__(
            self,
            "schema_id",
            self._schema_digest(
                descriptor_tuple,
                page_size_bytes,
                state_name,
                persistence_class,
            ),
        )

    @staticmethod
    def _validate_layout(
        descriptors: tuple[PersistentStateDescriptor, ...],
        page_size_bytes: int,
    ) -> None:
        if not descriptors:
            raise ValueError("persistent-state layout must not be empty")
        if page_size_bytes <= 0:
            raise ValueError("persistent-state page size must be positive")
        names = [descriptor.name for descriptor in descriptors]
        if len(set(names)) != len(names):
            raise ValueError("persistent-state descriptor names must be unique")
        maximum_alignment = max(descriptor.alignment_bytes for descriptor in descriptors)
        if page_size_bytes % maximum_alignment:
            raise ValueError("persistent-state page size must satisfy descriptor alignment")
        if any(page_size_bytes % descriptor.element_size_bytes for descriptor in descriptors):
            raise ValueError("persistent-state page size must satisfy dtype alignment")
        intervals = sorted(
            (
                descriptor.offset_bytes,
                descriptor.offset_bytes + descriptor.size_bytes,
                descriptor.name,
            )
            for descriptor in descriptors
        )
        for start, end, name in intervals:
            if end > page_size_bytes:
                raise ValueError(f"persistent-state descriptor {name!r} exceeds its page")
        for (_, previous_end, previous_name), (
            next_start,
            _,
            next_name,
        ) in zip(intervals, intervals[1:]):
            if previous_end > next_start:
                raise ValueError(f"persistent-state descriptors overlap: {previous_name!r}, {next_name!r}")
        aligned_extent = (
            (max(end for _, end, _ in intervals) + maximum_alignment - 1) // maximum_alignment * maximum_alignment
        )
        if page_size_bytes != aligned_extent:
            raise ValueError("persistent-state page size is not the aligned layout extent")

    @staticmethod
    def _schema_digest(
        descriptors: tuple[PersistentStateDescriptor, ...],
        page_size_bytes: int,
        state_name: str,
        persistence_class: str,
    ) -> str:
        payload = {
            "state_name": state_name,
            "persistence_class": persistence_class,
            "block_size": 1,
            "page_size_bytes": page_size_bytes,
            "descriptors": [
                {
                    "name": descriptor.name,
                    "shape": descriptor.shape,
                    "dtype": str(descriptor.dtype),
                    "offset_bytes": descriptor.offset_bytes,
                    "alignment_bytes": descriptor.alignment_bytes,
                    "initializer_module": getattr(descriptor.initializer, "__module__", ""),
                    "initializer_name": getattr(descriptor.initializer, "__qualname__", ""),
                }
                for descriptor in descriptors
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def page_size_bytes(self) -> int:
        return self._page_size_bytes

    def max_memory_usage_bytes(self, vllm_config: Any) -> int:
        # A persistent page is one resident slot, never a token-history bound.
        del vllm_config
        return self.page_size_bytes

    def copy_with_new_block_size(self, block_size: int) -> PersistentStateSpec:
        if block_size != 1:
            raise ValueError("persistent-state block size is fixed at one")
        return self


class PersistentStateLayerBase(nn.Module):
    """Purpose-named model declaration, independent of attention and Mamba."""

    def __init__(
        self,
        spec: PersistentStateSpec,
        *,
        prefix: str,
        vllm_config: Any | None = None,
    ) -> None:
        super().__init__()
        if not prefix:
            raise ValueError("persistent-state layer prefix must be nonempty")
        self._persistent_state_spec = spec
        self._persistent_state_storage: Any | None = None
        config = vllm_config or get_current_vllm_config()
        forward_context = config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self
        self.prefix = prefix

    # @spec PORT-STATE-002
    def get_kv_cache_spec(self, vllm_config: Any) -> PersistentStateSpec:
        del vllm_config
        return self._persistent_state_spec

    def bind_persistent_state_storage(self, storage: Any) -> None:
        """Bind the runner-owned allocation without claiming lifecycle authority."""

        if self._persistent_state_storage is not None:
            raise RuntimeError("persistent-state storage is already bound")
        if getattr(storage, "spec", None) != self._persistent_state_spec:
            raise ValueError("persistent-state storage schema mismatch")
        self._persistent_state_storage = storage

    @property
    def persistent_state_storage(self) -> Any:
        if self._persistent_state_storage is None:
            raise RuntimeError("persistent-state storage is not bound")
        return self._persistent_state_storage
