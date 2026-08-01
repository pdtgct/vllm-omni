# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw aggregate allocation and typed persistent-state views."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .spec import PersistentStateDescriptor, PersistentStateSpec


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    reversed_strides: list[int] = []
    for dimension in reversed(shape):
        reversed_strides.append(stride)
        stride *= dimension
    return tuple(reversed(reversed_strides))


@dataclass
class PersistentStateStorage:
    """One raw byte allocation plus descriptor-native typed views."""

    spec: PersistentStateSpec
    raw: torch.Tensor
    views: dict[str, torch.Tensor]
    _initialized_generations: dict[int, int] = field(default_factory=dict)

    # @spec PORT-STATE-003, PORT-STATE-009
    def initialize_fresh_state_slot(self, slot: int, generation: int) -> None:
        if slot < 0 or slot >= self.raw.numel() // self.spec.page_size_bytes:
            raise ValueError("persistent-state slot is out of range")
        if self._initialized_generations.get(slot) == generation:
            return
        page_start = slot * self.spec.page_size_bytes
        page_end = page_start + self.spec.page_size_bytes
        self.raw[page_start:page_end].zero_()
        for descriptor in self.spec.descriptors:
            target = self.views[descriptor.name][slot]
            initialized = descriptor.initializer(
                descriptor.shape,
                dtype=descriptor.dtype,
                device=target.device,
            )
            if not isinstance(initialized, torch.Tensor):
                initialized = torch.as_tensor(
                    initialized,
                    dtype=descriptor.dtype,
                    device=target.device,
                )
            if tuple(initialized.shape) != descriptor.shape:
                raise ValueError(f"initializer for {descriptor.name!r} returned wrong shape")
            target.copy_(initialized)
        self._initialized_generations[slot] = generation


def _typed_view(
    raw: torch.Tensor,
    descriptor: PersistentStateDescriptor,
    *,
    num_slots: int,
    page_size_bytes: int,
) -> torch.Tensor:
    element_size = descriptor.element_size_bytes
    if page_size_bytes % element_size:
        raise ValueError("persistent-state page stride is not dtype-aligned")
    typed_tail = raw[descriptor.offset_bytes :].view(descriptor.dtype)
    return torch.as_strided(
        typed_tail,
        size=(num_slots, *descriptor.shape),
        stride=(
            page_size_bytes // element_size,
            *_contiguous_strides(descriptor.shape),
        ),
    )


# @spec PORT-STATE-003, PORT-STATE-009, PORT-STATE-011
def allocate_persistent_state_storage(
    spec: PersistentStateSpec,
    num_slots: int,
    device: torch.device,
) -> PersistentStateStorage:
    if num_slots <= 0:
        raise ValueError("persistent-state allocation requires a positive slot count")
    raw = torch.empty(
        num_slots * spec.page_size_bytes,
        dtype=torch.uint8,
        device=device,
    )
    views = {
        descriptor.name: _typed_view(
            raw,
            descriptor,
            num_slots=num_slots,
            page_size_bytes=spec.page_size_bytes,
        )
        for descriptor in spec.descriptors
    }
    expected_bytes = num_slots * spec.page_size_bytes
    if raw.numel() != expected_bytes or not raw.is_contiguous():
        raise RuntimeError("persistent-state raw allocation is malformed")
    for descriptor in spec.descriptors:
        if math.prod(views[descriptor.name].shape[1:]) != math.prod(descriptor.shape):
            raise RuntimeError("persistent-state typed view is malformed")
    return PersistentStateStorage(spec=spec, raw=raw, views=views)


def persistent_state_storage_from_raw(
    spec: PersistentStateSpec,
    raw: torch.Tensor,
) -> PersistentStateStorage:
    """Project one core-sized raw allocation into descriptor-native views."""

    if raw.dtype not in (torch.int8, torch.uint8) or not raw.is_contiguous():
        raise ValueError("persistent-state backing storage must be contiguous bytes")
    if raw.numel() % spec.page_size_bytes:
        raise ValueError("persistent-state backing storage has a partial page")
    num_slots = raw.numel() // spec.page_size_bytes
    if num_slots <= 0:
        raise ValueError("persistent-state backing storage has no slots")
    views = {
        descriptor.name: _typed_view(
            raw,
            descriptor,
            num_slots=num_slots,
            page_size_bytes=spec.page_size_bytes,
        )
        for descriptor in spec.descriptors
    }
    return PersistentStateStorage(spec=spec, raw=raw, views=views)


def initialize_persistent_state_slot(
    storage: PersistentStateStorage,
    slot: int,
    generation: int,
) -> None:
    storage.initialize_fresh_state_slot(slot, generation)
