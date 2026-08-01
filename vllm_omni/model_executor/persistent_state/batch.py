# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed, preflight-validated row bindings for persistent state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .manager import PersistentStateManager, StateBinding


@dataclass(frozen=True)
class PersistentStateBatch:
    """One row-ordered invocation projection over manager-issued bindings."""

    rows: tuple[Mapping[str, Any], ...]

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        object.__setattr__(self, "rows", tuple(dict(row) for row in rows))

    # @spec PORT-STATE-007
    def validate(self, manager: PersistentStateManager) -> None:
        live_request_ids: set[str] = set()
        live_slots: set[int] = set()
        for index, row in enumerate(self.rows):
            if row.get("order") != index:
                raise ValueError("persistent-state batch row order mismatch")
            no_state = row.get("no_state") is True
            if no_state:
                if row.get("role") != "no_state" or row.get("binding") is not None:
                    raise ValueError("persistent-state dummy row must be explicit")
                forbidden = (
                    "request_id",
                    "generation",
                    "schema_id",
                    "profile_id",
                    "stage",
                    "replica",
                    "slot_id",
                )
                if any(row.get(name) is not None for name in forbidden):
                    raise ValueError("persistent-state dummy row carries live identity")
                if row.get("fresh") not in (False, None):
                    raise ValueError("persistent-state dummy row cannot be fresh")
                continue

            if row.get("role") != "resident":
                raise ValueError("persistent-state live row must be resident")
            binding = row.get("binding")
            if not isinstance(binding, StateBinding):
                raise TypeError("persistent-state live row lacks typed binding")
            request_id = row.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("persistent-state request identity must be nonempty")
            if row.get("request_key") != request_id:
                raise ValueError("persistent-state request key mismatch")
            authoritative = manager.get_state_binding(request_id)
            if authoritative is None or authoritative != binding:
                raise ValueError("persistent-state binding is stale or non-authoritative")
            expected = {
                "request_id": binding.request_id,
                "generation": binding.generation,
                "schema_id": binding.schema_id,
                "profile_id": binding.profile_id,
                "stage": binding.stage,
                "replica": binding.replica,
                "slot_id": binding.slot_id,
                "fresh": binding.fresh,
            }
            for field_name, value in expected.items():
                if row.get(field_name) != value:
                    raise ValueError(f"persistent-state {field_name} disagrees with manager binding")
            if binding.slot_id <= 0 or binding.slot_id >= manager.block_pool.num_gpu_blocks:
                raise ValueError("persistent-state slot is null or out of range")
            if request_id in live_request_ids:
                raise ValueError("duplicate persistent-state request identity")
            if binding.slot_id in live_slots:
                raise ValueError("duplicate persistent-state live slot")
            live_request_ids.add(request_id)
            live_slots.add(binding.slot_id)

    # @spec PORT-STATE-007, PORT-STATE-008
    def gather(
        self,
        manager: PersistentStateManager,
        gatherer: Callable[[Mapping[str, Any]], Any],
    ) -> tuple[Any, ...]:
        self.validate(manager)
        return tuple(gatherer(row) for row in self.rows if row.get("no_state") is not True)

    # @spec PORT-STATE-007, PORT-STATE-008
    def scatter(
        self,
        manager: PersistentStateManager,
        row_states: Sequence[Any],
        row_statuses: Sequence[str],
        scatterer: Callable[..., None],
    ) -> None:
        self.validate(manager)
        if len(row_states) != len(self.rows) or len(row_statuses) != len(self.rows):
            raise ValueError("persistent-state scatter cardinality mismatch")
        for row, state, status in zip(self.rows, row_states, row_statuses):
            if row.get("no_state") is True:
                if status != "no_state":
                    raise ValueError("persistent-state dummy status mismatch")
                continue
            if status == "clean":
                if state is None:
                    raise ValueError("clean persistent-state row lacks state")
                scatterer(row, state=state)
            elif status != "failed":
                raise ValueError(f"unknown persistent-state row status {status!r}")


def validate_persistent_state_batch(
    batch: PersistentStateBatch,
    manager: PersistentStateManager,
) -> None:
    batch.validate(manager)


def gather_persistent_state_batch(
    batch: PersistentStateBatch,
    manager: PersistentStateManager,
    gatherer: Callable[[Mapping[str, Any]], Any],
) -> tuple[Any, ...]:
    return batch.gather(manager, gatherer)


def scatter_persistent_state_batch(
    batch: PersistentStateBatch,
    manager: PersistentStateManager,
    row_states: Sequence[Any],
    row_statuses: Sequence[str],
    scatterer: Callable[..., None],
) -> None:
    batch.scatter(manager, row_states, row_statuses, scatterer)
