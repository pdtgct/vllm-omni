# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolved runtime limits for the persistent-state control plane."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _required_int(
    values: Mapping[str, Any],
    name: str,
    *,
    minimum: int,
) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an explicit integer >= {minimum}, got {value!r}"
        )
    return value


def _required_duration(values: Mapping[str, Any], name: str) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be an explicit positive finite duration, got {value!r}"
        )
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(
            f"{name} must be an explicit positive finite duration, got {value!r}"
        )
    return resolved


@dataclass(frozen=True)
class PersistentStateRuntimeConfig:
    """ENV-owned limits shared by API and EngineCore processes.

    The values come from ``VllmConfig.additional_config`` so they are part
    of vLLM's configuration hash and execution fingerprint.  The cleanup
    queue is derived from the resident-session limit: every acknowledged
    lease has one reserved cleanup entry and cannot be displaced by reserve
    pressure.
    """

    safety_reserve_slots: int
    max_resident_sessions: int
    reserve_queue_capacity: int
    operation_timeout_s: float
    reconciliation_timeout_s: float
    tombstone_ttl_s: float
    max_tombstones: int
    pending_claim_timeout_s: float

    @property
    def cleanup_queue_capacity(self) -> int:
        return self.max_resident_sessions

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> PersistentStateRuntimeConfig:
        raw = getattr(vllm_config, "additional_config", None)
        if not isinstance(raw, Mapping):
            raise ValueError(
                "persistent-state runtime limits require additional_config"
            )
        operation_timeout_s = _required_duration(
            raw, "persistent_state_operation_timeout_s"
        )
        reconciliation_timeout_s = _required_duration(
            raw, "persistent_state_reconciliation_timeout_s"
        )
        if reconciliation_timeout_s < operation_timeout_s:
            raise ValueError(
                "persistent_state_reconciliation_timeout_s must be no shorter "
                "than persistent_state_operation_timeout_s"
            )
        return cls(
            safety_reserve_slots=_required_int(
                raw,
                "persistent_state_safety_reserve_slots",
                minimum=0,
            ),
            max_resident_sessions=_required_int(
                raw, "max_resident_sessions", minimum=1
            ),
            reserve_queue_capacity=_required_int(
                raw,
                "persistent_state_reserve_queue_capacity",
                minimum=1,
            ),
            operation_timeout_s=operation_timeout_s,
            reconciliation_timeout_s=reconciliation_timeout_s,
            tombstone_ttl_s=_required_duration(
                raw, "persistent_state_tombstone_ttl_s"
            ),
            max_tombstones=_required_int(
                raw, "persistent_state_max_tombstones", minimum=1
            ),
            pending_claim_timeout_s=_required_duration(
                raw, "persistent_state_pending_claim_timeout_s"
            ),
        )
