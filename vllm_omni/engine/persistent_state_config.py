# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolved runtime envelope for persistent-state streaming serving."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_RFC1_SAMPLE_RATE_HZ = 16_000
_RFC1_MAX_CADENCE_S = 1.120
_RFC1_FINALIZATION_SERVICE_INTERVALS = 2
_UNPREFIXED_STREAMING_KEYS = frozenset(
    {
        "session_configuration_timeout_s",
        "session_idle_timeout_s",
        "session_finalization_timeout_s",
        "accepted_audio_capacity_samples",
        "max_retained_transcript_bytes",
        "max_session_duration_s",
    }
)

# PORT-owned envelope defaults (ENV-MIG-009): omission resolves through
# these — the exact values every qualified GPU round served — while an
# explicit null or invalid value still fails closed. Derived defaults
# (reconciliation >= operation, queue = sessions, tombstones and the
# finalization floor scaled from their inputs) are computed in
# ``from_vllm_config`` so a partial override stays self-consistent.
_DEFAULT_SAFETY_RESERVE_SLOTS = 1
_DEFAULT_MAX_RESIDENT_SESSIONS = 8
_DEFAULT_OPERATION_TIMEOUT_S = 10.0
_DEFAULT_RECONCILIATION_FLOOR_S = 30.0
_DEFAULT_TOMBSTONE_TTL_S = 600.0
_DEFAULT_TOMBSTONE_FLOOR = 32
_DEFAULT_PENDING_CLAIM_TIMEOUT_S = 15.0
_DEFAULT_SESSION_CONFIGURATION_TIMEOUT_S = 10.0
_DEFAULT_SESSION_IDLE_TIMEOUT_S = 60.0
_DEFAULT_FINALIZATION_FLOOR_S = 40.0
_DEFAULT_ACCEPTED_AUDIO_CAPACITY_SAMPLES = 30 * _RFC1_SAMPLE_RATE_HZ
_DEFAULT_MAX_RETAINED_TRANSCRIPT_BYTES = 1 << 20


def _validated_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _validated_duration(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a positive finite duration, got {value!r}"
        )
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(
            f"{name} must be a positive finite duration, got {value!r}"
        )
    return resolved


def _resolved_int(
    values: Mapping[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    if name not in values:
        return default
    return _validated_int(name, values[name], minimum=minimum)


def _resolved_duration(
    values: Mapping[str, Any],
    name: str,
    *,
    default: float,
) -> float:
    if name not in values:
        return default
    return _validated_duration(name, values[name])


def _optional_duration(
    values: Mapping[str, Any],
    name: str,
    *,
    default: float | None,
) -> float | None:
    value = values.get(name, default)
    if value is None:
        return None
    return _validated_duration(name, value)


@dataclass(frozen=True)
class PersistentStateRuntimeConfig:
    """ENV-owned limits shared by API and EngineCore processes.

    The values come from ``VllmConfig.additional_config`` so they are part
    of vLLM's configuration hash and execution fingerprint.  The cleanup
    queue is derived from the resident-session limit: every acknowledged
    lease has one reserved cleanup entry and cannot be displaced by reserve
    pressure. Session limits are resolved beside state limits so every live
    serving surface receives the exact same fingerprinted envelope.
    """

    safety_reserve_slots: int
    max_resident_sessions: int
    reserve_queue_capacity: int
    operation_timeout_s: float
    reconciliation_timeout_s: float
    tombstone_ttl_s: float
    max_tombstones: int
    pending_claim_timeout_s: float
    session_configuration_timeout_s: float
    session_idle_timeout_s: float
    session_finalization_timeout_s: float
    accepted_audio_capacity_samples: int
    max_retained_transcript_bytes: int
    max_session_duration_s: float | None

    @property
    def cleanup_queue_capacity(self) -> int:
        return self.max_resident_sessions

    @property
    def accepted_audio_budget_s(self) -> float:
        """Resolved accepted-audio capacity in the checkpoint sample rate."""

        return self.accepted_audio_capacity_samples / _RFC1_SAMPLE_RATE_HZ

    @property
    def max_session_samples(self) -> int | None:
        """Optional duration policy represented in the audio authority unit."""

        if self.max_session_duration_s is None:
            return None
        return int(self.max_session_duration_s * _RFC1_SAMPLE_RATE_HZ)

    @property
    def safe_finalization_timeout_s(self) -> float:
        """Conservative drain floor for the installed RFC-1 profile.

        Accepted audio can occupy the full sample-counted FIFO. One largest
        cadence service interval is reserved for the final tail and one for
        FLUSH. Hardware qualification may select a larger value, but never a
        smaller one.
        """

        return self.accepted_audio_budget_s + (
            _RFC1_FINALIZATION_SERVICE_INTERVALS * _RFC1_MAX_CADENCE_S
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> PersistentStateRuntimeConfig:
        raw = getattr(vllm_config, "additional_config", None)
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(
                "additional_config must be a mapping when provided"
            )
        unprefixed = sorted(_UNPREFIXED_STREAMING_KEYS.intersection(raw))
        if unprefixed:
            raise ValueError(
                "streaming serving keys in additional_config require the "
                f"streaming_ prefix: {unprefixed}"
            )
        operation_timeout_s = _resolved_duration(
            raw,
            "persistent_state_operation_timeout_s",
            default=_DEFAULT_OPERATION_TIMEOUT_S,
        )
        reconciliation_timeout_s = _resolved_duration(
            raw,
            "persistent_state_reconciliation_timeout_s",
            default=max(_DEFAULT_RECONCILIATION_FLOOR_S, operation_timeout_s),
        )
        if reconciliation_timeout_s < operation_timeout_s:
            raise ValueError(
                "persistent_state_reconciliation_timeout_s must be no shorter "
                "than persistent_state_operation_timeout_s"
            )
        max_resident_sessions = _resolved_int(
            raw,
            "max_resident_sessions",
            default=_DEFAULT_MAX_RESIDENT_SESSIONS,
            minimum=1,
        )
        accepted_audio_capacity_samples = _resolved_int(
            raw,
            "streaming_accepted_audio_capacity_samples",
            default=_DEFAULT_ACCEPTED_AUDIO_CAPACITY_SAMPLES,
            minimum=1,
        )
        safe_finalization_floor_s = (
            accepted_audio_capacity_samples / _RFC1_SAMPLE_RATE_HZ
            + _RFC1_FINALIZATION_SERVICE_INTERVALS * _RFC1_MAX_CADENCE_S
        )
        resolved = cls(
            safety_reserve_slots=_resolved_int(
                raw,
                "persistent_state_safety_reserve_slots",
                default=_DEFAULT_SAFETY_RESERVE_SLOTS,
                minimum=0,
            ),
            max_resident_sessions=max_resident_sessions,
            reserve_queue_capacity=_resolved_int(
                raw,
                "persistent_state_reserve_queue_capacity",
                default=max_resident_sessions,
                minimum=1,
            ),
            operation_timeout_s=operation_timeout_s,
            reconciliation_timeout_s=reconciliation_timeout_s,
            tombstone_ttl_s=_resolved_duration(
                raw,
                "persistent_state_tombstone_ttl_s",
                default=_DEFAULT_TOMBSTONE_TTL_S,
            ),
            max_tombstones=_resolved_int(
                raw,
                "persistent_state_max_tombstones",
                default=max(
                    _DEFAULT_TOMBSTONE_FLOOR, 4 * max_resident_sessions
                ),
                minimum=1,
            ),
            pending_claim_timeout_s=_resolved_duration(
                raw,
                "persistent_state_pending_claim_timeout_s",
                default=_DEFAULT_PENDING_CLAIM_TIMEOUT_S,
            ),
            session_configuration_timeout_s=_resolved_duration(
                raw,
                "streaming_session_configuration_timeout_s",
                default=_DEFAULT_SESSION_CONFIGURATION_TIMEOUT_S,
            ),
            session_idle_timeout_s=_resolved_duration(
                raw,
                "streaming_session_idle_timeout_s",
                default=_DEFAULT_SESSION_IDLE_TIMEOUT_S,
            ),
            session_finalization_timeout_s=_resolved_duration(
                raw,
                "streaming_session_finalization_timeout_s",
                default=max(
                    _DEFAULT_FINALIZATION_FLOOR_S, safe_finalization_floor_s
                ),
            ),
            accepted_audio_capacity_samples=accepted_audio_capacity_samples,
            max_retained_transcript_bytes=_resolved_int(
                raw,
                "streaming_max_retained_transcript_bytes",
                default=_DEFAULT_MAX_RETAINED_TRANSCRIPT_BYTES,
                minimum=1,
            ),
            max_session_duration_s=_optional_duration(
                raw,
                "streaming_max_session_duration_s",
                default=None,
            ),
        )
        if (
            resolved.session_finalization_timeout_s
            < resolved.safe_finalization_timeout_s
        ):
            raise ValueError(
                "streaming_session_finalization_timeout_s must be at least "
                "the safe "
                f"drain bound {resolved.safe_finalization_timeout_s:.3f}s"
            )
        return resolved
