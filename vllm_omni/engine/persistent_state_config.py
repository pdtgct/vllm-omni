# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolved runtime envelope for persistent-state streaming serving."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

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
_DEFAULT_ADMISSION_AGING_S = 1.0
_DEFAULT_ADMISSION_WAIT_S = 5.0
_DEFAULT_ADMISSION_RETRY_FLOOR_MS = 100
_DEFAULT_ADMISSION_RETRY_JITTER_MS = 25
_DEFAULT_RECOVERY_BACKOFF_S = (0.1, 0.5, 1.0)
_DEFAULT_RELEASE_CONVERGENCE_S = 61.0

_A36_REQUIRED_KEYS = (
    "persistent_state_admission_waiter_capacity",
    "persistent_state_admission_max_inflight_reserves",
    "persistent_state_admission_dispatch_budget",
    "persistent_state_admission_aging_threshold_s",
    "persistent_state_admission_wait_timeout_s",
    "persistent_state_admission_retry_floor_ms",
    "persistent_state_admission_retry_jitter_ms",
    "persistent_state_recovery_backoff_s",
    "persistent_state_release_convergence_timeout_s",
    "streaming_unadmitted_connection_timeout_s",
    "persistent_state_service_profile_trailing_rounds",
    "persistent_state_startup_priming_round_timeout_s",
    "persistent_state_startup_priming_timeout_s",
    "persistent_state_service_profile_derating_factor",
    "persistent_state_admission_policy",
)

_KNOWN_PERSISTENT_STATE_KEYS = frozenset(
    {
        "persistent_state_safety_reserve_slots",
        "persistent_state_reserve_queue_capacity",
        "persistent_state_operation_timeout_s",
        "persistent_state_reconciliation_timeout_s",
        "persistent_state_tombstone_ttl_s",
        "persistent_state_max_tombstones",
        "persistent_state_pending_claim_timeout_s",
        *_A36_REQUIRED_KEYS,
    }
)


def _validated_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _validated_duration(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite duration, got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be a positive finite duration, got {value!r}")
    return resolved


def _validated_ratio(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite ratio in (0, 1], got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0 < resolved <= 1:
        raise ValueError(f"{name} must be a finite ratio in (0, 1], got {value!r}")
    return resolved


def _validated_backoff(name: str, value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a non-empty sequence of positive durations")
    try:
        resolved = tuple(_validated_duration(name, item) for item in value)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-empty sequence of positive durations") from error
    if not resolved:
        raise ValueError(f"{name} must be a non-empty sequence of positive durations")
    return resolved


def _validated_admission_policy(
    name: str,
    value: object,
) -> Literal["profile", "hard_cap"]:
    if value == "profile":
        return "profile"
    if value == "hard_cap":
        return "hard_cap"
    raise ValueError(f"{name} must be exactly 'profile' or 'hard_cap', got {value!r}")


def _resolve_a36_envelope(
    values: Mapping[str, Any],
    *,
    configured_ceiling: int,
) -> dict[str, Any]:
    """Resolve the mandatory A36 fields and report all omissions together."""

    policy_value = values.get("persistent_state_admission_policy")
    try:
        policy = _validated_admission_policy(
            "persistent_state_admission_policy",
            policy_value,
        )
    except ValueError as error:
        missing = [name for name in _A36_REQUIRED_KEYS if name not in values]
        invalid_policy_fields = [name for name in _A36_REQUIRED_KEYS if name in values and values[name] is None]
        if (
            "persistent_state_admission_policy" in values
            and "persistent_state_admission_policy" not in invalid_policy_fields
        ):
            invalid_policy_fields.append("persistent_state_admission_policy")
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if invalid_policy_fields:
            parts.append(f"invalid={invalid_policy_fields}")
        raise ValueError(
            "persistent-state A36 admission envelope is incomplete or invalid: " + ", ".join(parts)
        ) from error
    profile_keys = {
        "persistent_state_service_profile_trailing_rounds",
        "persistent_state_startup_priming_round_timeout_s",
        "persistent_state_startup_priming_timeout_s",
        "persistent_state_service_profile_derating_factor",
    }
    if policy == "hard_cap" and profile_keys.intersection(values):
        raise ValueError(
            "hard_cap rejects profile-only persistent-state fields: "
            + ", ".join(sorted(profile_keys.intersection(values)))
        )
    defaults: dict[str, Any] = {}
    if policy == "hard_cap":
        defaults = {
            "persistent_state_admission_waiter_capacity": 2 * configured_ceiling,
            "persistent_state_admission_max_inflight_reserves": min(4, configured_ceiling),
            "persistent_state_admission_dispatch_budget": 1,
            "persistent_state_admission_aging_threshold_s": _DEFAULT_ADMISSION_AGING_S,
            "persistent_state_admission_wait_timeout_s": _DEFAULT_ADMISSION_WAIT_S,
            "persistent_state_admission_retry_floor_ms": _DEFAULT_ADMISSION_RETRY_FLOOR_MS,
            "persistent_state_admission_retry_jitter_ms": _DEFAULT_ADMISSION_RETRY_JITTER_MS,
            "persistent_state_recovery_backoff_s": _DEFAULT_RECOVERY_BACKOFF_S,
            "persistent_state_release_convergence_timeout_s": _DEFAULT_RELEASE_CONVERGENCE_S,
            "streaming_unadmitted_connection_timeout_s": (
                2 * (_DEFAULT_SESSION_CONFIGURATION_TIMEOUT_S + _DEFAULT_ADMISSION_WAIT_S)
                + (_DEFAULT_ADMISSION_RETRY_FLOOR_MS + _DEFAULT_ADMISSION_RETRY_JITTER_MS) / 1000
            ),
        }
    required = tuple(
        name
        for name in _A36_REQUIRED_KEYS
        if name != "persistent_state_admission_policy" and (policy == "profile" or name not in profile_keys)
    )
    missing = [name for name in required if name not in values and name not in defaults]
    invalid: list[str] = []
    resolved: dict[str, Any] = {"persistent_state_admission_policy": policy}

    def positive_int(name: str, value: object) -> int:
        return _validated_int(name, value, minimum=1)

    validators = {
        "persistent_state_admission_waiter_capacity": positive_int,
        "persistent_state_admission_max_inflight_reserves": positive_int,
        "persistent_state_admission_dispatch_budget": positive_int,
        "persistent_state_admission_aging_threshold_s": _validated_duration,
        "persistent_state_admission_wait_timeout_s": _validated_duration,
        "persistent_state_admission_retry_floor_ms": lambda name, value: _validated_int(name, value, minimum=1),
        "persistent_state_admission_retry_jitter_ms": lambda name, value: _validated_int(name, value, minimum=0),
        "persistent_state_recovery_backoff_s": _validated_backoff,
        "persistent_state_release_convergence_timeout_s": _validated_duration,
        "streaming_unadmitted_connection_timeout_s": _validated_duration,
        "persistent_state_service_profile_trailing_rounds": lambda name, value: _validated_int(name, value, minimum=3),
        "persistent_state_startup_priming_round_timeout_s": (_validated_duration),
        "persistent_state_startup_priming_timeout_s": _validated_duration,
        "persistent_state_service_profile_derating_factor": _validated_ratio,
        "persistent_state_admission_policy": _validated_admission_policy,
    }
    for name in required:
        if name not in values and name not in defaults:
            continue
        try:
            resolved[name] = validators[name](
                name,
                values[name] if name in values else defaults[name],
            )
        except ValueError:
            invalid.append(name)
    if missing or invalid:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if invalid:
            parts.append(f"invalid={invalid}")
        raise ValueError("persistent-state A36 admission envelope is incomplete or invalid: " + ", ".join(parts))
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
    admission_waiter_capacity: int
    admission_max_inflight_reserves: int
    admission_dispatch_budget: int
    admission_aging_threshold_s: float
    admission_wait_timeout_s: float
    admission_retry_floor_ms: int
    admission_retry_jitter_ms: int
    recovery_backoff_s: tuple[float, ...]
    release_convergence_timeout_s: float
    unadmitted_connection_timeout_s: float
    service_profile_trailing_rounds: int | None
    startup_priming_round_timeout_s: float | None
    startup_priming_timeout_s: float | None
    service_profile_derating_factor: float | None
    admission_policy: Literal["profile", "hard_cap"]
    priming_budget_descriptor: Any | None
    priming_budget_sha256: str | None
    priming_configured_population_ceiling: int
    bootstrap_operation_budget: int
    runtime_tombstone_allowance: int

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

        return self.accepted_audio_budget_s + (_RFC1_FINALIZATION_SERVICE_INTERVALS * _RFC1_MAX_CADENCE_S)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: Any,
        *,
        startup_provider: Any | None = None,
    ) -> PersistentStateRuntimeConfig:
        raw = getattr(vllm_config, "additional_config", None)
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("additional_config must be a mapping when provided")
        unprefixed = sorted(_UNPREFIXED_STREAMING_KEYS.intersection(raw))
        if unprefixed:
            raise ValueError(f"streaming serving keys in additional_config require the streaming_ prefix: {unprefixed}")
        invalid_nulls = sorted(
            name
            for name, value in raw.items()
            if value is None and name != "streaming_max_session_duration_s" and name not in _A36_REQUIRED_KEYS
        )
        if invalid_nulls:
            raise ValueError(f"explicit null is invalid for persistent-state serving keys: {invalid_nulls}")
        unknown_persistent_state_keys = sorted(
            name for name in raw if name.startswith("persistent_state_") and name not in _KNOWN_PERSISTENT_STATE_KEYS
        )
        if unknown_persistent_state_keys:
            raise ValueError(f"unknown persistent-state additional_config keys: {unknown_persistent_state_keys}")
        scheduler_max_num_seqs = int(
            getattr(
                getattr(vllm_config, "scheduler_config", None),
                "max_num_seqs",
                _DEFAULT_MAX_RESIDENT_SESSIONS,
            )
        )
        requested_resident_sessions = _resolved_int(
            raw,
            "max_resident_sessions",
            default=scheduler_max_num_seqs,
            minimum=1,
        )
        hard_cap_selected = raw.get("persistent_state_admission_policy") == "hard_cap"
        max_resident_sessions = (
            min(requested_resident_sessions, scheduler_max_num_seqs)
            if hard_cap_selected
            else requested_resident_sessions
        )
        a36 = _resolve_a36_envelope(
            raw,
            configured_ceiling=max_resident_sessions,
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
                "persistent_state_reconciliation_timeout_s must be no shorter than persistent_state_operation_timeout_s"
            )
        if startup_provider is None:
            model_config = getattr(vllm_config, "model_config", None)
            architectures = getattr(model_config, "architectures", None)
            if architectures is None:
                hf_config = getattr(model_config, "hf_config", None)
                architectures = getattr(hf_config, "architectures", ())
            if isinstance(architectures, str):
                architectures = (architectures,)
            from vllm_omni.model_executor.models.registry import (
                get_persistent_state_startup_provider,
            )

            startup_provider = get_persistent_state_startup_provider(tuple(architectures or ()))
        runtime_tombstone_allowance = max(
            _DEFAULT_TOMBSTONE_FLOOR,
            4 * max_resident_sessions,
        )
        priming_budget_descriptor = None
        priming_budget_sha256: str | None = None
        priming_configured_population_ceiling = 0
        bootstrap_operation_budget = 0
        if startup_provider is not None and a36["persistent_state_admission_policy"] == "profile":
            scheduler_config = getattr(vllm_config, "scheduler_config", None)
            execution_ceiling = _validated_int(
                "scheduler.max_num_seqs",
                getattr(scheduler_config, "max_num_seqs", None),
                minimum=1,
            )
            priming_configured_population_ceiling = min(
                max_resident_sessions,
                execution_ceiling,
            )
            priming_budget_descriptor = startup_provider.build_priming_budget_descriptor(
                configured_population_ceiling=(priming_configured_population_ceiling),
                trailing_rounds=a36["persistent_state_service_profile_trailing_rounds"],
            )
            priming_budget_sha256 = str(priming_budget_descriptor.sha256)
            bootstrap_operation_budget = int(priming_budget_descriptor.bootstrap_operation_budget)
        minimum_tombstones = bootstrap_operation_budget + runtime_tombstone_allowance
        max_tombstones = _resolved_int(
            raw,
            "persistent_state_max_tombstones",
            default=minimum_tombstones,
            minimum=1,
        )
        if startup_provider is not None and max_tombstones < minimum_tombstones:
            raise ValueError(
                "persistent_state_max_tombstones cannot fund the bootstrap "
                f"operation budget and runtime allowance; need {minimum_tombstones}"
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
            max_tombstones=max_tombstones,
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
                default=max(_DEFAULT_FINALIZATION_FLOOR_S, safe_finalization_floor_s),
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
            admission_waiter_capacity=a36["persistent_state_admission_waiter_capacity"],
            admission_max_inflight_reserves=a36["persistent_state_admission_max_inflight_reserves"],
            admission_dispatch_budget=a36["persistent_state_admission_dispatch_budget"],
            admission_aging_threshold_s=a36["persistent_state_admission_aging_threshold_s"],
            admission_wait_timeout_s=a36["persistent_state_admission_wait_timeout_s"],
            admission_retry_floor_ms=a36["persistent_state_admission_retry_floor_ms"],
            admission_retry_jitter_ms=a36["persistent_state_admission_retry_jitter_ms"],
            recovery_backoff_s=a36["persistent_state_recovery_backoff_s"],
            release_convergence_timeout_s=a36["persistent_state_release_convergence_timeout_s"],
            unadmitted_connection_timeout_s=a36["streaming_unadmitted_connection_timeout_s"],
            service_profile_trailing_rounds=a36["persistent_state_service_profile_trailing_rounds"]
            if a36["persistent_state_admission_policy"] == "profile"
            else None,
            startup_priming_round_timeout_s=a36["persistent_state_startup_priming_round_timeout_s"]
            if a36["persistent_state_admission_policy"] == "profile"
            else None,
            startup_priming_timeout_s=a36["persistent_state_startup_priming_timeout_s"]
            if a36["persistent_state_admission_policy"] == "profile"
            else None,
            service_profile_derating_factor=a36["persistent_state_service_profile_derating_factor"]
            if a36["persistent_state_admission_policy"] == "profile"
            else None,
            admission_policy=a36["persistent_state_admission_policy"],
            priming_budget_descriptor=priming_budget_descriptor,
            priming_budget_sha256=priming_budget_sha256,
            priming_configured_population_ceiling=(priming_configured_population_ceiling),
            bootstrap_operation_budget=bootstrap_operation_budget,
            runtime_tombstone_allowance=runtime_tombstone_allowance,
        )
        if resolved.session_finalization_timeout_s < resolved.safe_finalization_timeout_s:
            raise ValueError(
                "streaming_session_finalization_timeout_s must be at least "
                "the safe "
                f"drain bound {resolved.safe_finalization_timeout_s:.3f}s"
            )
        if not (
            1
            <= resolved.admission_dispatch_budget
            <= resolved.admission_max_inflight_reserves
            <= min(
                resolved.admission_waiter_capacity,
                resolved.reserve_queue_capacity,
            )
        ):
            raise ValueError(
                "persistent_state_admission_dispatch_budget must be <= "
                "persistent_state_admission_max_inflight_reserves, and both "
                "must fit the waiter and reserve capacities"
            )
        if resolved.admission_aging_threshold_s >= resolved.admission_wait_timeout_s:
            raise ValueError(
                "persistent_state_admission_aging_threshold_s must be shorter "
                "than persistent_state_admission_wait_timeout_s"
            )
        minimum_unadmitted_lifetime_s = (
            2 * (resolved.session_configuration_timeout_s + resolved.admission_wait_timeout_s)
            + (resolved.admission_retry_floor_ms + resolved.admission_retry_jitter_ms) / 1000
        )
        if resolved.unadmitted_connection_timeout_s < minimum_unadmitted_lifetime_s:
            raise ValueError(
                "streaming_unadmitted_connection_timeout_s must cover two "
                "complete configuration/admission attempts and one retry cycle "
                f"({minimum_unadmitted_lifetime_s:.3f}s)"
            )
        minimum_release_convergence_s = 2 * resolved.reconciliation_timeout_s + resolved.recovery_backoff_s[0]
        if resolved.release_convergence_timeout_s < minimum_release_convergence_s:
            raise ValueError(
                "persistent_state_release_convergence_timeout_s must cover "
                "two reconciliation windows and the first recovery backoff "
                f"({minimum_release_convergence_s:.3f}s)"
            )
        return resolved
