# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-profile manifests (PORT-WGT-004 / PORT-STATE-009 / PORT-INT-005).

The reusable capability boundary the port authors at publish time and
asserts at model startup: four small, checkpoint-derived manifests
(state / geometry / transition / emission) plus a checkpoint profile
naming their content hashes. Every manifest is canonical-JSON-hashable
so a startup mismatch names the first offending field rather than
silently drifting (PORT-STATE-009).

STDLIB ONLY — no ``torch``, no ``vllm_omni`` package import. This is
what lets ``test_manifests.py`` load this module by file path and run
locally on macOS, unlike the rest of the port (contrast
``test_advance.py``, which is pod-tier). ``author_*`` and
``verify_state_manifest`` are tests-first stubs (Phase 5): they raise
``NotImplementedError`` with the documented schema/contract so pod
tests pin the future behavior now and turn green in Phase 6.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: The five published att-context (left, right) pairs, by nominal
#: cadence label — PORT-EVAL's five-cadence contract (port-design.md).
CADENCES: dict[str, tuple[int, int]] = {
    "80ms": (56, 0),
    "160ms": (56, 1),
    "320ms": (56, 3),
    "560ms": (56, 6),
    "1120ms": (56, 13),
}


def canonical_json(obj: Any) -> str:
    """The canonical JSON serialization every manifest hash is over.

    Sorted keys, compact separators, ASCII-only — so byte-identical
    content always hashes identically regardless of key insertion
    order or platform default separators.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def manifest_hash(obj: Any) -> str:
    """``"sha256:" + hex`` over :func:`canonical_json`'s bytes."""
    digest = hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()
    return f"sha256:{digest}"


def author_state_manifest(config: Any) -> dict[str, Any]:
    """Author the state manifest (PORT-WGT-004).

    Schema (``"schema": "state-manifest-v1"``):
    ``{"schema": ..., "entries": [{"name", "shape", "dtype", "init"},
    ...], "total_page_bytes": int}``. Entries enumerate the per-session
    resumable state exactly (PORT-STATE-001): per encoder layer the
    channel window, time window, and valid-length slot; predictor
    ``h``/``c``; the emission queue; the session book. ``init`` is one
    of ``"zeros"`` or ``"blank_label"`` (the predictor's last-label
    slot alone starts at the blank id, not zero).

    Args:
        config: the checkpoint's ``NemotronASRConfig`` (or equivalent),
            supplying ``n_layers``/window/d_model/kernel/queue geometry.

    Returns:
        The state manifest dict.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
    """
    raise NotImplementedError(
        "author_state_manifest lands in Phase 6 (PORT-WGT-004)"
    )


def author_geometry_manifest(config: Any) -> dict[str, Any]:
    """Author the geometry manifest (PORT-WGT-004).

    Schema: ``{"schema": "geometry-manifest-v1", "cadences": {label:
    {"att_context": [l, r], "frames_per_chunk": r + 1}, ...},
    "carrier_width": config.hidden_size, "feat": 128}``. One entry per
    :data:`CADENCES` label — the five published cadences, immutable
    per session.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
    """
    raise NotImplementedError(
        "author_geometry_manifest lands in Phase 6 (PORT-WGT-004)"
    )


def author_transition_manifest(config: Any) -> dict[str, Any]:
    """Author the transition manifest (PORT-WGT-004).

    Schema: ``{"schema": "transition-manifest-v1", "transition":
    "advance_session-v1", "roles": ["CHUNK"], "session_first_init":
    "zero-state-blank-label", "drop_extra":
    "zero-on-session-first-else-pre-encode-cache", "final_tail":
    "actual-residual-final-stft", "echo_policy": "mrv1-echo-verify"}``
    — the fixed contract every native/fallback/probe caller invokes
    through (PORT-ADV-001/003).

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
    """
    raise NotImplementedError(
        "author_transition_manifest lands in Phase 6 (PORT-WGT-004)"
    )


def author_emission_manifest(config: Any) -> dict[str, Any]:
    """Author the emission manifest (PORT-WGT-004).

    Schema: ``{"schema": "emission-manifest-v1", "per_geometry":
    {label: {"max_valid_encoder_frames": int, "max_symbols_per_step":
    int, "max_emission_tokens": frames * symbols + 1}, ...}}`` — the
    PORT-INT-002 budget formula, ``+ 1`` for the park token, one entry
    per :data:`CADENCES` label.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
    """
    raise NotImplementedError(
        "author_emission_manifest lands in Phase 6 (PORT-WGT-004)"
    )


def author_checkpoint_profile(manifest_hashes: dict[str, str]) -> dict[str, Any]:
    """Author the checkpoint profile (PORT-INT-005).

    Schema: ``{"schema": "checkpoint-profile-v1", "id":
    "cp-nemotron-3.5-asr-streaming-0.6b", "state_manifest_hash": ...,
    "geometry_manifest_hash": ..., "transition_manifest_hash": ...,
    "emission_manifest_hash": ..., "content_hash": "sha256:..."}`` —
    ``content_hash`` is :func:`manifest_hash` over the profile dict
    WITHOUT the ``content_hash`` key itself (no self-reference).

    Args:
        manifest_hashes: the four manifest hashes, keyed
            ``"state"``/``"geometry"``/``"transition"``/``"emission"``.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
    """
    raise NotImplementedError(
        "author_checkpoint_profile lands in Phase 6 (PORT-INT-005)"
    )


def verify_state_manifest(
    manifest: dict[str, Any],
    *,
    n_layers: int,
    window: int,
    feat: int,
    hidden: int,
    queue_capacity: int,
    dtype: str,
) -> None:
    """Assert the state manifest against the live geometry (PORT-STATE-009).

    The startup assertion seam: raises naming the first mismatching
    entry or field (never a bare shape-mismatch count), and asserts
    ``total_page_bytes`` equals both the sum over ``entries`` AND the
    FP32 bring-up pin (see ``docs/intent/port/port-design.md``'s
    per-family byte table in the notes repo for the authored total —
    changing weight or compute precision must not implicitly change
    this manifest, per PORT-STATE-009).

    Args:
        manifest: the state manifest to check (``author_state_manifest``'s
            schema).
        n_layers: the live encoder layer count.
        window: the live encoder left-context window.
        feat: the live mel-bin count.
        hidden: the live encoder ``d_model``.
        queue_capacity: the live replay-queue capacity
            (``max_symbols_per_step * max_frames_per_chunk``).
        dtype: the live bring-up dtype string (e.g. ``"float32"``).

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
        ValueError: (Phase 6 behavior, documented here for the pinned
            contract) naming the first mismatching entry/field, or a
            ``total_page_bytes`` that disagrees with the summed entries.
    """
    raise NotImplementedError(
        "verify_state_manifest lands in Phase 6 (PORT-STATE-009)"
    )
