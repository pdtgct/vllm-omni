# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-profile manifests (PORT-WGT-004 / PORT-STATE-009 / PORT-INT-005).

The reusable capability boundary the port authors at publish time and
asserts at model startup: four small, checkpoint-derived manifests
(state / geometry / transition / emission) plus a checkpoint profile
naming their content hashes. Every manifest is canonical-JSON-hashable
so a startup mismatch names the first offending field rather than
silently drifting (PORT-STATE-009).

This module is the single source of the canonical field NAMES and
ORDER for the seven-slot session book, the eight frontend counters,
the six envelope header slots, and the frontend constants/boundary
formulas — ``advance.py``'s torch-side slot constants and the Phase-6
``state_layers``/``rnnt`` layouts must match these tuples exactly.

STDLIB ONLY — no ``torch``, no ``vllm_omni`` package import. This is
what lets ``test_manifests.py`` load this module by file path and run
locally on macOS, unlike the rest of the port (contrast
``test_advance.py``, which is pod-tier). ``author_*`` and
``verify_state_manifest`` are tests-first stubs (Phase 5): they raise
``NotImplementedError``; the behavioral tests that call them and
assert their exact outputs are EXPECTED TO FAIL until Phase 6.
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

#: Raw FP32 mono samples per CHUNK at 16 kHz, by cadence label.
RAW_SAMPLES_PER_CHUNK: dict[str, int] = {
    "80ms": 1_280,
    "160ms": 2_560,
    "320ms": 5_120,
    "560ms": 8_960,
    "1120ms": 17_920,
}

#: Chunk-envelope header slot names, in slot order (design §Chunk
#: envelope; ``advance.py``'s ``ENV_*`` indices mirror this tuple).
ENVELOPE_HEADER_FIELDS: tuple[str, ...] = (
    "version",
    "valid_samples",
    "geometry_id",
    "final_tail",
    "prompt_index",
    "chunk_sequence",
)

#: The seven-slot session book, in page-slot order with each field's
#: OWN initialization rule (the book never initializes wholesale as
#: ``blank_label``). The first four slots keep ``rnnt.py``'s pinned
#: QUEUE_HEAD/QUEUE_LEN/QUEUE_LAST_LABEL/QUEUE_PROMPT order; init
#: sources: ``zeros`` — literal zero; ``blank_label`` — the
#: checkpoint's blank id; ``admitted_prompt``/``admitted_geometry`` —
#: stamped from scheduler admission metadata at fresh-row init, never
#: read from a recycled page.
BOOK_FIELDS: tuple[tuple[str, str], ...] = (
    ("queue_head", "zeros"),
    ("queue_length", "zeros"),
    ("last_label", "blank_label"),
    ("prompt", "admitted_prompt"),
    ("geometry", "admitted_geometry"),
    ("pending_echo", "zeros"),
    ("expected_label", "zeros"),
)

#: The eight int64 frontend counters, in page-slot order. The LLD's
#: six named concepts plus the two accounting slots the transition
#: needs: ``encoded_mel_frames`` (mel frames already consumed by
#: encoder chunking — the final-tail "at least eight NEW mel frames"
#: rule is ``committed_mel_frames - encoded_mel_frames >= 8``) and
#: ``mel_tail_length`` (valid frames in the fixed-width mel tail,
#: < 9 early in a session). All initialize to zero.
FRONTEND_COUNTER_FIELDS: tuple[str, ...] = (
    "total_valid_samples",
    "committed_mel_frames",
    "encoded_mel_frames",
    "raw_tail_origin",
    "raw_tail_length",
    "mel_tail_length",
    "expected_chunk_sequence",
    "finalized",
)

#: Frontend constants and boundary formulas (design §Exact Bounded
#: Frontend) — authored into the geometry manifest so a checkpoint
#: whose featurizer differs cannot silently reuse another profile.
FRONTEND_CONSTANTS: dict[str, Any] = {
    "sample_rate": 16_000,
    "preemphasis": 0.97,
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "n_mels": 128,
    "log_guard": "2**-24",
    "pre_encode_cache_frames": 9,
    "subsampling_factor": 8,
    "raw_tail_capacity": 1_953,
    "raw_tail_formula": "pre_encode_cache_frames*hop_length + n_fft + 1",
    "commit_rule": "stable-stft-window-v1",
    "final_tail_rule": "centered-stft-actual-residual-v1",
    "final_tail_min_new_mel_frames": 8,
}

#: The component-scoped precision-policy identity for FP32 bring-up
#: (A8: weight/compute precision changes never implicitly change the
#: state manifest).
PRECISION_POLICY_ID = "fp32-bringup-v1"

#: Safe session limits: every envelope header integer must stay
#: exactly FP32-representable, so the chunk-sequence ceiling is
#: ``2**24 - 1``; per-cadence maximum session duration derives from it
#: and ``CADENCES`` — it is not an independent knob. Queue/emission
#: limits are the PORT-INT-002 inputs.
SESSION_LIMITS: dict[str, int] = {
    "max_symbols_per_step": 10,
    "max_frames_per_chunk": 14,
    "queue_capacity": 140,
    "max_session_chunks": 2**24 - 1,
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

    Schema (``"schema": "state-manifest-v1"``): ``{"schema": ...,
    "precision_policy": PRECISION_POLICY_ID, "entries": [{"name",
    "shape", "dtype", "init"}, ...], "total_page_bytes": int}``.
    ``entries`` enumerate the per-session resumable state exactly
    (PORT-STATE-001), in this order:

    - per encoder layer ``i`` in ``range(config.n_layers)``:
      ``encoder.layers.{i}.window.channel`` ``[att_context_left,
      d_model]`` float32 zeros; ``encoder.layers.{i}.conv.time``
      ``[d_model, conv_kernel - 1]`` float32 zeros;
      ``encoder.layers.{i}.window.valid`` ``[1]`` int32 zeros;
    - ``frontend.raw_tail`` ``[FRONTEND_CONSTANTS["raw_tail_capacity"]]``
      float32 zeros; ``frontend.mel_tail`` ``[n_mels, 9]`` float32
      zeros; then one ``frontend.counters.{name}`` ``[1]`` int64 zeros
      per :data:`FRONTEND_COUNTER_FIELDS` name, in tuple order;
    - ``predictor.layers.0.lstm_state.h`` and ``....c``, each
      ``[pred_rnn_layers, pred_hidden]`` float32 zeros;
    - ``decode.layers.0.replay.queue``
      ``[SESSION_LIMITS["queue_capacity"]]`` int32 zeros; then one
      ``decode.layers.0.replay.book.{name}`` ``[1]`` int32 entry per
      :data:`BOOK_FIELDS` pair, in tuple order, each with ITS OWN
      ``init`` (the pair's second element — the book never initializes
      wholesale).

    ``init`` vocabulary: ``"zeros"`` | ``"blank_label"`` |
    ``"admitted_prompt"`` | ``"admitted_geometry"``.
    ``total_page_bytes`` is the exact sum over entries (the A8 pin:
    6,314,864 for the published checkpoint's geometry).

    Args:
        config: the checkpoint's ``NemotronASRConfig`` (or equivalent),
            supplying ``n_layers``, ``att_context_left``, ``d_model``,
            ``conv_kernel``, ``pred_rnn_layers``, ``pred_hidden``, and
            ``n_mels``.

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
    {"att_context": [l, r], "frames_per_chunk": r + 1,
    "raw_samples_per_chunk": RAW_SAMPLES_PER_CHUNK[label]}, ...},
    "envelope": {"version": 1, "header_fields":
    list(ENVELOPE_HEADER_FIELDS)}, "carrier_width":
    config.hidden_size, "frontend": FRONTEND_CONSTANTS}``. One cadence
    entry per :data:`CADENCES` label — the five published cadences,
    immutable per session. ``carrier_width`` must cover the header
    plus the largest admitted raw cadence
    (``len(ENVELOPE_HEADER_FIELDS) + max(RAW_SAMPLES_PER_CHUNK
    .values())``); it is derived and authored, never a copied magic
    constant.

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
    "advance_session-v1", "roles": ["CHUNK", "REPLAY", "FLUSH"],
    "chunk_roles_entering_transition": ["CHUNK"],
    "session_first_init": "metadata-books-zero-scratch",
    "encode_overlap": "pre-encode-cache-on-non-first-chunks",
    "final_tail": "actual-residual-final-stft",
    "echo_policy": "mrv1-echo-verify"}`` — the fixed contract every
    native/fallback/probe caller invokes through (PORT-ADV-001/003).

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
    """
    raise NotImplementedError(
        "author_transition_manifest lands in Phase 6 (PORT-WGT-004)"
    )


def author_emission_manifest(config: Any) -> dict[str, Any]:
    """Author the emission manifest (PORT-WGT-004).

    Schema: ``{"schema": "emission-manifest-v1", "limits":
    SESSION_LIMITS, "per_geometry": {label:
    {"max_valid_encoder_frames": int, "max_symbols_per_step":
    SESSION_LIMITS["max_symbols_per_step"], "max_emission_tokens":
    frames * symbols + 1}, ...}}`` — the PORT-INT-002 budget formula,
    ``+ 1`` for the park token, one entry per :data:`CADENCES` label
    with ``max_valid_encoder_frames = frames_per_chunk`` for that
    cadence.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
    """
    raise NotImplementedError(
        "author_emission_manifest lands in Phase 6 (PORT-WGT-004)"
    )


def author_checkpoint_profile(
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    """Author the checkpoint profile (PORT-INT-005).

    Schema: ``{"schema": "checkpoint-profile-v1", "id":
    "cp-nemotron-3.5-asr-streaming-0.6b", "precision_policy":
    PRECISION_POLICY_ID, "limits": SESSION_LIMITS,
    "state_manifest_hash": ..., "geometry_manifest_hash": ...,
    "transition_manifest_hash": ..., "emission_manifest_hash": ...,
    "content_hash": "sha256:..."}`` — ``content_hash`` is
    :func:`manifest_hash` over the profile dict WITHOUT the
    ``content_hash`` key itself (no self-reference).

    Args:
        manifest_hashes: the four manifest hashes, keyed
            ``"state"``/``"geometry"``/``"transition"``/``"emission"``.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
        ValueError: (Phase 6 behavior, pinned now) if
            ``manifest_hashes`` is missing a key or a value is not a
            well-formed ``"sha256:" + 64-hex`` string.
    """
    raise NotImplementedError(
        "author_checkpoint_profile lands in Phase 6 (PORT-INT-005)"
    )


def verify_state_manifest(
    manifest: dict[str, Any],
    page_specs: list[dict[str, Any]],
) -> None:
    """Assert the state manifest against the ACTUAL registered pages
    (PORT-STATE-009).

    The startup assertion seam. ``page_specs`` is the live model's
    registered per-session state inventory — one ``{"name", "shape",
    "dtype"}`` dict per state tensor, flattened from the semantic
    page-layer classes' emitted specs in registration order — NOT a
    handful of scalar geometry arguments, so the comparison covers
    exactly what the engine allocated. Verification raises naming the
    FIRST mismatch: an entry missing from either side, an
    out-of-order/misnamed entry, a shape or dtype disagreement, an
    unknown ``init`` value, or a ``total_page_bytes`` that disagrees
    with the summed entries or with the summed live specs.

    Args:
        manifest: the state manifest to check
            (``author_state_manifest``'s schema).
        page_specs: the live registered page inventory as described
            above.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
        ValueError: (Phase 6 behavior, pinned now) naming the first
            mismatching entry/field — never a bare mismatch count.
    """
    raise NotImplementedError(
        "verify_state_manifest lands in Phase 6 (PORT-STATE-009)"
    )
