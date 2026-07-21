# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Task-7 Phase 1: PORT-capture -> ``GoldenManifest`` schema translation.

Pure bookkeeping only (STDLIB ONLY, no ``torch``/``vllm_omni`` import —
mirrors ``manifests.py``'s discipline so this module loads and tests on
macOS): the arithmetic that turns one committed
``advance.CaptureRecord`` (PORT's own per-chunk capture, tail-window
shaped) into one ``chunk_records`` entry and tensor-slice geometry in
the harness's ``GoldenManifest`` schema (``nemotron_omni_port.eval.
manifest``, EVAL-GOLD-003/EVAL-PAR-010). ``p7_parity_capture_probe.py``
(pod-tier: real weights, real ``advance_model_rows`` calls) is the only
caller — it does the CUDA/D2H work and hands this module plain ints,
using :func:`chunk_delta_geometry`'s result to slice/transpose the
actual tensors itself (kept out of this module to keep it torch-free).

Why translation is needed at all (EVAL-PAR-010's rationale): PORT's
``CaptureRecord.mel_length`` is NOT the oracle's ``valid_feature_frames``
delta — it is the tail-window's total valid width, which INCLUDES the
9-frame pre-encode cache prefix on every continuing chunk (never on a
session-first chunk, which carries no prior cache). Traced to source at
``advance.py:1081-1144,1289-1309`` (Task-7 Phase-0 investigation,
2026-07-20): for a continuing row, the captured ``frontend_mel``
tensor's columns ``[0, MEL_TAIL_FRAMES)`` are the unchanged prior tail
and columns ``[MEL_TAIL_FRAMES, mel_length)`` are this chunk's actual
NEW frames — exactly the oracle's ``valid_feature_frames`` quantity.
For a session-first row the prefix is zero width, so the whole
``[0, mel_length)`` span already IS the delta. ``encoder_length``
(``advance.py:1153-1160``, PRE_ENCODE_DROP already subtracted) needs no
such conversion — it already counts only this chunk's new encoder
output frames, the same quantity as the oracle's
``valid_encoder_frames``.

``has_tensors`` mirrors ``oracle_capture.py``'s ``_final_tail_records``:
the oracle sets it False only for its synthetic exact-cadence-multiple
terminal record (no real NeMo yield exists to store). PORT never
fabricates a record (PORT-DEC-004 commits a REAL, if zero-valid-length,
capture for a zero-residual final tail) — the faithful translation is
therefore ``has_tensors = not (is_final_tail and no new content)``: a
regular CHUNK's geometry (PORT-FEAT-003) always yields at least one new
frame by construction, so this can only actually fire on the final
tail.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, NamedTuple

#: The persistent pre-encode mel cache prefix every CONTINUING chunk's
#: captured ``frontend_mel`` begins with (``frontend.MEL_TAIL_FRAMES``,
#: currently 9 — advance.py:1082,1134-1144); zero for a session-first
#: chunk (``chunk_sequence == 0``), which carries no prior cache.
MEL_TAIL_FRAMES = 9

#: The named-checkpoint set every capture/golden set carries
#: (EVAL-GOLD-003), in the fixed order ``tensors.safetensors`` keys and
#: ``GoldenManifest.tensor_checkpoints`` both use.
TENSOR_CHECKPOINTS: tuple[str, ...] = (
    "frontend_mel",
    "encoder_raw",
    "encoder_conditioned",
)

#: The five PORT-EVAL cadence labels, matching ``manifests.CADENCES``'
#: iteration order (geometry_id 0..4) — duplicated here as plain
#: strings only (no torch/vllm_omni import) so this module stays
#: loader-safe; the probe cross-checks against the real
#: ``manifests.CADENCES`` at import time instead of trusting this list
#: alone to stay in sync.
CADENCE_LABELS: tuple[str, ...] = ("80ms", "160ms", "320ms", "560ms", "1120ms")


class ChunkDeltaGeometry(NamedTuple):
    """Where in a PORT ``CaptureRecord``'s tail-window tensors this
    chunk's oracle-comparable NEW content starts, and how much of it
    there is."""

    mel_start: int
    """First valid column of ``frontend_mel`` to keep (0, or
    :data:`MEL_TAIL_FRAMES` on a continuing chunk)."""

    new_mel_frames: int
    """Oracle-comparable ``valid_feature_frames``: the count of NEW mel
    columns this chunk committed, excluding the pre-encode prefix."""

    has_tensors: bool
    """Whether this chunk actually has real, oracle-comparable
    tensor content to write (EVAL-PAR-010) — False only for the
    zero-work final tail."""


def chunk_delta_geometry(
    *,
    chunk_sequence: int,
    mel_length: int,
    encoder_length: int,
    is_final_tail: bool,
    mel_tail_frames: int = MEL_TAIL_FRAMES,
) -> ChunkDeltaGeometry:
    """Derive one chunk's delta-slice geometry from its committed
    ``CaptureRecord`` fields.

    Args:
        chunk_sequence: The record's ``chunk_sequence`` (0 for a
            session-first CHUNK — PORT-ADV/SESS numbering, not an
            index into any accumulated list).
        mel_length: The record's ``mel_length`` — the tail-window's
            total valid width (prefix included on continuing chunks;
            forced to 0 on a zero-commit chunk, ``advance.py:1293``).
        encoder_length: The record's ``encoder_length`` — already the
            NEW encoder output frame count this chunk (no conversion
            needed, ``advance.py:1153-1160``).
        is_final_tail: Whether this is the session's one final-tail
            CHUNK (host-tracked by the caller/probe, never derived
            from the record).
        mel_tail_frames: The pre-encode cache prefix width (override
            only for testing; production callers use the default).

    Returns:
        The slice/skip geometry plus the ``has_tensors`` disposition.

    Raises:
        ValueError: On a negative input (a malformed record — this
            function never clamps its own inputs, only the DERIVED
            frame count, so a defect upstream is never silently
            absorbed).
    """
    if chunk_sequence < 0:
        raise ValueError(f"chunk_sequence must be >= 0, got {chunk_sequence}")
    if mel_length < 0:
        raise ValueError(f"mel_length must be >= 0, got {mel_length}")
    if encoder_length < 0:
        raise ValueError(f"encoder_length must be >= 0, got {encoder_length}")
    is_session_first = chunk_sequence == 0
    prefix = 0 if is_session_first else mel_tail_frames
    # Clamped, not asserted: a zero-commit chunk stages mel_length == 0
    # regardless of prefix width (advance.py:1293's cap_mel_len rule),
    # so mel_length < prefix is an EXPECTED state, not a defect.
    new_mel_frames = max(mel_length - prefix, 0)
    has_tensors = not (is_final_tail and new_mel_frames == 0 and encoder_length == 0)
    return ChunkDeltaGeometry(
        mel_start=prefix, new_mel_frames=new_mel_frames, has_tensors=has_tensors
    )


def build_chunk_record(
    *,
    chunk_index: int,
    geometry: ChunkDeltaGeometry,
    encoder_length: int,
    is_final_tail: bool,
) -> dict[str, Any]:
    """Assemble one ``chunk_records`` entry
    (``manifest.py``'s ``_CHUNK_RECORD_REQUIRED_KEYS``).

    Args:
        chunk_index: 0-based position in this session's ordered
            capture sequence (assigned by the caller — the PORT
            ``chunk_sequence`` counter and this index coincide for a
            single, uninterrupted session, but the caller owns the
            assignment rather than this function re-deriving it).
        geometry: This chunk's :func:`chunk_delta_geometry` result.
        encoder_length: The record's raw ``encoder_length`` (needs no
            delta conversion, unlike ``mel_length`` — see module
            docstring); reported as 0 when ``geometry.has_tensors`` is
            False, matching the oracle's synthetic-terminal
            convention (``oracle_capture.py``'s zero-work record).
        is_final_tail: Carried straight into the record.
    """
    if chunk_index < 0:
        raise ValueError(f"chunk_index must be >= 0, got {chunk_index}")
    return {
        "chunk_index": chunk_index,
        "valid_feature_frames": geometry.new_mel_frames,
        "valid_encoder_frames": encoder_length if geometry.has_tensors else 0,
        "is_final_tail": is_final_tail,
        "has_tensors": geometry.has_tensors,
    }


def sha256_file(path: Path) -> str:
    """``sha256:<64hex>`` over a file's exact bytes — the digest format
    ``manifest.py``'s ``_TENSORS_DIGEST_RE`` requires, reused here for
    both ``tensors_digest`` and clip/model-artifact checksums."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def build_manifest(
    *,
    model_revision: str,
    nemo_commit: str,
    precision_policy_id: str,
    execution_fingerprint: dict[str, Any],
    seed: int,
    cadence: str,
    clip_checksum: str,
    att_context_size: tuple[int, int],
    tensors_digest: str,
    tensors_size_bytes: int,
    partial_transcripts: list[str],
    final_transcript: str,
    chunk_timing_ms: list[float],
    chunk_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the full ``manifest.json`` dict
    (``manifest.py``'s ``_REQUIRED_KEYS``/``_WORKLOAD_REQUIRED_KEYS``).

    ``golden_matrix_id`` is always ``None`` here, matching
    ``oracle_capture.py``'s own run-capture manifests: matrix adoption
    is a separate, later pinning step (EVAL-GOLD-007), never something
    a capture-producing probe claims for itself. ``dtype`` is always
    ``"float32"`` — PORT-PREC-003's sole oracle-correctness lane.
    """
    return {
        "golden_matrix_id": None,
        "model_revision": model_revision,
        "nemo_commit": nemo_commit,
        "dtype": "float32",
        "precision_policy_id": precision_policy_id,
        "execution_fingerprint": execution_fingerprint,
        "workload_fingerprint": {
            "seed": seed,
            "cadence": cadence,
            "clip_checksum": clip_checksum,
            "att_context_size": list(att_context_size),
        },
        "tensor_checkpoints": list(TENSOR_CHECKPOINTS),
        "tensors_digest": tensors_digest,
        "tensors_size_bytes": tensors_size_bytes,
        "partial_transcripts": list(partial_transcripts),
        "final_transcript": final_transcript,
        "chunk_timing_ms": list(chunk_timing_ms),
        "chunk_records": chunk_records,
    }
