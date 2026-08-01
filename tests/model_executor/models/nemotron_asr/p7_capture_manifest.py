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
import json
import re
from dataclasses import dataclass
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

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_DIGEST_RE = re.compile(r"^(?P<repository>[^\s@]+)@sha256:(?P<digest>[0-9a-f]{64})$")
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_MANIFEST_FILES = {
    "state": "state-manifest.json",
    "geometry": "geometry-manifest.json",
    "transition": "transition-manifest.json",
    "emission": "emission-manifest.json",
}


@dataclass(frozen=True)
class CheckpointIdentity:
    """Verified identity fields consumed by the parity fingerprint."""

    source_checkpoint_digest: str
    converted_dump_digest: str
    derived_model_digest: str
    checkpoint_profile_id: str
    checkpoint_profile_hash: str
    manifest_hashes: dict[str, str]
    prompt_dictionary: dict[str, int]
    num_prompts: int


def validate_capture_qualification_inputs(
    *,
    seed: int,
    container_image_digest: str,
    venv_freeze_hash: str,
    source_tree_clean: bool,
    cublas_workspace_config: str | None,
) -> None:
    """Fail closed on the matrix's qualifying capture inputs.

    This is intentionally loader-safe so the negative contract can run
    without CUDA. Exploratory captures belong in a different probe; this
    Task-7 command produces only adoption-eligible evidence.

    Raises:
        ValueError: If any qualifying input is missing or inconsistent.
    """
    if seed != 42:
        raise ValueError(f"qualifying P7 capture requires seed 42, got {seed}")
    container_match = _CONTAINER_DIGEST_RE.fullmatch(container_image_digest)
    if (
        container_match is None
        or container_match.group("repository").lower() in {"unknown", "none", "placeholder"}
        or int(container_match.group("digest"), 16) == 0
    ):
        raise ValueError("container_image_digest must be a non-placeholder repo@sha256:<64-hex> identity")
    if not _SHA256_RE.fullmatch(venv_freeze_hash) or int(venv_freeze_hash.removeprefix("sha256:"), 16) == 0:
        raise ValueError("venv_freeze_hash must be a non-placeholder sha256:<64-hex>")
    if not source_tree_clean:
        raise ValueError("qualifying P7 capture requires a clean source tree")
    if cublas_workspace_config != _CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(f"CUBLAS_WORKSPACE_CONFIG must equal {_CUBLAS_WORKSPACE_CONFIG!r}")


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
    return ChunkDeltaGeometry(mel_start=prefix, new_mel_frames=new_mel_frames, has_tensors=has_tensors)


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


def _canonical_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def load_checkpoint_identity(checkpoint_dir: Path) -> CheckpointIdentity:
    """Verify and return every checkpoint-bound fingerprint input.

    The source checkpoint is not expected to be present in the served
    directory; its digest is authored into the checkpoint profile by
    ``publish.py``. Every derived file that is present is rehashed here
    before capture, so a stale or edited artifact fails before any GPU
    work can produce apparently comparable evidence.
    """
    profile_path = checkpoint_dir / "checkpoint-profile.json"
    profile = _load_object(profile_path)
    profile_hash = profile.get("content_hash")
    if not isinstance(profile_hash, str) or not _SHA256_RE.fullmatch(profile_hash):
        raise ValueError(f"{profile_path} has malformed content_hash")
    body = {key: value for key, value in profile.items() if key != "content_hash"}
    if _canonical_hash(body) != profile_hash:
        raise ValueError(f"{profile_path} content_hash does not match its content")

    required = {
        "schema",
        "id",
        "precision_policy",
        "limits",
        "source_checkpoint_digest",
        "converted_dump_digest",
        "derived_model_digest",
        "prompt_dictionary_hash",
        *(f"{name}_manifest_hash" for name in _MANIFEST_FILES),
    }
    missing = sorted(required - set(body))
    if missing:
        raise ValueError(f"{profile_path} missing required fields: {missing}")
    if body["schema"] != "checkpoint-profile-v2":
        raise ValueError(f"{profile_path} schema must be 'checkpoint-profile-v2'")
    if body["precision_policy"] != "fp32-bringup-v1":
        raise ValueError(f"{profile_path} must bind precision_policy 'fp32-bringup-v1'")

    manifest_hashes: dict[str, str] = {}
    for name, filename in _MANIFEST_FILES.items():
        expected = profile[f"{name}_manifest_hash"]
        actual = _canonical_hash(_load_object(checkpoint_dir / filename))
        if expected != actual:
            raise ValueError(f"{filename} hash mismatch: profile={expected!r}, actual={actual!r}")
        manifest_hashes[name] = actual

    derived_model_digest = sha256_file(checkpoint_dir / "model.safetensors")
    if profile["derived_model_digest"] != derived_model_digest:
        raise ValueError("model.safetensors digest disagrees with checkpoint profile")

    config = _load_object(checkpoint_dir / "config.json")
    num_prompts = config.get("num_prompts")
    if isinstance(num_prompts, bool) or not isinstance(num_prompts, int) or num_prompts <= 0:
        raise ValueError("config.json num_prompts must be a positive integer")
    prompt_value = config.get("prompt_dictionary")
    if not isinstance(prompt_value, dict) or not prompt_value:
        raise ValueError("config.json must carry a non-empty prompt_dictionary")
    prompt_dictionary: dict[str, int] = {}
    for locale, index in prompt_value.items():
        if (
            not isinstance(locale, str)
            or not locale
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < num_prompts
        ):
            raise ValueError(f"prompt_dictionary must map non-empty strings to rows in [0, {num_prompts})")
        prompt_dictionary[locale] = index
    prompt_hash = _canonical_hash(prompt_dictionary)
    if profile["prompt_dictionary_hash"] != prompt_hash:
        raise ValueError("prompt_dictionary hash disagrees with checkpoint profile")

    source_digest = profile["source_checkpoint_digest"]
    if not isinstance(source_digest, str) or not _SHA256_RE.fullmatch(source_digest):
        raise ValueError("source_checkpoint_digest is missing or malformed")
    converted_dump_digest = profile["converted_dump_digest"]
    if not isinstance(converted_dump_digest, str) or not _SHA256_RE.fullmatch(converted_dump_digest):
        raise ValueError("converted_dump_digest is missing or malformed")
    profile_id = profile["id"]
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("checkpoint profile id must be a non-empty string")
    return CheckpointIdentity(
        source_checkpoint_digest=source_digest,
        converted_dump_digest=converted_dump_digest,
        derived_model_digest=derived_model_digest,
        checkpoint_profile_id=profile_id,
        checkpoint_profile_hash=profile_hash,
        manifest_hashes=manifest_hashes,
        prompt_dictionary=prompt_dictionary,
        num_prompts=num_prompts,
    )


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
