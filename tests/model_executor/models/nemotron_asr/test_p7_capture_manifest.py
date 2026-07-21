# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local tests for ``p7_capture_manifest.py`` (STDLIB ONLY, runs on
macOS like ``test_manifests.py``) — the delta-geometry arithmetic that
Task-7 Phase 0 (EVAL-PAR-010) flagged as the part most likely to
silently corrupt a parity comparison if wrong, so it is exercised here
independent of the pod-tier probe."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from p7_capture_manifest import (
    TENSOR_CHECKPOINTS,
    build_chunk_record,
    build_manifest,
    chunk_delta_geometry,
    sha256_file,
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_session_first_chunk_has_no_prefix_to_strip() -> None:
    # chunk_sequence == 0: prefix width is 0, so the entire mel_length
    # span is already the oracle-comparable delta (advance.py's p=0
    # branch for session-first rows).
    geom = chunk_delta_geometry(
        chunk_sequence=0, mel_length=14, encoder_length=6, is_final_tail=False
    )
    assert geom.mel_start == 0
    assert geom.new_mel_frames == 14
    assert geom.has_tensors is True


def test_continuing_chunk_strips_nine_frame_prefix() -> None:
    # A continuing chunk's mel_length INCLUDES the 9-frame pre-encode
    # cache; only the trailing frames past it are this chunk's delta.
    geom = chunk_delta_geometry(
        chunk_sequence=3, mel_length=16, encoder_length=8, is_final_tail=False
    )
    assert geom.mel_start == 9
    assert geom.new_mel_frames == 7
    assert geom.has_tensors is True


def test_zero_commit_final_tail_has_no_tensors() -> None:
    # advance.py:1293 forces mel_length to 0 on a zero-commit row
    # regardless of prefix width; the exact-cadence-multiple final tail
    # is PORT's equivalent of oracle_capture.py's synthetic terminal.
    geom = chunk_delta_geometry(
        chunk_sequence=5, mel_length=0, encoder_length=0, is_final_tail=True
    )
    assert geom.mel_start == 9  # prefix width, irrelevant once clamped
    assert geom.new_mel_frames == 0
    assert geom.has_tensors is False


def test_zero_commit_non_final_chunk_still_reports_has_tensors() -> None:
    # By construction (PORT-FEAT-003) a regular CHUNK's geometry always
    # yields >=1 new frame, so this input combination should not occur
    # in practice — but the function must not silently mirror the
    # final-tail no-op convention onto a non-final chunk, since only
    # the final tail is legitimately allowed to be zero-work.
    geom = chunk_delta_geometry(
        chunk_sequence=2, mel_length=0, encoder_length=0, is_final_tail=False
    )
    assert geom.has_tensors is True
    assert geom.new_mel_frames == 0


def test_short_final_tail_with_real_content_keeps_tensors() -> None:
    # A final tail that DID commit real (if small) content is not the
    # zero-work case — has_tensors stays True, matching the oracle's
    # "genuinely short final chunk IS the final tail" branch.
    geom = chunk_delta_geometry(
        chunk_sequence=4, mel_length=11, encoder_length=1, is_final_tail=True
    )
    assert geom.mel_start == 9
    assert geom.new_mel_frames == 2
    assert geom.has_tensors is True


def test_mel_length_below_prefix_clamps_rather_than_going_negative() -> None:
    # Defensive: a continuing chunk with mel_length < prefix width
    # should never happen (mel_length is either 0 or >= prefix per
    # advance.py's own cap_mel_len rule), but this function must fail
    # safe (clamp to 0) rather than return a negative frame count.
    geom = chunk_delta_geometry(
        chunk_sequence=1, mel_length=4, encoder_length=0, is_final_tail=False
    )
    assert geom.new_mel_frames == 0


@pytest.mark.parametrize(
    ("chunk_sequence", "mel_length", "encoder_length"),
    [(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
)
def test_negative_inputs_raise(
    chunk_sequence: int, mel_length: int, encoder_length: int
) -> None:
    with pytest.raises(ValueError):
        chunk_delta_geometry(
            chunk_sequence=chunk_sequence,
            mel_length=mel_length,
            encoder_length=encoder_length,
            is_final_tail=False,
        )


def test_build_chunk_record_has_exactly_the_required_keys() -> None:
    geom = chunk_delta_geometry(
        chunk_sequence=1, mel_length=16, encoder_length=8, is_final_tail=False
    )
    record = build_chunk_record(
        chunk_index=1, geometry=geom, encoder_length=8, is_final_tail=False
    )
    assert set(record) == {
        "chunk_index",
        "valid_feature_frames",
        "valid_encoder_frames",
        "is_final_tail",
        "has_tensors",
    }
    assert record["chunk_index"] == 1
    assert record["valid_feature_frames"] == 7
    assert record["valid_encoder_frames"] == 8
    assert record["is_final_tail"] is False
    assert record["has_tensors"] is True


def test_build_chunk_record_zeroes_encoder_frames_when_no_tensors() -> None:
    geom = chunk_delta_geometry(
        chunk_sequence=6, mel_length=0, encoder_length=0, is_final_tail=True
    )
    record = build_chunk_record(
        chunk_index=6, geometry=geom, encoder_length=0, is_final_tail=True
    )
    assert record["valid_encoder_frames"] == 0
    assert record["has_tensors"] is False


def test_build_chunk_record_rejects_negative_index() -> None:
    geom = chunk_delta_geometry(
        chunk_sequence=0, mel_length=1, encoder_length=1, is_final_tail=False
    )
    with pytest.raises(ValueError):
        build_chunk_record(
            chunk_index=-1, geometry=geom, encoder_length=1, is_final_tail=False
        )


def test_sha256_file_matches_digest_format(tmp_path: Path) -> None:
    target = tmp_path / "sample.bin"
    target.write_bytes(b"parity-capture-probe-fixture")
    digest = sha256_file(target)
    assert _DIGEST_RE.match(digest)


def test_sha256_file_is_content_stable(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")
    assert sha256_file(a) == sha256_file(b)
    c = tmp_path / "c.bin"
    c.write_bytes(b"different content")
    assert sha256_file(a) != sha256_file(c)


def test_build_manifest_contains_all_required_keys() -> None:
    geom = chunk_delta_geometry(
        chunk_sequence=0, mel_length=5, encoder_length=2, is_final_tail=True
    )
    record = build_chunk_record(
        chunk_index=0, geometry=geom, encoder_length=2, is_final_tail=True
    )
    manifest = build_manifest(
        model_revision="test-rev",
        nemo_commit="deadbeef",
        precision_policy_id="fp32-bringup",
        execution_fingerprint={"platform_backend": "cuda"},
        seed=0,
        cadence="80ms",
        clip_checksum="sha256:" + "0" * 64,
        att_context_size=(56, 0),
        tensors_digest="sha256:" + "1" * 64,
        tensors_size_bytes=123,
        partial_transcripts=["hello"],
        final_transcript="hello",
        chunk_timing_ms=[1.0],
        chunk_records=[record],
    )
    required = {
        "golden_matrix_id",
        "model_revision",
        "nemo_commit",
        "dtype",
        "precision_policy_id",
        "execution_fingerprint",
        "workload_fingerprint",
        "tensor_checkpoints",
        "tensors_digest",
        "tensors_size_bytes",
        "partial_transcripts",
        "final_transcript",
        "chunk_timing_ms",
        "chunk_records",
    }
    assert required <= set(manifest)
    assert manifest["golden_matrix_id"] is None
    assert manifest["dtype"] == "float32"
    assert manifest["tensor_checkpoints"] == list(TENSOR_CHECKPOINTS)
    assert manifest["workload_fingerprint"]["att_context_size"] == [56, 0]
    assert manifest["chunk_records"] == [record]
