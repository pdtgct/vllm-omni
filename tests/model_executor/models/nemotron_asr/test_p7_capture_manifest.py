# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local tests for ``p7_capture_manifest.py`` (STDLIB ONLY, runs on
macOS like ``test_manifests.py``) — the delta-geometry arithmetic that
Task-7 Phase 0 (EVAL-PAR-010) flagged as the part most likely to
silently corrupt a parity comparison if wrong, so it is exercised here
independent of the pod-tier probe."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from p7_capture_manifest import (
    TENSOR_CHECKPOINTS,
    build_chunk_record,
    build_manifest,
    chunk_delta_geometry,
    load_checkpoint_identity,
    sha256_file,
    validate_capture_qualification_inputs,
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


def _write_checkpoint_identity_fixture(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"derived-model")
    prompt_dictionary = {"en-US": 0, "auto": 127}
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"prompt_dictionary": prompt_dictionary, "num_prompts": 128}
        ),
        encoding="utf-8",
    )
    manifest_hashes: dict[str, str] = {}
    for name in ("state", "geometry", "transition", "emission"):
        body = {"schema": f"{name}-fixture-v1", "name": name}
        path = tmp_path / f"{name}-manifest.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        manifest_hashes[name] = _canonical_hash(body)
    profile_body = {
        "schema": "checkpoint-profile-v2",
        "id": "cp-test",
        "precision_policy": "fp32-bringup-v1",
        "limits": {},
        "source_checkpoint_digest": "sha256:" + "a" * 64,
        "converted_dump_digest": "sha256:" + "b" * 64,
        "derived_model_digest": sha256_file(model),
        "prompt_dictionary_hash": _canonical_hash(prompt_dictionary),
        **{f"{name}_manifest_hash": digest for name, digest in manifest_hashes.items()},
    }
    profile = dict(profile_body, content_hash=_canonical_hash(profile_body))
    (tmp_path / "checkpoint-profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )


def _canonical_hash(value: object) -> str:
    import hashlib

    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_checkpoint_identity_binds_source_derived_manifests_and_prompt(
    tmp_path: Path,
) -> None:
    _write_checkpoint_identity_fixture(tmp_path)
    identity = load_checkpoint_identity(tmp_path)
    assert identity.source_checkpoint_digest == "sha256:" + "a" * 64
    assert identity.converted_dump_digest == "sha256:" + "b" * 64
    assert identity.derived_model_digest == sha256_file(
        tmp_path / "model.safetensors"
    )
    assert identity.checkpoint_profile_id == "cp-test"
    assert identity.prompt_dictionary == {"en-US": 0, "auto": 127}
    assert identity.num_prompts == 128
    assert set(identity.manifest_hashes) == {
        "state",
        "geometry",
        "transition",
        "emission",
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda root: (root / "model.safetensors").write_bytes(b"changed"),
        lambda root: (root / "state-manifest.json").write_text(
            json.dumps({"schema": "changed"}), encoding="utf-8"
        ),
        lambda root: (root / "config.json").write_text(
            json.dumps(
                {"prompt_dictionary": {"en-US": 3}, "num_prompts": 128}
            ),
            encoding="utf-8",
        ),
    ),
)
def test_checkpoint_identity_rejects_content_drift(
    tmp_path: Path, mutate: object
) -> None:
    _write_checkpoint_identity_fixture(tmp_path)
    mutate(tmp_path)  # type: ignore[operator]
    with pytest.raises(ValueError):
        load_checkpoint_identity(tmp_path)


@pytest.mark.parametrize(
    "config",
    (
        {"prompt_dictionary": {"en-US": 0}},
        {"prompt_dictionary": {"en-US": 0}, "num_prompts": 0},
        {"prompt_dictionary": {"en-US": 4}, "num_prompts": 4},
    ),
)
def test_checkpoint_identity_rejects_invalid_prompt_row_bounds(
    tmp_path: Path, config: dict[str, object]
) -> None:
    _write_checkpoint_identity_fixture(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="num_prompts|rows"):
        load_checkpoint_identity(tmp_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"seed": 0}, "seed 42"),
        ({"container_image_digest": "unknown"}, "container_image_digest"),
        (
            {"container_image_digest": "unknown@sha256:" + "c" * 64},
            "container_image_digest",
        ),
        (
            {"container_image_digest": "image@sha256:" + "0" * 64},
            "container_image_digest",
        ),
        ({"venv_freeze_hash": "unknown"}, "venv_freeze_hash"),
        ({"venv_freeze_hash": "sha256:" + "0" * 64}, "venv_freeze_hash"),
        ({"source_tree_clean": False}, "clean source tree"),
        ({"cublas_workspace_config": ":16:8"}, "CUBLAS_WORKSPACE_CONFIG"),
    ),
)
def test_capture_qualification_inputs_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    inputs: dict[str, object] = {
        "seed": 42,
        "container_image_digest": "image@sha256:" + "c" * 64,
        "venv_freeze_hash": "sha256:" + "d" * 64,
        "source_tree_clean": True,
        "cublas_workspace_config": ":4096:8",
    }
    inputs.update(overrides)
    with pytest.raises(ValueError, match=message):
        validate_capture_qualification_inputs(**inputs)  # type: ignore[arg-type]


def test_capture_qualification_inputs_accept_exact_matrix_contract() -> None:
    validate_capture_qualification_inputs(
        seed=42,
        container_image_digest="image@sha256:" + "c" * 64,
        venv_freeze_hash="sha256:" + "d" * 64,
        source_tree_clean=True,
        cublas_workspace_config=":4096:8",
    )


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
