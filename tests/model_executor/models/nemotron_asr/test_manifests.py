# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: checkpoint-profile manifests.

``manifests.py`` is STDLIB-ONLY (no ``torch``, no ``vllm_omni`` package
import), so — unlike the rest of the port — this file loads it by
direct file location (``importlib.util.spec_from_file_location``),
bypassing the ``vllm_omni`` package ``__init__`` chain (which pulls
``vllm``, uninstallable on macOS). That is what lets this file run
locally; every other nemotron_asr test file is pod-tier.

``canonical_json``/``manifest_hash``/``CADENCES`` are real, PASSING
tests. ``author_*`` and ``verify_state_manifest`` are tests-first
stubs (Phase 5): calls raise ``NotImplementedError`` until Phase 6,
which is the recorded expected-fail evidence for those cases.

The canonical layout below pins the DECIDED design inventory
(PORT-STATE-001/009, ``port-design.md``'s state-page table): total
6,314,864 bytes (~6.0223 MiB) FP32 bring-up. Current pool code totals
6,302,368 bytes — Phase 6 owes exactly the difference before these
tests can go green against a live model:

- the frontend-continuity state page (PORT-STATE-001; no
  ``state_layers.py`` class yet): a 1,953-sample raw audio tail
  (fp32) + a ``(128, 9)`` mel tail (fp32) + eight int64 control
  counters = ``1953*4 + 128*9*4 + 8*8`` = 12,484 bytes;
- three more int32 session-book fields (the decided 7-field book —
  admitted geometry, pending-echo flag, expected label — beyond the
  current 4: ``rnnt.QUEUE_HEAD/QUEUE_LEN/QUEUE_LAST_LABEL/
  QUEUE_PROMPT``): ``3 * 4`` = 12 bytes;
- dtype alignment: the valid-length scalar, queue, and book are
  int32 in the design manifest (byte-identical to today's
  float-pool storage, so only naming, not size, changes).

``12,484 + 12 == 12,496``; ``6,302,368 + 12,496 == 6,314,864``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_MANIFESTS_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/nemotron_asr/manifests.py"
)


def _load_manifests() -> Any:
    """Load ``manifests.py`` by file path — no ``vllm_omni`` import."""
    spec = importlib.util.spec_from_file_location(
        "nemotron_asr_manifests_under_test", _MANIFESTS_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifests = _load_manifests()

# ---- the DESIGN canonical FP32 bring-up layout ----------------------------
#
# Geometry from the fork's live pool construction, extended by the
# decided design inventory (port-design.md state-page table):
#   - encoder.py FastConformerEncoder(n_layers=24) (nemotron_asr.py's
#     NemotronASRCore default and state_layers.py's n_layers source);
#   - configuration_nemotron_asr.py: d_model=1024, conv_kernel=9,
#     att_context_left=56, pred_hidden=640, pred_rnn_layers=2;
#   - rnnt.py: MAX_SYMBOLS_PER_STEP=10;
#   - nemotron_asr.py: _MAX_FRAMES_PER_CHUNK=14 → queue capacity
#     10*14=140 (ReplayQueuePage);
#   - the DESIGN 7-slot session book (head, len, last label, prompt,
#     admitted geometry, pending-echo flag, expected label);
#   - the frontend-continuity page: raw tail capacity
#     pre_encode_cache(9) × hop(160) + n_fft(512) + 1 = 1,953 samples,
#     a (128, 9) mel tail, eight int64 control counters.
_N_LAYERS = 24
_WINDOW = 56
_D_MODEL = 1024
_CONV_TAIL = 8  # conv_kernel(9) - 1
_PRED_LAYERS = 2
_PRED_HIDDEN = 640
_QUEUE_CAPACITY = 140  # MAX_SYMBOLS_PER_STEP(10) * _MAX_FRAMES_PER_CHUNK(14)
_BOOK_WIDTH = 7  # design book: 4 current fields + geometry/echo/expected
_RAW_TAIL = 1953  # pre_encode_cache(9) * hop(160) + n_fft(512) + 1
_MEL_TAIL = (128, 9)
_FRONTEND_COUNTERS = 8  # int64
_FP32 = 4  # bytes


def _canonical_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for i in range(_N_LAYERS):
        entries.append({
            "name": f"encoder.layers.{i}.window.channel",
            "shape": [_WINDOW, _D_MODEL],
            "dtype": "float32",
            "init": "zeros",
        })
        entries.append({
            "name": f"encoder.layers.{i}.conv.time",
            "shape": [_D_MODEL, _CONV_TAIL],
            "dtype": "float32",
            "init": "zeros",
        })
        entries.append({
            "name": f"encoder.layers.{i}.window.valid",
            "shape": [1],
            "dtype": "int32",
            "init": "zeros",
        })
    entries.append({
        "name": "frontend.raw_tail",
        "shape": [_RAW_TAIL],
        "dtype": "float32",
        "init": "zeros",
    })
    entries.append({
        "name": "frontend.mel_tail",
        "shape": list(_MEL_TAIL),
        "dtype": "float32",
        "init": "zeros",
    })
    entries.append({
        "name": "frontend.counters",
        "shape": [_FRONTEND_COUNTERS],
        "dtype": "int64",
        "init": "zeros",
    })
    entries.append({
        "name": "predictor.layers.0.lstm_state.h",
        "shape": [_PRED_LAYERS, _PRED_HIDDEN],
        "dtype": "float32",
        "init": "zeros",
    })
    entries.append({
        "name": "predictor.layers.0.lstm_state.c",
        "shape": [_PRED_LAYERS, _PRED_HIDDEN],
        "dtype": "float32",
        "init": "zeros",
    })
    entries.append({
        "name": "decode.layers.0.replay.queue",
        "shape": [_QUEUE_CAPACITY],
        "dtype": "int32",
        "init": "zeros",
    })
    entries.append({
        "name": "decode.layers.0.replay.book",
        "shape": [_BOOK_WIDTH],
        "dtype": "int32",
        "init": "blank_label",
    })
    return entries


_DTYPE_BYTES = {"float32": 4, "int32": 4, "int64": 8}


def _entry_bytes(entry: dict[str, Any]) -> int:
    size = _DTYPE_BYTES[entry["dtype"]]
    n = 1
    for dim in entry["shape"]:
        n *= dim
    return n * size


def _canonical_total_page_bytes() -> int:
    return sum(_entry_bytes(e) for e in _canonical_entries())


def _canonical_state_manifest() -> dict[str, Any]:
    return {
        "schema": "state-manifest-v1",
        "entries": _canonical_entries(),
        "total_page_bytes": _canonical_total_page_bytes(),
    }


def test_canonical_fp32_layout_arithmetic_is_auditable() -> None:
    # @spec PORT-STATE-009
    # Per-layer: channel (56*1024*4=229376) + time (1024*8*4=32768) +
    # valid (1*4=4) = 262148 bytes/layer; * 24 layers = 6,291,552.
    per_layer = _WINDOW * _D_MODEL * _FP32 + _D_MODEL * _CONV_TAIL * _FP32 + _FP32
    assert per_layer == 262_148
    window_and_conv_total = per_layer * _N_LAYERS
    assert window_and_conv_total == 6_291_552

    # Predictor h/c: 2*640*4 = 5120 bytes each; two tensors = 10,240.
    predictor_total = 2 * (_PRED_LAYERS * _PRED_HIDDEN * _FP32)
    assert predictor_total == 10_240

    # Emission queue: 140*4 = 560; the DESIGN 7-slot book: 7*4 = 28.
    queue_total = _QUEUE_CAPACITY * 4
    book_total = _BOOK_WIDTH * 4
    assert queue_total == 560
    assert book_total == 28

    # Frontend continuity: 1953*4 + 128*9*4 + 8*8 = 12,484.
    frontend_total = (
        _RAW_TAIL * _FP32
        + _MEL_TAIL[0] * _MEL_TAIL[1] * _FP32
        + _FRONTEND_COUNTERS * 8
    )
    assert frontend_total == 12_484

    total = (
        window_and_conv_total
        + predictor_total
        + queue_total
        + book_total
        + frontend_total
    )
    # The PORT-STATE-009 / A8-decided FP32 bring-up pin (port-design.md
    # state-page table). Current pool code totals 6,302,368 — Phase 6
    # owes the frontend page + 3 book slots (module docstring).
    assert total == 6_314_864
    assert total == _canonical_total_page_bytes()


# ---- canonical_json / manifest_hash: real, PASSING helpers ----------------


def test_canonical_json_is_deterministic_regardless_of_key_order() -> None:
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert manifests.canonical_json(a) == manifests.canonical_json(b)


def test_canonical_json_is_compact_and_ascii() -> None:
    s = manifests.canonical_json({"x": [1, 2], "y": "café"})
    assert " " not in s  # compact separators
    assert "\\u00e9" in s  # ensure_ascii


def test_manifest_hash_shape_is_sha256_prefixed_hex() -> None:
    h = manifests.manifest_hash({"a": 1})
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64
    assert all(c in "0123456789abcdef" for c in h[len("sha256:"):])


def test_manifest_hash_is_deterministic_and_content_sensitive() -> None:
    h1 = manifests.manifest_hash({"a": 1, "b": 2})
    h2 = manifests.manifest_hash({"b": 2, "a": 1})
    h3 = manifests.manifest_hash({"a": 1, "b": 3})
    assert h1 == h2
    assert h1 != h3


# ---- five-cadence completeness: local-pass ---------------------------------


def test_cadences_are_the_five_published_pairs_exactly() -> None:
    assert manifests.CADENCES == {
        "80ms": (56, 0),
        "160ms": (56, 1),
        "320ms": (56, 3),
        "560ms": (56, 6),
        "1120ms": (56, 13),
    }


# ---- PORT-WGT-004: author_* stubs (expected-fail evidence) -----------------


@pytest.mark.parametrize(
    "fn_name",
    [
        "author_state_manifest",
        "author_geometry_manifest",
        "author_transition_manifest",
        "author_emission_manifest",
    ],
)
def test_author_functions_pin_their_schema_and_fail_until_phase_6(
    fn_name: str,
) -> None:
    # @spec PORT-WGT-004
    fn = getattr(manifests, fn_name)
    config = types.SimpleNamespace()
    with pytest.raises(NotImplementedError):
        fn(config)


# ---- PORT-STATE-009: verify_state_manifest stub ----------------------------


def test_verify_state_manifest_accepts_the_canonical_layout_stub() -> None:
    # @spec PORT-STATE-009
    # Written against the real accept contract; fails NotImplementedError
    # on the stub until Phase 6 (expected-fail evidence) — see the module
    # docstring for why this canonical layout's total is 6,302,368, not
    # the final 6,314,864 pin.
    with pytest.raises(NotImplementedError):
        manifests.verify_state_manifest(
            _canonical_state_manifest(),
            n_layers=_N_LAYERS,
            window=_WINDOW,
            feat=128,
            hidden=_D_MODEL,
            queue_capacity=_QUEUE_CAPACITY,
            dtype="float32",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["entries"].pop(),  # missing entry
        lambda m: m["entries"].__setitem__(
            0, {**m["entries"][0], "shape": [1, 1]}
        ),  # wrong shape
        lambda m: m["entries"].__setitem__(
            0, {**m["entries"][0], "dtype": "float16"}
        ),  # wrong dtype
        lambda m: m.__setitem__("total_page_bytes", 1),  # wrong total
    ],
    ids=["missing_entry", "wrong_shape", "wrong_dtype", "wrong_total_bytes"],
)
def test_verify_state_manifest_rejects_each_defect_class(
    mutate: Any,
) -> None:
    # @spec PORT-STATE-009
    manifest = _canonical_state_manifest()
    mutate(manifest)
    # Phase 6: each defect raises ValueError naming the offending
    # entry/field. Today the stub raises NotImplementedError first —
    # expected-fail evidence, not a claim that ValueError is observed.
    with pytest.raises(NotImplementedError):
        manifests.verify_state_manifest(
            manifest,
            n_layers=_N_LAYERS,
            window=_WINDOW,
            feat=128,
            hidden=_D_MODEL,
            queue_capacity=_QUEUE_CAPACITY,
            dtype="float32",
        )


# ---- PORT-INT-005: checkpoint-profile content-hash recomputation ----------


def test_checkpoint_profile_content_hash_recomputation_matches() -> None:
    # @spec PORT-INT-005
    # Exercises manifest_hash/canonical_json directly (both real, not
    # stubs) — author_checkpoint_profile itself is a Phase-5 stub, but
    # the recomputation property it must satisfy is checkable now.
    profile_body = {
        "schema": "checkpoint-profile-v1",
        "id": "cp-nemotron-3.5-asr-streaming-0.6b",
        "state_manifest_hash": "sha256:" + "1" * 64,
        "geometry_manifest_hash": "sha256:" + "2" * 64,
        "transition_manifest_hash": "sha256:" + "3" * 64,
        "emission_manifest_hash": "sha256:" + "4" * 64,
    }
    recomputed = manifests.manifest_hash(profile_body)
    assert recomputed == manifests.manifest_hash(dict(profile_body))


def test_checkpoint_profile_content_hash_is_sensitive_to_each_manifest_hash() -> (
    None
):
    # @spec PORT-INT-005
    base = {
        "schema": "checkpoint-profile-v1",
        "id": "cp-nemotron-3.5-asr-streaming-0.6b",
        "state_manifest_hash": "sha256:" + "1" * 64,
        "geometry_manifest_hash": "sha256:" + "2" * 64,
        "transition_manifest_hash": "sha256:" + "3" * 64,
        "emission_manifest_hash": "sha256:" + "4" * 64,
    }
    base_hash = manifests.manifest_hash(base)
    for key in (
        "state_manifest_hash",
        "geometry_manifest_hash",
        "transition_manifest_hash",
        "emission_manifest_hash",
    ):
        changed = dict(base, **{key: "sha256:" + "9" * 64})
        assert manifests.manifest_hash(changed) != base_hash
