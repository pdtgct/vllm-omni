# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-context probe boundary and realtime-only regime pins.

Source-level pins (``Path.read_text`` — no imports, so these run on any
loader): the full-context pipeline lives only in the EVAL-tier probe,
no shipped module reaches it, and RFC-1 exposes realtime only.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PACKAGE_DIR = _REPO_ROOT / "vllm_omni"
_NEMOTRON_ASR_DIR = _PACKAGE_DIR / "model_executor/models/nemotron_asr"
_PROBE = Path(__file__).resolve().parent / "p2_parity_probe.py"

#: Shipped trees a serving or entrypoint module could live in. The
#: RFC-2 ingress plugin is absent on this branch; the scan skips a
#: missing root rather than pinning the branch's tree layout.
_SHIPPED_ROOTS = ("vllm_omni", "nemotron_asr_ingress")

_PROBE_SYMBOLS = ("transcribe_full_context", "p2_parity_probe")


def _shipped_sources() -> list[Path]:
    files: list[Path] = []
    for root in _SHIPPED_ROOTS:
        tree = _REPO_ROOT / root
        if tree.is_dir():
            files.extend(sorted(tree.rglob("*.py")))
    return files


def test_full_context_pipeline_is_defined_only_in_the_probe_tier() -> None:
    # @spec PORT-REGIME-003
    probe_src = _PROBE.read_text()
    assert "def transcribe_full_context(" in probe_src
    model_src = (_NEMOTRON_ASR_DIR / "nemotron_asr.py").read_text()
    assert "def transcribe_full_context(" not in model_src


def test_no_shipped_module_reaches_the_full_context_probe() -> None:
    # @spec PORT-REGIME-003
    # The scan covers every shipped module, which strictly contains the
    # serving and entrypoint tiers the spec names.
    sources = _shipped_sources()
    assert sources, "no shipped sources found — check _SHIPPED_ROOTS"
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}: {symbol}"
        for path in sources
        for symbol in _PROBE_SYMBOLS
        if symbol in path.read_text()
    ]
    assert not offenders, f"full-context probe reachable from shipped code: {offenders}"


def test_probe_marks_the_full_context_pipeline_as_a_parity_oracle() -> None:
    # @spec PORT-REGIME-003
    # Whitespace-normalized and case-folded: the marking is a wrapped
    # docstring line that may open a sentence.
    probe_src = " ".join(_PROBE.read_text().lower().split())
    assert "eval oracle only; not an rfc-1 serving regime" in probe_src


def test_model_module_docstring_does_not_claim_full_context_serving() -> None:
    # @spec PORT-REGIME-003
    model_src = (_NEMOTRON_ASR_DIR / "nemotron_asr.py").read_text()
    head = model_src[: model_src.index('"""', model_src.index('"""') + 3)]
    assert "full-context single-shot regime backs" not in head
    compact = head.replace("``", "").replace(" ", "").lower()
    assert "supportsrealtime" in compact
    assert "supportstranscription" in compact


def test_encoder_does_not_tag_full_context_as_port_regime_002() -> None:
    # @spec PORT-REGIME-001
    encoder_src = (_NEMOTRON_ASR_DIR / "encoder.py").read_text()
    tagged = re.findall(r"PORT-REGIME-[\d/]+", encoder_src)
    assert tagged, "encoder.py must still carry its regime tag"
    assert all(tag == "PORT-REGIME-001" for tag in tagged), tagged


def _function_source(src: str, name: str) -> str:
    """Slice one top-level function's body out of the probe source.

    Cuts at the next top-level ``def`` (or EOF) so nested helper
    definitions don't bleed into an earlier function's slice.
    """
    pattern = re.compile(rf"^def {re.escape(name)}\(.*?(?=^def |\Z)", re.DOTALL | re.MULTILINE)
    match = pattern.search(src)
    assert match, f"could not locate a top-level `def {name}(` in the probe"
    return match.group(0)


def test_probe_has_exactly_one_full_context_pipeline_implementation() -> None:
    # @spec PORT-REGIME-003
    # A second `DecodeState(...)` construction is exactly what a second,
    # hand-inlined copy of the featurizer -> encoder -> LID -> decode
    # sequence would add — the two prior copies diverged on it (one
    # policy-derived, one hardcoded). Pinning the count to one means
    # there is nowhere left for that divergence to reappear.
    probe_src = _PROBE.read_text()
    assert probe_src.count("DecodeState(") == 1, (
        "expected exactly one DecodeState(...) construction in the probe; "
        f"found {probe_src.count('DecodeState(')} — a second construction "
        "site is a second, independently-drifting pipeline copy"
    )


def test_main_reaches_the_relocated_oracles_helper() -> None:
    # @spec PORT-REGIME-003
    # `main`'s golden comparison must execute the same private helper
    # `transcribe_full_context` delegates to, not a separately
    # hand-inlined copy of the same sequence.
    probe_src = _PROBE.read_text()
    oracle_body = _function_source(probe_src, "transcribe_full_context")
    main_body = _function_source(probe_src, "main")
    oracle_helper_calls = set(re.findall(r"\b(_[A-Za-z0-9_]+)\(", oracle_body))
    assert oracle_helper_calls, "transcribe_full_context calls no private helper"
    main_calls = set(re.findall(r"\b(_[A-Za-z0-9_]+)\(", main_body))
    shared = oracle_helper_calls & main_calls
    assert shared, (
        "main() shares no private helper call with transcribe_full_context "
        "— its golden comparison may be running a re-inlined, divergent "
        "copy of the pipeline instead of the relocated oracle's code path"
    )
