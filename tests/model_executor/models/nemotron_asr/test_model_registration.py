# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration contracts for the native Nemotron ASR model."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[4]
_MODEL_PACKAGE = _ROOT / "vllm_omni/model_executor/models/nemotron_asr"
_IDENTITY_MODULE = "vllm_omni.model_executor.models.nemotron_asr.identity"
_ARCHITECTURE = "Nemotron3_5AsrForRNNT"
_LAZY_TARGET = (
    "nemotron_asr",
    "nemotron_asr",
    "NemotronASRForRNNT",
)
_LAZY_MODULE = "vllm_omni.model_executor.models.nemotron_asr.nemotron_asr"


def _require_identity_module() -> ModuleType:
    try:
        return importlib.import_module(_IDENTITY_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _IDENTITY_MODULE:
            raise
        pytest.fail(
            "PORT-INT-001 missing dependency-light model identity module",
            pytrace=False,
        )
        raise AssertionError


def _architecture_assignments() -> list[Path]:
    assignments: list[Path] = []
    for path in sorted(_MODEL_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            targets: list[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if any(isinstance(target, ast.Name) and target.id == "ARCHITECTURE" for target in targets):
                assignments.append(path.relative_to(_MODEL_PACKAGE))
    return assignments


def test_nemotron_architecture_has_one_lightweight_authority() -> None:
    # @spec PORT-INT-001
    identity = _require_identity_module()

    assert identity.ARCHITECTURE == _ARCHITECTURE
    assert _architecture_assignments() == [Path("identity.py")]

    tree = ast.parse(Path(identity.__file__).read_text())
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "transformers" not in imported_roots
    assert "torch" not in imported_roots


def test_config_pipeline_and_registry_share_architecture_identity() -> None:
    # @spec PORT-INT-001, PORT-WGT-004
    identity = _require_identity_module()
    from vllm_omni.model_executor.models.nemotron_asr import (
        configuration_nemotron_asr,
    )
    from vllm_omni.model_executor.models.nemotron_asr.pipeline import (
        NEMOTRON_ASR_PIPELINE,
    )
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert configuration_nemotron_asr.ARCHITECTURE == identity.ARCHITECTURE
    assert NEMOTRON_ASR_PIPELINE.model_arch == identity.ARCHITECTURE
    assert _OMNI_MODELS[identity.ARCHITECTURE] == _LAZY_TARGET


def test_hf_config_registration_is_independent_of_model_registry() -> None:
    # @spec PORT-INT-001, PORT-WGT-004
    from transformers import AutoConfig

    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (
        MODEL_TYPE,
        NemotronASRConfig,
    )
    from vllm_omni.transformers_utils.configs import nemotron_asr as _registration

    assert _registration.NemotronASRConfig is NemotronASRConfig
    assert type(AutoConfig.for_model(MODEL_TYPE)) is NemotronASRConfig


def test_omni_registry_declares_exact_lazy_target() -> None:
    # @spec PORT-INT-001
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert _OMNI_MODELS.get(_ARCHITECTURE) == _LAZY_TARGET


def test_core_inspection_subprocess_resolves_native_class_idempotently() -> None:
    # @spec PORT-INT-001
    from vllm.model_executor.models.registry import _run_in_subprocess

    def inspect_registered_nemotron() -> tuple[
        bool,
        str | None,
        str | None,
        str | None,
    ]:
        """Run after core's subprocess loads installed general plugins."""
        from vllm.model_executor.models import ModelRegistry

        from vllm_omni.engine.arg_utils import register_omni_models_to_vllm

        before = ModelRegistry.models.get(_ARCHITECTURE)
        register_omni_models_to_vllm()
        register_omni_models_to_vllm()
        after = ModelRegistry.models.get(_ARCHITECTURE)
        info = ModelRegistry._try_inspect_model_cls(_ARCHITECTURE)
        return (
            before is after,
            getattr(after, "module_name", None),
            getattr(after, "class_name", None),
            None if info is None else info.architecture,
        )

    assert _run_in_subprocess(inspect_registered_nemotron) == (
        True,
        _LAZY_MODULE,
        "NemotronASRForRNNT",
        "NemotronASRForRNNT",
    )
