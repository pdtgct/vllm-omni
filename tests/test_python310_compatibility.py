# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executable Python 3.10 source and import compatibility contract."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (ROOT / "vllm_omni", ROOT / "apps")

PY311_MODULES = frozenset({"tomllib"})
PY311_SYMBOLS: dict[str, frozenset[str]] = {
    "asyncio": frozenset({"TaskGroup", "timeout", "timeout_at"}),
    "datetime": frozenset({"UTC"}),
    "enum": frozenset(
        {
            "FlagBoundary",
            "ReprEnum",
            "StrEnum",
            "global_enum",
            "member",
            "nonmember",
            "show_flag_values",
            "verify",
        }
    ),
    "typing": frozenset(
        {
            "LiteralString",
            "Never",
            "NotRequired",
            "Required",
            "Self",
            "TypeVarTuple",
            "Unpack",
            "assert_never",
            "assert_type",
            "clear_overloads",
            "dataclass_transform",
            "get_overloads",
            "reveal_type",
        }
    ),
}


def _python311_references(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename, feature_version=(3, 10))
    module_aliases: dict[str, str] = {}
    findings: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                root_module = imported.name.split(".", 1)[0]
                local_name = imported.asname or root_module
                module_aliases[local_name] = root_module
                if root_module in PY311_MODULES:
                    findings.append(
                        f"{filename}:{node.lineno}: import {root_module}"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            root_module = node.module.split(".", 1)[0]
            if root_module in PY311_MODULES:
                findings.append(f"{filename}:{node.lineno}: from {node.module}")
            forbidden = PY311_SYMBOLS.get(root_module, frozenset())
            for imported in node.names:
                if imported.name in forbidden:
                    findings.append(
                        f"{filename}:{node.lineno}: "
                        f"from {root_module} import {imported.name}"
                    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(
            node.value, ast.Name
        ):
            continue
        module = module_aliases.get(node.value.id)
        if module is not None and node.attr in PY311_SYMBOLS.get(
            module, frozenset()
        ):
            findings.append(
                f"{filename}:{node.lineno}: {node.value.id}.{node.attr}"
            )

    return sorted(set(findings))


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# @spec ENV-MIG-004
def test_repository_source_targets_python310_import_surface() -> None:
    findings: list[str] = []
    for source_root in SOURCE_ROOTS:
        for path in sorted(source_root.rglob("*.py")):
            findings.extend(
                _python311_references(
                    path.read_text(encoding="utf-8"),
                    str(path.relative_to(ROOT)),
                )
            )
    assert not findings, "\n" + "\n".join(findings)


# @spec ENV-MIG-004
@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("source", "expected"),
    [
        (
            "from typing import (\n    Any,\n    NotRequired,\n)\n",
            "from typing import NotRequired",
        ),
        ("import enum as enums\nenums.StrEnum\n", "enums.StrEnum"),
        ("import tomllib\n", "import tomllib"),
    ],
)
def test_python310_scan_rejects_runtime_incompatibilities(
    source: str, expected: str
) -> None:
    assert any(
        expected in finding
        for finding in _python311_references(source, "probe.py")
    )


# @spec ENV-MIG-004
def test_python310_compatibility_modules_import_with_stable_semantics() -> None:
    output_modality = _load_source_module(
        "python310_output_modality",
        ROOT / "vllm_omni" / "engine" / "output_modality.py",
    )
    comfy_types = _load_source_module(
        "python310_comfy_types",
        ROOT
        / "apps"
        / "ComfyUI-vLLM-Omni"
        / "comfyui_vllm_omni"
        / "utils"
        / "types.py",
    )

    assert str(output_modality.OutputModalityNames.TEXT) == "text"
    assert "payload_preprocessor" in comfy_types.Spec.__optional_keys__
