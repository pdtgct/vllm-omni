# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for Omni-owned HTTP launcher lifecycle hooks."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
from vllm.entrypoints.launcher import serve_http as upstream_serve_http

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[3]
_API_SERVER_PATH = _ROOT / "vllm_omni/entrypoints/openai/api_server.py"
_LAUNCHER_PATH = _ROOT / "vllm_omni/entrypoints/launcher.py"
_MIRRORED_SURFACE = {
    "engine_shutdown",
    "h11_limits",
    "port_conflict_diagnostics",
    "route_logging",
    "signal_cleanup",
    "ssl_refresh",
    "uvicorn_configuration",
    "watchdog",
}


def _load_launcher():
    assert _LAUNCHER_PATH.exists(), "Omni-owned launcher is not implemented"
    spec = importlib.util.spec_from_file_location(
        "_omni_launcher_under_test",
        _LAUNCHER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_api_server_imports_the_omni_owned_launcher() -> None:
    """The activation unit has no vLLM-core launcher dependency."""
    # @spec ING-VEH-017
    tree = ast.parse(_API_SERVER_PATH.read_text(encoding="utf-8"))
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert (
        "vllm_omni.entrypoints.launcher",
        "serve_http",
    ) in imports
    assert ("vllm.entrypoints.launcher", "serve_http") not in imports


def test_launcher_adds_only_the_optional_lifecycle_hook_to_the_public_shape() -> None:
    """The no-hook call shape stays paired with vLLM's launcher."""
    # @spec ING-VEH-017
    launcher = _load_launcher()
    omni_parameters = inspect.signature(launcher.serve_http).parameters
    upstream_parameters = inspect.signature(upstream_serve_http).parameters

    assert "lifecycle_hook" in omni_parameters
    assert omni_parameters["lifecycle_hook"].default is None
    assert {
        name: parameter
        for name, parameter in omni_parameters.items()
        if name != "lifecycle_hook"
    } == dict(upstream_parameters)


def test_launcher_hook_names_the_shutdown_linearization_and_barriers() -> None:
    """The interface separates synchronous close from awaited barriers."""
    # @spec ING-VEH-017, ING-VEH-022
    launcher = _load_launcher()
    hook = launcher.ApplicationLifecycleHook
    names = {
        name
        for name, member in inspect.getmembers(hook)
        if inspect.isfunction(member)
    }

    assert {
        "on_bound",
        "on_shutdown_requested",
        "before_http_shutdown",
        "before_engine_shutdown",
    } <= names
    assert not inspect.iscoroutinefunction(hook.on_shutdown_requested)
    assert inspect.iscoroutinefunction(hook.on_bound)
    assert inspect.iscoroutinefunction(hook.before_http_shutdown)
    assert inspect.iscoroutinefunction(hook.before_engine_shutdown)


def test_launcher_declares_the_complete_pin_bump_rediff_surface() -> None:
    """A pin bump cannot silently omit one mirrored launcher behavior."""
    # @spec ING-VEH-017
    launcher = _load_launcher()

    assert set(launcher.MIRRORED_UPSTREAM_BEHAVIORS) == _MIRRORED_SURFACE
