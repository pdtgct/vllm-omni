# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the generic router-plugin hook (PORT-PLUG-001..003).

These tests exercise the hook GPU-free against a duck-typed app. They
never import `vllm_omni.entrypoints.openai.api_server` (which pulls in
vllm and an engine); the loader module is deliberately import-light, so
it is loaded BY FILE PATH under a bare name -- the same technique the
session/orchestrator tests use -- so a plain `pytest` run collects this
file without `vllm` on the box (importing it through the
`vllm_omni.entrypoints.openai` package would trigger the vllm-patching
`vllm_omni/__init__`). The graduated specs now own the PORT-PLUG-* IDs
(port-specs.md §vLLM-Omni integration).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

_MODULE_PATH = Path(
    os.environ.get("ROUTER_PLUGINS_PATH")
    or (
        Path(__file__).resolve().parents[2]
        / "vllm_omni/entrypoints/openai/router_plugins.py"
    )
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "router_plugins_under_test", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_RP = _load_module()
ROUTER_PLUGINS_GROUP = _RP.ROUTER_PLUGINS_GROUP
load_router_plugins = _RP.load_router_plugins

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODULE = "router_plugins_under_test"


class StubApp:
    """Duck-typed stand-in for the FastAPI app handed to `register`."""

    def __init__(self) -> None:
        self.state = SimpleNamespace()
        self.included: list[Any] = []

    def include_router(self, router: Any) -> None:
        self.included.append(router)


class StubEntryPoint:
    """Entry point whose `load()` returns (or raises) whatever a test needs."""

    def __init__(self, name: str, result: Any = None, error: BaseException | None = None) -> None:
        self.name = name
        self.value = f"tests.stub:{name}"
        self._result = result
        self._error = error
        self.load_calls = 0

    def load(self) -> Any:
        self.load_calls += 1
        if self._error is not None:
            raise self._error
        return self._result


InstallPlugins = Callable[..., tuple[StubEntryPoint, ...]]


def run_hook(app: StubApp) -> None:
    """Invoke the hook with the duck-typed app the loader is agnostic to."""
    load_router_plugins(cast(Any, app))


@pytest.fixture
def discovered(monkeypatch: pytest.MonkeyPatch) -> InstallPlugins:
    """Install a controllable `entry_points` in the loader's namespace."""

    def _install(*plugins: StubEntryPoint) -> tuple[StubEntryPoint, ...]:
        def fake_entry_points(*, group: str) -> list[StubEntryPoint]:
            assert group == ROUTER_PLUGINS_GROUP
            return list(plugins)

        monkeypatch.setattr(f"{MODULE}.entry_points", fake_entry_points)
        return plugins

    return _install


@pytest.fixture(autouse=True)
def stub_vllm_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for `vllm.envs`, parsing VLLM_PLUGINS exactly as vLLM does."""
    envs = types.ModuleType("vllm.envs")

    def module_getattr(name: str) -> Any:
        if name == "VLLM_PLUGINS":
            raw = os.environ.get("VLLM_PLUGINS")
            return None if raw is None else raw.split(",")
        raise AttributeError(name)

    setattr(envs, "__getattr__", module_getattr)
    if "vllm" in sys.modules:
        monkeypatch.setattr(sys.modules["vllm"], "envs", envs, raising=False)
    else:
        parent = types.ModuleType("vllm")
        setattr(parent, "envs", envs)
        monkeypatch.setitem(sys.modules, "vllm", parent)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)


# @spec PORT-PLUG-001
def test_plugin_router_is_mounted(discovered: InstallPlugins) -> None:
    router = object()
    discovered(StubEntryPoint("nim", result=lambda app: app.include_router(router)))
    app = StubApp()

    run_hook(app)

    assert app.included == [router]


# @spec PORT-PLUG-001
def test_every_plugin_is_invoked_when_allowlist_unset(discovered: InstallPlugins) -> None:
    discovered(
        StubEntryPoint("a", result=lambda app: app.include_router("a")),
        StubEntryPoint("b", result=lambda app: app.include_router("b")),
    )
    app = StubApp()

    run_hook(app)

    assert app.included == ["a", "b"]


# @spec PORT-PLUG-001
def test_no_entry_points_is_a_noop(discovered: InstallPlugins) -> None:
    discovered()
    app = StubApp()

    run_hook(app)

    assert app.included == []


# @spec PORT-PLUG-002
def test_allowlist_admits_only_named_plugins(discovered: InstallPlugins, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_PLUGINS", "b")
    plugins = discovered(
        StubEntryPoint("a", result=lambda app: app.include_router("a")),
        StubEntryPoint("b", result=lambda app: app.include_router("b")),
    )
    app = StubApp()

    run_hook(app)

    assert app.included == ["b"]
    # A filtered-out plugin is never even loaded.
    assert plugins[0].load_calls == 0


# @spec PORT-PLUG-002
def test_allowlist_excluding_a_failing_plugin_leaves_startup_intact(
    discovered: InstallPlugins, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_PLUGINS", "good")
    discovered(
        StubEntryPoint("bad", error=RuntimeError("must not load")),
        StubEntryPoint("good", result=lambda app: app.include_router("good")),
    )
    app = StubApp()

    run_hook(app)

    assert app.included == ["good"]


# @spec PORT-PLUG-003
def test_load_failure_propagates_instead_of_log_and_continue(discovered: InstallPlugins) -> None:
    # The deliberate divergence from vllm.plugins.load_plugins_by_group, whose
    # per-plugin except-block logs the failure and continues.
    plugins = discovered(
        StubEntryPoint("bad", error=RuntimeError("boom")),
        StubEntryPoint("later", result=lambda app: app.include_router("later")),
    )
    app = StubApp()

    with pytest.raises(RuntimeError, match="boom"):
        run_hook(app)

    assert app.included == []
    assert plugins[1].load_calls == 0


# @spec PORT-PLUG-003
def test_register_failure_propagates(discovered: InstallPlugins) -> None:
    def half_mounting_register(app: StubApp) -> None:
        app.include_router("first")
        raise ValueError("half-mounted")

    plugins = discovered(
        StubEntryPoint("bad", result=half_mounting_register),
        StubEntryPoint("later", result=lambda app: app.include_router("later")),
    )
    app = StubApp()

    with pytest.raises(ValueError, match="half-mounted"):
        run_hook(app)

    # The half-mounted route exists but startup aborts before it can serve.
    assert app.included == ["first"]
    assert plugins[1].load_calls == 0


# @spec PORT-PLUG-001
def test_serving_state_is_populated_when_plugin_runs(discovered: InstallPlugins) -> None:
    seen: list[Any] = []

    def register(app: StubApp) -> None:
        seen.append(app.state.serving_transcription)

    discovered(StubEntryPoint("nim", result=register))
    app = StubApp()
    app.state.serving_transcription = "serving-object"

    run_hook(app)

    assert seen == ["serving-object"]
