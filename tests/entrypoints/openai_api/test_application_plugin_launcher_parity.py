# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract: the no-hook path is the paired vLLM launcher.

ING-VEH-017 requires strict zero-plugin parity: with no lifecycle hook the
Omni launcher preserves the paired vLLM launcher's behavior, including its
shutdown ordering. The strongest form of that guarantee is delegation — the
no-hook call is a pass-through to `vllm.entrypoints.launcher.serve_http`, so
parity holds by construction rather than by mirroring.

The pin guard makes the pin-bump re-diff executable: the launcher module
declares the SHA-256 of the upstream launcher source it was last dispositioned
against, and the test compares it to the installed source. A vLLM pin bump
fails here until the re-diff is done and the digest re-stamped.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER_PATH = _ROOT / "vllm_omni/entrypoints/launcher.py"


def _load_launcher():
    assert _LAUNCHER_PATH.exists(), "Omni-owned launcher is not implemented"
    spec = importlib.util.spec_from_file_location(
        "_omni_launcher_parity_under_test",
        _LAUNCHER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_zero_plugin_serve_delegates_to_the_paired_vllm_launcher(
    monkeypatch,
) -> None:
    """No hook means the upstream launcher runs — parity by construction."""
    # @spec ING-VEH-017
    import vllm.entrypoints.launcher as upstream_module

    recorded: dict[str, object] = {}
    sentinel_result = object()

    async def recording_upstream(app, sock, enable_ssl_refresh=False, **kw):
        recorded["app"] = app
        recorded["sock"] = sock
        recorded["enable_ssl_refresh"] = enable_ssl_refresh
        recorded["kwargs"] = kw
        return sentinel_result

    monkeypatch.setattr(upstream_module, "serve_http", recording_upstream)

    constructed: list[object] = []
    original_server = uvicorn.Server

    class RecordingServer(original_server):
        def __init__(self, config) -> None:
            constructed.append(config)
            super().__init__(config)

    monkeypatch.setattr(uvicorn, "Server", RecordingServer)

    # Load after patching so the module binds the recording upstream symbol.
    launcher = _load_launcher()

    app = FastAPI()
    app.state.engine_client = object()
    result = await launcher.serve_http(
        app,
        None,
        port=8123,
        log_level="error",
    )

    assert result is sentinel_result
    assert recorded["app"] is app
    assert recorded["sock"] is None
    assert recorded["enable_ssl_refresh"] is False
    assert recorded["kwargs"] == {"port": 8123, "log_level": "error"}
    assert constructed == [], "the no-hook path must not own a Uvicorn server"


def test_mirrored_surface_declares_shutdown_ordering() -> None:
    """The behavior inventory and the parity suite name the same surface."""
    # @spec ING-VEH-017
    launcher = _load_launcher()

    assert "shutdown_ordering" in launcher.MIRRORED_UPSTREAM_BEHAVIORS, (
        "shutdown ordering is a mirrored upstream behavior with a "
        "dispositioned hook-path divergence; the re-diff inventory must "
        "name it"
    )


def test_launcher_pins_the_dispositioned_upstream_launcher_source() -> None:
    """A vLLM pin bump cannot pass silently: re-diff, then re-stamp."""
    # @spec ING-VEH-017
    import vllm.entrypoints.launcher as upstream_module

    launcher = _load_launcher()
    source_path = inspect.getsourcefile(upstream_module)
    assert source_path is not None
    actual = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()

    assert launcher.UPSTREAM_LAUNCHER_SHA256 == actual, (
        "vllm.entrypoints.launcher changed under the pin: re-diff every "
        "behavior in MIRRORED_UPSTREAM_BEHAVIORS against the new source, "
        "incorporate or disposition each change, then re-stamp "
        "UPSTREAM_LAUNCHER_SHA256"
    )
