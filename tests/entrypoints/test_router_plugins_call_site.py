# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invocation-point contract for the router-plugin hook.

Separate from `test_router_plugins.py` because importing `api_server` pulls in
vllm and cannot run on a CPU-only dev box; this file is venv/pod-gated.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# @spec PORT-PLUG-001
# @spec PORT-PLUG-003
def test_hook_runs_after_app_state_and_before_background_startup() -> None:
    from vllm_omni.entrypoints.openai import api_server

    source = inspect.getsource(api_server.omni_run_server_worker)
    init_state = source.index("omni_init_app_state(engine_client")
    hook = source.index("load_router_plugins(app)")
    background = source.index("STORAGE_MANAGER.start()")
    serve = source.index("serve_http(")

    assert init_state < hook < background < serve
