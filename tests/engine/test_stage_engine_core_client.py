# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health()."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_client(*, engine_dead=False):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 0
    client.resources = SimpleNamespace(engine_dead=engine_dead)
    return client


def test_check_health_passes_when_alive():
    client = _make_client(engine_dead=False)
    client.check_health()  # no exception


def test_check_health_raises_when_resources_engine_dead():
    client = _make_client(engine_dead=True)
    with pytest.raises(EngineDeadError, match="engine core is dead"):
        client.check_health()


def test_utility_call_runs_on_the_stage_clients_owner_loop(monkeypatch):
    # @spec PORT-STATE-013
    owner_loop = asyncio.new_event_loop()
    owner_ready = threading.Event()
    owner_thread_id: list[int] = []

    def run_owner_loop():
        asyncio.set_event_loop(owner_loop)
        owner_thread_id.append(threading.get_ident())
        owner_ready.set()
        owner_loop.run_forever()

    owner_thread = threading.Thread(target=run_owner_loop)
    owner_thread.start()
    assert owner_ready.wait(timeout=2.0)

    observed_thread_id: list[int] = []

    async def fake_call_utility_async(self, method, *args):
        observed_thread_id.append(threading.get_ident())
        return method, args

    monkeypatch.setattr(
        AsyncMPClient,
        "call_utility_async",
        fake_call_utility_async,
    )
    client = object.__new__(StageEngineCoreClient)
    client._owner_loop = owner_loop

    try:
        result = asyncio.run(
            client.call_utility_async("persistent_state_snapshot", "arg")
        )
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=2.0)
        owner_loop.close()

    assert result == ("persistent_state_snapshot", ("arg",))
    assert observed_thread_id == owner_thread_id
