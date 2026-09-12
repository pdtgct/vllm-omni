# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native persistent sessions keep their own admission path on the new host."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_omni.entrypoints.openai import api_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("duplex_query", [None, "true", "false"])
async def test_native_model_bypasses_duplex_handler_and_its_warmup(monkeypatch, duplex_query):
    """@spec PORT-STATE-018: duplex warmup cannot grant native admission."""
    serving = SimpleNamespace(_is_recognized_streaming_model=True, park_token_id=17)
    duplex = SimpleNamespace(handle_realtime_session=AsyncMock())
    warmup = SimpleNamespace(is_set=lambda: False, wait=AsyncMock(side_effect=AssertionError("duplex wait")))
    state = SimpleNamespace(openai_serving_realtime=serving, openai_serving_duplex=duplex, duplex_warmup_done=warmup)
    websocket = SimpleNamespace(app=SimpleNamespace(state=state), query_params={"duplex": duplex_query})
    native = SimpleNamespace(handle_connection=AsyncMock())
    observer = object()
    monkeypatch.setattr(api_server.streaming_install, "resolve_installed_observer", lambda _: observer)
    captured = []

    def connection(socket, selected, **kwargs):
        captured.append((socket, selected, kwargs))
        return native

    monkeypatch.setattr(api_server, "RealtimeConnection", connection)
    await api_server.realtime_websocket(websocket)
    duplex.handle_realtime_session.assert_not_called()
    warmup.wait.assert_not_called()
    native.handle_connection.assert_awaited_once()
    assert captured == [(websocket, serving, {"observer": observer, "park_token_id": 17})]


@pytest.mark.asyncio
async def test_other_models_keep_upstream_default_duplex_route(monkeypatch):
    duplex = SimpleNamespace(handle_realtime_session=AsyncMock())
    serving = SimpleNamespace(_is_recognized_streaming_model=False)
    state = SimpleNamespace(openai_serving_realtime=serving, openai_serving_duplex=duplex)
    websocket = SimpleNamespace(app=SimpleNamespace(state=state), query_params={})
    await api_server.realtime_websocket(websocket)
    duplex.handle_realtime_session.assert_awaited_once_with(websocket)
