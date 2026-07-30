# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-Uvicorn tests for application-plugin WebSocket admission."""

from __future__ import annotations

import asyncio
import importlib.util
import socket
import sys
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE_PATH = Path(__file__).resolve().parents[3] / "vllm_omni/entrypoints/openai/application_plugins.py"
_SPEC = importlib.util.spec_from_file_location(
    "_application_plugins_uvicorn_under_test",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
application_plugins = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = application_plugins
_SPEC.loader.exec_module(application_plugins)


async def _wait_started(server: uvicorn.Server) -> None:
    for _ in range(200):
        if server.started:
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("Uvicorn did not start")


def _listen_socket() -> tuple[socket.socket, str, int]:
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(("127.0.0.1", 0))
    listen_socket.listen()
    listen_socket.setblocking(False)
    host, port = listen_socket.getsockname()
    return listen_socket, host, port


async def _start_server(
    app: FastAPI,
    listen_socket: socket.socket,
    **config: object,
):
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="error",
            lifespan="off",
            **config,
        )
    )
    server_task = asyncio.create_task(server.serve(sockets=[listen_socket]))
    await _wait_started(server)
    return server, server_task


async def _open_websocket(
    host: str,
    port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes]:
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(
        (
            "GET /v1/realtime HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode()
    )
    await writer.drain()
    response_head = await asyncio.wait_for(
        reader.readuntil(b"\r\n\r\n"),
        timeout=2,
    )
    return reader, writer, response_head


async def _read_websocket_close_code(reader: asyncio.StreamReader) -> int:
    first, second = await asyncio.wait_for(reader.readexactly(2), timeout=2)
    assert first & 0x0F == 0x08
    payload_length = second & 0x7F
    assert payload_length >= 2
    assert payload_length < 126
    payload = await reader.readexactly(payload_length)
    return int.from_bytes(payload[:2], "big")


@pytest.mark.asyncio
async def test_closed_websocket_is_denied_as_http_before_accept() -> None:
    """Uvicorn's denial extension projects a pre-handshake HTTP 503."""
    # @spec ING-VEH-019, ING-VEH-024

    class ClosedAdmission:
        def try_acquire(self):
            return None

    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.close()

    slot = application_plugins.prepare_application_plugin_asgi_wrappers(
        app,
        admission=ClosedAdmission(),
    )
    slot.seal()

    listen_socket, host, port = _listen_socket()
    server, server_task = await _start_server(app, listen_socket)
    try:
        _, writer, response_head = await _open_websocket(host, port)
        writer.close()
        await writer.wait_closed()

        assert response_head.split(b"\r\n", 1)[0] == (b"HTTP/1.1 503 Service Unavailable")
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=2)
        listen_socket.close()


@pytest.mark.asyncio
async def test_post_accept_admission_race_closes_websocket_with_1013() -> None:
    """A detached owner denied after acceptance gets the promised close code."""
    # @spec ING-VEH-019, ING-VEH-024
    accepted = asyncio.Event()
    try_detached_owner = asyncio.Event()
    admission = application_plugins.create_application_admission()
    await admission.open_after_http_listener_bound()
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        accepted.set()
        await try_detached_owner.wait()
        detached_lease = admission.view.try_acquire()
        if detached_lease is None:
            await websocket.close(code=1013)
            return
        detached_lease.release()
        await websocket.close()

    slot = application_plugins.prepare_application_plugin_asgi_wrappers(
        app,
        admission=admission.view,
    )
    slot.seal()

    listen_socket, host, port = _listen_socket()
    server, server_task = await _start_server(app, listen_socket)
    writer = None
    try:
        reader, writer, response_head = await _open_websocket(host, port)
        assert response_head.split(b"\r\n", 1)[0] == (b"HTTP/1.1 101 Switching Protocols")
        await asyncio.wait_for(accepted.wait(), timeout=2)
        admission.close_from_launcher_thread()
        try_detached_owner.set()
        assert await _read_websocket_close_code(reader) == 1013
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=2)
        listen_socket.close()


@pytest.mark.asyncio
async def test_open_websocket_lease_survives_close_until_disconnect() -> None:
    """Drain closure preserves an admitted WebSocket until scope cleanup."""
    # @spec ING-VEH-016, ING-VEH-019, ING-VEH-022, ING-VEH-024
    active_leases = 0
    admission_open = True
    accepted = asyncio.Event()
    disconnected = asyncio.Event()

    class Lease:
        def __init__(self) -> None:
            self._released = False

        def release(self) -> None:
            nonlocal active_leases
            if self._released:
                return
            self._released = True
            active_leases -= 1

    class Admission:
        def try_acquire(self):
            nonlocal active_leases
            if not admission_open:
                return None
            active_leases += 1
            return Lease()

    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        accepted.set()
        message = await websocket.receive()
        assert message["type"] == "websocket.disconnect"
        disconnected.set()

    slot = application_plugins.prepare_application_plugin_asgi_wrappers(
        app,
        admission=Admission(),
    )
    slot.seal()

    listen_socket, host, port = _listen_socket()
    server, server_task = await _start_server(app, listen_socket)
    writer = None
    try:
        _, writer, response_head = await _open_websocket(host, port)
        assert response_head.split(b"\r\n", 1)[0] == (b"HTTP/1.1 101 Switching Protocols")
        await asyncio.wait_for(accepted.wait(), timeout=2)
        assert active_leases == 1

        admission_open = False
        assert active_leases == 1

        writer.close()
        await writer.wait_closed()
        writer = None
        await asyncio.wait_for(disconnected.wait(), timeout=2)
        for _ in range(100):
            if active_leases == 0:
                break
            await asyncio.sleep(0.01)
        assert active_leases == 0
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=2)
        listen_socket.close()


@pytest.mark.asyncio
async def test_h11_incomplete_event_limit_precedes_raw_asgi_composition() -> None:
    """Raw composition cannot bypass the launcher's h11 parser bound."""
    # @spec ING-VEH-017, ING-VEH-024
    acquisitions = 0

    class Lease:
        def release(self) -> None:
            return

    class Admission:
        def try_acquire(self):
            nonlocal acquisitions
            acquisitions += 1
            return Lease()

    app = FastAPI()

    @app.post("/v1/work")
    async def work() -> dict[str, bool]:
        return {"ok": True}

    slot = application_plugins.prepare_application_plugin_asgi_wrappers(
        app,
        admission=Admission(),
    )
    slot.seal()

    listen_socket, host, port = _listen_socket()
    server, server_task = await _start_server(
        app,
        listen_socket,
        http="h11",
        h11_max_incomplete_event_size=128,
    )
    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.write((f"POST /v1/work HTTP/1.1\r\nHost: {host}:{port}\r\nX-Oversized: " + ("a" * 512)).encode())
        await writer.drain()
        response_head = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"),
            timeout=2,
        )

        assert response_head.split(b"\r\n", 1)[0] == b"HTTP/1.1 400 Bad Request"
        assert acquisitions == 0
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=2)
        listen_socket.close()
