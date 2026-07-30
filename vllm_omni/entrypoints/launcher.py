# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni-owned HTTP launcher with optional application lifecycle barriers."""

import asyncio
import signal
import socket
from collections.abc import Awaitable
from functools import partial
from typing import Any, Protocol, cast

import uvicorn
from fastapi import FastAPI
from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http as _upstream_serve_http
from vllm.entrypoints.serve.utils.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.serve.utils.ssl import SSLCertRefresher
from vllm.logger import init_logger
from vllm.utils.network_utils import find_process_using_port

logger = init_logger(__name__)

# Every vLLM pin bump must re-diff these behaviors against
# vllm.entrypoints.launcher before the pin is accepted.
MIRRORED_UPSTREAM_BEHAVIORS = frozenset(
    {
        "engine_shutdown",
        "h11_limits",
        "port_conflict_diagnostics",
        "route_logging",
        "signal_cleanup",
        "ssl_refresh",
        "uvicorn_configuration",
        "watchdog",
    }
)


class ApplicationLifecycleHook(Protocol):
    """Optional barriers owned by the selected application participants."""

    async def on_bound(self) -> None:
        """Open composed admission after Uvicorn is listening."""

    def on_shutdown_requested(
        self,
        cause: BaseException | None = None,
    ) -> None:
        """Synchronously close admission at the shutdown linearization point."""

    async def before_http_shutdown(self) -> None:
        """Quiesce and drain participant-owned work before HTTP stops."""

    async def before_engine_shutdown(self) -> None:
        """Unwind participant contexts before the engine stops."""

    async def wait_failed(self) -> None:
        """Raise the first post-start participant failure."""


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    lifecycle_hook: ApplicationLifecycleHook | None = None,
    **uvicorn_kwargs: Any,
) -> Awaitable[None]:
    """Serve HTTP, adding lifecycle barriers only when explicitly selected."""
    # @spec ING-VEH-017, ING-VEH-022
    if lifecycle_hook is None:
        return cast(
            Awaitable[None],
            await _upstream_serve_http(
                app,
                sock,
                enable_ssl_refresh=enable_ssl_refresh,
                **uvicorn_kwargs,
            ),
        )
    return await _serve_http_with_lifecycle(
        app,
        sock,
        enable_ssl_refresh=enable_ssl_refresh,
        lifecycle_hook=lifecycle_hook,
        uvicorn_kwargs=uvicorn_kwargs,
    )


async def _serve_http_with_lifecycle(
    app: FastAPI,
    sock: socket.socket | None,
    *,
    enable_ssl_refresh: bool,
    lifecycle_hook: ApplicationLifecycleHook,
    uvicorn_kwargs: dict[str, Any],
) -> Awaitable[None]:
    """Own Uvicorn so admission closure precedes every shutdown trigger."""
    _log_routes(app)
    engine_client = app.state.engine_client
    server, config = _build_server(app, uvicorn_kwargs)
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))
    watchdog_task = loop.create_task(_watchdog_loop(server, engine_client))
    programmatic_stop_task = loop.create_task(_wait_for_programmatic_stop(server))
    failure_task = loop.create_task(lifecycle_hook.wait_failed())
    shutdown_event = asyncio.Event()
    primary_failure: BaseException | None = None
    secondary_failures: list[BaseException] = []

    ssl_cert_refresher = (
        None
        if not enable_ssl_refresh
        else SSLCertRefresher(
            ssl_context=config.ssl,
            key_path=config.ssl_keyfile,
            cert_path=config.ssl_certfile,
            ca_path=config.ssl_ca_certs,
        )
    )

    def request_shutdown(cause: BaseException | None = None) -> None:
        nonlocal primary_failure
        if cause is not None:
            if primary_failure is None:
                primary_failure = cause
            elif cause is not primary_failure:
                secondary_failures.append(cause)
        if shutdown_event.is_set():
            return
        try:
            lifecycle_hook.on_shutdown_requested(cause)
        except BaseException as error:
            secondary_failures.append(error)
            logger.exception("Application lifecycle shutdown notification failed")
        shutdown_event.set()

    def signal_handler() -> None:
        logger.info_once("[shutdown] API server: shutdown triggered")
        request_shutdown()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    async def dummy_shutdown() -> None:
        pass

    try:
        try:
            await _wait_for_server_bound(server, server_task)
            await lifecycle_hook.on_bound()
        except BaseException as error:
            request_shutdown(error)

        if not shutdown_event.is_set():
            shutdown_wait = loop.create_task(shutdown_event.wait())
            done, _ = await asyncio.wait(
                {
                    server_task,
                    failure_task,
                    watchdog_task,
                    programmatic_stop_task,
                    shutdown_wait,
                },
                return_when=asyncio.FIRST_COMPLETED,
            )
            if failure_task in done:
                try:
                    await failure_task
                except BaseException as error:
                    request_shutdown(error)
            elif watchdog_task in done:
                try:
                    await watchdog_task
                except BaseException as error:
                    request_shutdown(error)
            elif shutdown_wait in done:
                request_shutdown()
            elif programmatic_stop_task in done:
                request_shutdown()
            elif server_task in done:
                server_error = _task_failure(server_task)
                request_shutdown(server_error or RuntimeError("HTTP server stopped unexpectedly"))
            else:
                request_shutdown()
            if not shutdown_wait.done():
                shutdown_wait.cancel()
            await asyncio.gather(
                shutdown_wait,
                return_exceptions=True,
            )

        await _coordinated_shutdown(
            engine_client,
            server,
            server_task,
            watchdog_task,
            programmatic_stop_task,
            ssl_cert_refresher,
            lifecycle_hook,
            uvicorn_kwargs,
            secondary_failures,
        )
        if primary_failure is not None:
            if secondary_failures:
                logger.error(
                    "Application shutdown retained %d secondary failure(s)",
                    len(secondary_failures),
                )
            raise primary_failure
        if secondary_failures:
            raise secondary_failures[0]
        return dummy_shutdown()
    except asyncio.CancelledError:
        request_shutdown()
        await asyncio.shield(
            _coordinated_shutdown(
                engine_client,
                server,
                server_task,
                watchdog_task,
                programmatic_stop_task,
                ssl_cert_refresher,
                lifecycle_hook,
                uvicorn_kwargs,
                secondary_failures,
            )
        )
        raise
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        for task in (
            failure_task,
            watchdog_task,
            programmatic_stop_task,
        ):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            failure_task,
            watchdog_task,
            programmatic_stop_task,
            return_exceptions=True,
        )


async def _coordinated_shutdown(
    engine_client: EngineClient,
    server: uvicorn.Server,
    server_task: asyncio.Task[Any],
    watchdog_task: asyncio.Task[Any],
    programmatic_stop_task: asyncio.Task[Any],
    ssl_cert_refresher: SSLCertRefresher | None,
    lifecycle_hook: ApplicationLifecycleHook,
    uvicorn_kwargs: dict[str, Any],
    errors: list[BaseException],
) -> None:
    """Run every barrier even when an earlier drain operation fails."""
    try:
        await lifecycle_hook.before_http_shutdown()
    except BaseException as error:
        errors.append(error)
        logger.exception("Application participant drain failed")
    finally:
        server.should_exit = True
        watchdog_task.cancel()
        programmatic_stop_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()
        try:
            if not server_task.done():
                await server_task
            else:
                server_error = _task_failure(server_task)
                if server_error is not None:
                    errors.append(server_error)
        except asyncio.CancelledError:
            try:
                await server.shutdown()
            except BaseException as error:
                errors.append(error)
        except BaseException as error:
            errors.append(error)
            port = uvicorn_kwargs["port"]
            process = find_process_using_port(port)
            if process is not None:
                logger.warning(
                    "port %s is used by process %s launched with command:\n%s",
                    port,
                    process,
                    " ".join(process.cmdline()),
                )

    try:
        await lifecycle_hook.before_engine_shutdown()
    except BaseException as error:
        errors.append(error)
        logger.exception("Application participant exit failed")
    finally:
        timeout = engine_client.vllm_config.shutdown_timeout
        mode = "abort" if timeout == 0 else "drain"
        logger.info(
            "[shutdown] API server: stopping engine client mode=%s timeout=%ss",
            mode,
            timeout,
        )
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                partial(engine_client.shutdown, timeout=timeout),
            )
        except BaseException as error:
            errors.append(error)
        logger.info_once("[shutdown] API server: engine client stopped")


def _log_routes(app: FastAPI) -> None:
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if methods is not None and path is not None:
            logger.info("Route: %s, Methods: %s", path, ", ".join(methods))
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if endpoint is not None and path is not None and methods is None:
            logger.info("Route: %s, Endpoint: %s", path, endpoint.__name__)


def _build_server(
    app: FastAPI,
    uvicorn_kwargs: dict[str, Any],
) -> tuple[uvicorn.Server, uvicorn.Config]:
    h11_max_incomplete_event_size = uvicorn_kwargs.pop(
        "h11_max_incomplete_event_size",
        None,
    )
    h11_max_header_count = uvicorn_kwargs.pop(
        "h11_max_header_count",
        None,
    )
    if h11_max_incomplete_event_size is None:
        h11_max_incomplete_event_size = H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT
    if h11_max_header_count is None:
        h11_max_header_count = H11_MAX_HEADER_COUNT_DEFAULT

    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.h11_max_incomplete_event_size = h11_max_incomplete_event_size
    config.h11_max_header_count = h11_max_header_count
    config.load()
    server = uvicorn.Server(config)
    app.state.server = server
    return server, config


async def _wait_for_server_bound(
    server: uvicorn.Server,
    server_task: asyncio.Task[Any],
) -> None:
    while not server.started:
        if server_task.done():
            error = _task_failure(server_task)
            if error is not None:
                raise error
            raise RuntimeError("HTTP server stopped before listener bind")
        await asyncio.sleep(0.01)


async def _wait_for_programmatic_stop(server: uvicorn.Server) -> None:
    while not server.should_exit:
        await asyncio.sleep(0.01)


def _task_failure(task: asyncio.Task[Any]) -> BaseException | None:
    if task.cancelled():
        return asyncio.CancelledError()
    return task.exception()


async def _watchdog_loop(
    server: uvicorn.Server,
    engine: EngineClient,
) -> None:
    while True:
        await asyncio.sleep(5.0)
        engine_errored = engine.errored and not engine.is_running
        if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored:
            raise RuntimeError("engine client failed its HTTP watchdog")
