# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic application-plugin contracts for the OpenAI API server.

This module deliberately contains no compatibility-frontend behavior. It
provides the generic host-owned lifecycle hook used by explicitly selected
application plugins.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from types import TracebackType
from typing import Any, Protocol

APPLICATION_PLUGIN_ENTRY_POINT_GROUP = "vllm_omni.application_plugins"
APPLICATION_PLUGIN_OPERATIONAL_PATHS = frozenset({"/health", "/live", "/metrics"})
DEFAULT_APPLICATION_PLUGIN_STARTUP_TIMEOUT = 60.0
DEFAULT_WS_MAX_SIZE = 16 * 1024 * 1024
ApplicationASGI = Callable[[Any, Any, Any], Awaitable[None]]
ApplicationASGIWrapper = Callable[[ApplicationASGI], ApplicationASGI]
ApplicationASGIInstaller = Callable[[ApplicationASGIWrapper], None]


class ApplicationASGIComposition(Protocol):
    """Host-owned application-plugin ASGI composition transaction."""

    def install(self, wrapper: ApplicationASGIWrapper) -> None:
        """Register one wrapper while its selected plugin is entering."""

    def seal(self) -> None:
        """Construct and freeze the wrapper chain before HTTP starts."""

    def fail(self) -> None:
        """Irreversibly invalidate the composition transaction."""


class ApplicationAdmissionState(Enum):
    """Host-owned admission state for application inference paths."""

    CLOSED = "closed"
    OPEN = "open"


class AdmissionLease(Protocol):
    """One admitted application owner registration."""

    def release(self) -> None:
        """Release this owner registration idempotently."""


class ApplicationAdmissionView(Protocol):
    """Acquisition-only projection of the host admission controller."""

    def try_acquire(self) -> AdmissionLease | None:
        """Atomically register one owner, or reject it after closure."""


class ApplicationAdmission(Protocol):
    """Host-internal controller for application inference admission."""

    @property
    def state(self) -> ApplicationAdmissionState:
        """Return the current admission state."""

    @property
    def view(self) -> ApplicationAdmissionView:
        """Return the stable acquisition-only external capability."""

    async def open_after_http_listener_bound(self) -> None:
        """Make the one host-owned readiness transition after HTTP binds."""

    async def close_before_owner_drain(self) -> None:
        """Close admission before the process drains application owners."""

    def close_from_launcher_thread(self) -> None:
        """Close admission synchronously when the HTTP launcher gets signal."""


@dataclass(frozen=True)
class ApplicationPluginHostContext:
    """Generic host values shared by all explicitly selected plugins."""

    app: Any
    engine_client: Any
    admission: ApplicationAdmissionView
    install_asgi_wrapper: ApplicationASGIInstaller


@dataclass(frozen=True)
class ApplicationPluginContext(ApplicationPluginHostContext):
    """One selected plugin's host context and opaque configuration value."""

    plugin_name: str
    config: str | None


class ApplicationPlugin(Protocol):
    """A lifecycle-aware application plugin entry point."""

    config_optional: bool

    def __call__(
        self,
        context: ApplicationPluginContext,
    ) -> ApplicationPluginParticipant:
        """Return the selected plugin's async lifetime context manager."""


class ApplicationPluginParticipant(
    AbstractAsyncContextManager["ApplicationPluginParticipant"],
    Protocol,
):
    """One selected plugin's readiness, failure, and drain participant."""

    async def wait_ready(self) -> None:
        """Attest that owned resources are bound and locally reachable."""

    async def wait_failed(self) -> None:
        """Raise the first owned-resource failure and never return normally."""

    async def mark_serving(self) -> None:
        """Observe that host application admission is now open."""

    async def quiesce_and_drain(self) -> None:
        """Stop new work and drain this plugin's owners before host exit."""

    @property
    def shutdown_grace(self) -> float:
        """Return this participant's finite shutdown grace in seconds."""


ApplicationPluginEntryPoint = Callable[
    [ApplicationPluginContext],
    ApplicationPluginParticipant,
]


@dataclass(frozen=True)
class SelectedApplicationPlugin:
    """One discovered entry point paired with its validated opaque config."""

    name: str
    entry_point: ApplicationPluginEntryPoint
    config: str | None


class ApplicationPluginLifetime(
    AbstractAsyncContextManager["ApplicationPluginLifetime"],
    Protocol,
):
    """Selected-plugin lifetime controlled by the generic API-server host."""

    async def __aenter__(self) -> ApplicationPluginLifetime:
        """Enter selected plugins in explicit CLI order."""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Unwind every entered plugin in reverse order."""

    async def wait_ready(self) -> None:
        """Await all participant readiness attestations concurrently."""

    async def wait_failed(self) -> None:
        """Raise the first supervised participant failure."""

    async def mark_serving(self) -> None:
        """Tell selected plugins that host admission is now open."""

    async def quiesce_and_drain(self) -> None:
        """Quiesce selected plugins and drain their owners before exit."""

    @property
    def shutdown_grace(self) -> float:
        """Return the finite shared HTTP/plugin shutdown grace."""


class ApplicationPluginError(RuntimeError):
    """Python-3.10-compatible aggregate for independent lifecycle faults."""

    def __init__(
        self,
        message: str,
        exceptions: Sequence[BaseException],
    ) -> None:
        self.exceptions = tuple(exceptions)
        detail = "; ".join(str(error) for error in self.exceptions)
        super().__init__(f"{message}: {detail}" if detail else message)


def add_application_plugin_args(parser: argparse.ArgumentParser) -> None:
    """Add repeatable generic application-plugin selection CLI arguments."""
    parser.add_argument(
        "--application-plugin",
        action="append",
        default=[],
        metavar="NAME",
        help=("Explicitly enable one application plugin entry point (repeatable)."),
    )
    parser.add_argument(
        "--application-plugin-config",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=("Opaque configuration for one selected application plugin (repeatable)."),
    )
    parser.add_argument(
        "--ws-max-size",
        type=_positive_integer,
        default=DEFAULT_WS_MAX_SIZE,
        metavar="BYTES",
        help="Maximum accepted WebSocket message size in bytes.",
    )
    parser.add_argument(
        "--application-plugin-startup-timeout",
        type=_positive_float,
        default=DEFAULT_APPLICATION_PLUGIN_STARTUP_TIMEOUT,
        metavar="SECONDS",
        help=("One aggregate deadline for selected application-plugin entry and readiness."),
    )


def _positive_integer(value: str) -> int:
    """Parse one finite positive byte count for the HTTP server."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    """Parse one finite positive duration for the API server."""
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def validate_application_plugin_options(
    selected_names: Sequence[str],
    config_values: Sequence[str],
    *,
    api_server_worker_count: int,
) -> Mapping[str, str | None]:
    """Validate explicit plugin selection and opaque keyed configuration."""
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("duplicate selected application plugin name")
    if selected_names and api_server_worker_count != 1:
        raise ValueError("selected application plugins require exactly one API-server worker")

    config: dict[str, str | None] = {}
    selected = set(selected_names)
    for value in config_values:
        name, separator, opaque_value = value.partition("=")
        if not separator or not name:
            raise ValueError("application plugin config must have NAME=VALUE form")
        if name in config:
            raise ValueError("duplicate config key for application plugin")
        if name not in selected:
            raise ValueError("application plugin config names an unselected plugin")
        config[name] = opaque_value
    return {name: config.get(name) for name in selected_names}


def discover_application_plugins(
    selected_names: Sequence[str],
) -> Sequence[ApplicationPluginEntryPoint]:
    """Discover only the selected entry points in explicit CLI order."""
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        entries = entry_points.select(group=APPLICATION_PLUGIN_ENTRY_POINT_GROUP)
    else:
        entries = getattr(entry_points, "get")(
            APPLICATION_PLUGIN_ENTRY_POINT_GROUP,
            (),
        )
    by_name: dict[str, list[Any]] = {}
    for entry_point in entries:
        by_name.setdefault(entry_point.name, []).append(entry_point)

    plugins: list[ApplicationPluginEntryPoint] = []
    for name in selected_names:
        try:
            matches = by_name[name]
        except KeyError as exc:
            raise ValueError(f"selected application plugin is not installed: {name}") from exc
        if len(matches) != 1:
            raise ValueError(f"selected application plugin name is ambiguous: {name}")
        plugin = matches[0].load()
        if not callable(plugin):
            raise ValueError(f"application plugin entry point is not callable: {name}")
        plugins.append(plugin)
    return plugins


def create_application_admission() -> ApplicationAdmission:
    """Create the host-owned admission state, initially closed."""
    # @spec ING-VEH-016, ING-VEH-019
    return _ApplicationAdmission()


def prepare_application_plugin_asgi_wrappers(
    app: Any,
    *,
    admission: ApplicationAdmissionView | None = None,
) -> ApplicationASGIComposition:
    """Prepare one stable plugin slot inside the host middleware boundary."""
    # @spec ING-VEH-019, ING-VEH-021, ING-VEH-024
    router = getattr(app, "router", None)
    if not callable(router):
        raise TypeError("application host router must be callable")
    middleware_stack = getattr(app, "middleware_stack", None)
    if admission is not None and not callable(getattr(admission, "try_acquire", None)):
        raise TypeError("application admission view must expose try_acquire()")
    slot = _ApplicationASGICompositionSlot(router, admission)

    try:
        app.router = slot
        if middleware_stack is not None:
            build_middleware_stack = getattr(
                app,
                "build_middleware_stack",
                None,
            )
            if not callable(build_middleware_stack):
                raise TypeError("application host must expose a callable build_middleware_stack")
            rebuilt_stack = build_middleware_stack()
            if not callable(rebuilt_stack):
                raise TypeError("application host middleware builder must return a callable")
            app.middleware_stack = rebuilt_stack
    except BaseException:
        slot.fail()
        app.router = router
        app.middleware_stack = middleware_stack
        raise
    return slot


class _ApplicationASGICompositionState(Enum):
    """Lifecycle state for the host-owned composition slot."""

    PREPARED = "prepared"
    SEALED = "sealed"
    FAILED = "failed"


class _ApplicationASGICompositionSlot:
    """Stable router identity populated before the host starts serving."""

    def __init__(
        self,
        host_router: ApplicationASGI,
        admission: ApplicationAdmissionView | None,
    ) -> None:
        self._host_router = host_router
        self._admission = admission
        self._wrappers: list[ApplicationASGIWrapper] = []
        self._installed: ApplicationASGI | None = None
        self._state = _ApplicationASGICompositionState.PREPARED

    def __getattr__(self, name: str) -> Any:
        return getattr(self._host_router, name)

    def install(self, wrapper: ApplicationASGIWrapper) -> None:
        """Register one wrapper before the composition is sealed."""
        self._require_prepared("install an application ASGI wrapper")
        if not callable(wrapper):
            raise TypeError("application ASGI wrapper must be callable")
        self._wrappers.append(wrapper)

    def seal(self) -> None:
        """Construct wrappers in CLI order and freeze the slot."""
        self._require_prepared("seal application ASGI wrappers")
        installed: ApplicationASGI = self._host_router
        if self._admission is not None:
            installed = _ApplicationAdmissionGuard(
                installed,
                self._admission,
            )
        try:
            for wrapper in reversed(self._wrappers):
                installed = wrapper(installed)
                if not callable(installed):
                    raise TypeError("application ASGI wrapper must return a callable")
        except BaseException:
            self.fail()
            raise
        self._installed = installed
        self._wrappers.clear()
        self._state = _ApplicationASGICompositionState.SEALED

    def fail(self) -> None:
        """Irreversibly invalidate this composition transaction."""
        self._installed = None
        self._wrappers.clear()
        self._state = _ApplicationASGICompositionState.FAILED

    def _require_prepared(self, action: str) -> None:
        if self._state is _ApplicationASGICompositionState.SEALED:
            raise RuntimeError(f"cannot {action}: application ASGI composition is sealed")
        if self._state is _ApplicationASGICompositionState.FAILED:
            raise RuntimeError(f"cannot {action}: application ASGI composition has failed")

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        installed = self._installed
        if installed is None:
            if self._state is _ApplicationASGICompositionState.FAILED:
                raise RuntimeError("application ASGI composition has failed")
            raise RuntimeError("application ASGI composition is not sealed")
        await installed(scope, receive, send)


class _ApplicationAdmissionGuard:
    """Innermost guard that atomically admits each original-router scope."""

    def __init__(
        self,
        host_router: ApplicationASGI,
        admission: ApplicationAdmissionView,
    ) -> None:
        self._host_router = host_router
        self._admission = admission

    # @spec ING-VEH-016, ING-VEH-019
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        path = scope.get("path", "")
        if scope_type not in {"http", "websocket"} or path in APPLICATION_PLUGIN_OPERATIONAL_PATHS:
            await self._host_router(scope, receive, send)
            return

        lease = self._admission.try_acquire()
        if lease is None:
            if scope_type == "websocket":
                await _reject_websocket_before_accept(scope, send)
            else:
                await _reject_http(send)
            return

        try:
            await self._host_router(scope, receive, send)
        finally:
            lease.release()


async def _reject_websocket_before_accept(scope: Any, send: Any) -> None:
    extensions = scope.get("extensions", {})
    if "websocket.http.response" in extensions:
        body = b"Application inference is unavailable"
        await send(
            {
                "type": "websocket.http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send(
            {
                "type": "websocket.http.response.body",
                "body": body,
            }
        )
        return
    await send({"type": "websocket.close"})


async def _reject_http(send: Any) -> None:
    body = json.dumps(
        {
            "error": {
                "message": "Application inference is unavailable",
                "type": "ServiceUnavailableError",
                "param": None,
                "code": 503,
            }
        },
        separators=(",", ":"),
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class _EntryScopedApplicationASGIInstaller:
    """Installer capability valid only during one plugin's awaited entry."""

    def __init__(self, installer: ApplicationASGIInstaller) -> None:
        self._installer = installer
        self._active = True

    def __call__(self, wrapper: ApplicationASGIWrapper) -> None:
        if not self._active:
            raise RuntimeError("application ASGI installer is active only during plugin entry")
        self._installer(wrapper)

    def close(self) -> None:
        """Invalidate this plugin's installer capability."""
        self._active = False


def application_plugin_lifetime(
    plugins: Sequence[SelectedApplicationPlugin],
    host_context: ApplicationPluginHostContext,
) -> ApplicationPluginLifetime:
    """Create the CLI-ordered, host-supervised participant lifetime."""
    # @spec ING-VEH-010, ING-VEH-017, ING-VEH-018, ING-VEH-022
    return _ManagedApplicationPluginLifetime(plugins, host_context)


class _ApplicationAdmission:
    """Single-process linearization point for application inference admission."""

    def __init__(self) -> None:
        self._state = ApplicationAdmissionState.CLOSED
        self._opened_once = False
        self._active_owners = 0
        self._lock = threading.Lock()
        self._view = _ReadOnlyApplicationAdmissionView(self)

    @property
    def state(self) -> ApplicationAdmissionState:
        with self._lock:
            return self._state

    @property
    def view(self) -> ApplicationAdmissionView:
        return self._view

    async def open_after_http_listener_bound(self) -> None:
        """Perform the one host-owned readiness transition."""
        with self._lock:
            if self._opened_once:
                raise RuntimeError("application admission cannot reopen after shutdown")
            self._opened_once = True
            self._state = ApplicationAdmissionState.OPEN

    # @spec ING-VEH-017
    async def close_before_owner_drain(self) -> None:
        """Reject further application owners before draining existing owners."""
        self.close_from_launcher_thread()

    def close_from_launcher_thread(self) -> None:
        """Perform the synchronous OPEN->CLOSED shutdown transition."""
        with self._lock:
            self._state = ApplicationAdmissionState.CLOSED

    def try_acquire(self) -> AdmissionLease | None:
        """Check admission and register one owner under the same lock."""
        with self._lock:
            if self._state is not ApplicationAdmissionState.OPEN:
                return None
            self._active_owners += 1
        return _ApplicationAdmissionLease(self)

    def release(self) -> None:
        with self._lock:
            if self._active_owners <= 0:
                raise RuntimeError("application admission owner underflow")
            self._active_owners -= 1


class _ApplicationAdmissionLease:
    """Unique, idempotently releasable application owner registration."""

    __slots__ = ("_admission", "_lock", "_released")

    def __init__(self, admission: _ApplicationAdmission) -> None:
        self._admission = admission
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._admission.release()


class _ReadOnlyApplicationAdmissionView:
    """Stable capability exposing only atomic owner acquisition."""

    __slots__ = ("_admission",)

    def __init__(self, admission: _ApplicationAdmission) -> None:
        self._admission = admission

    def try_acquire(self) -> AdmissionLease | None:
        return self._admission.try_acquire()


@dataclass
class _EnteredParticipant:
    """One entered context and its validated bounded-shutdown contract."""

    manager: ApplicationPluginParticipant
    participant: ApplicationPluginParticipant
    shutdown_grace: float


class _ManagedApplicationPluginLifetime:
    """Enter selected plugins and supervise their owned resources."""

    def __init__(
        self,
        plugins: Sequence[SelectedApplicationPlugin],
        host_context: ApplicationPluginHostContext,
    ) -> None:
        self._plugins = plugins
        self._host_context = host_context
        self._entered: list[_EnteredParticipant] = []
        self._failure_tasks: list[asyncio.Task[None]] = []
        self._orderly_shutdown = False
        self._drained = False
        self._exited = False

    async def __aenter__(self) -> _ManagedApplicationPluginLifetime:
        try:
            for selected in self._plugins:
                await self._enter_one(selected)
        except BaseException as primary:
            cleanup_errors = await self._exit_all(None, None, None)
            if cleanup_errors:
                _retain_secondary_errors(primary, cleanup_errors)
            raise
        return self

    async def _enter_one(self, selected: SelectedApplicationPlugin) -> None:
        scoped_installer = _EntryScopedApplicationASGIInstaller(self._host_context.install_asgi_wrapper)
        try:
            context = ApplicationPluginContext(
                app=self._host_context.app,
                engine_client=self._host_context.engine_client,
                admission=self._host_context.admission,
                install_asgi_wrapper=scoped_installer,
                plugin_name=selected.name,
                config=selected.config,
            )
            manager = selected.entry_point(context)
            participant = await manager.__aenter__()
            try:
                shutdown_grace = _validate_shutdown_grace(participant.shutdown_grace)
            except BaseException:
                await manager.__aexit__(None, None, None)
                raise
            self._entered.append(
                _EnteredParticipant(
                    manager=manager,
                    participant=participant,
                    shutdown_grace=shutdown_grace,
                )
            )
            self._failure_tasks.append(asyncio.create_task(self._supervise_failure(selected.name, participant)))
        finally:
            scoped_installer.close()

    async def _supervise_failure(
        self,
        name: str,
        participant: ApplicationPluginParticipant,
    ) -> None:
        try:
            await participant.wait_failed()
        except asyncio.CancelledError:
            if self._orderly_shutdown:
                raise
            raise RuntimeError(f"application plugin failure supervisor cancelled: {name}") from None
        raise RuntimeError(f"application plugin wait_failed() returned normally: {name}")

    async def wait_ready(self) -> None:
        """Await readiness while giving an already-latched failure priority."""
        if not self._entered:
            return
        ready_tasks = [asyncio.create_task(item.participant.wait_ready()) for item in self._entered]
        ready_group: asyncio.Future[list[None]] = asyncio.gather(*ready_tasks)
        failure_wait = asyncio.create_task(self.wait_failed())
        try:
            done, _ = await asyncio.wait(
                set[asyncio.Future[Any]]((ready_group, failure_wait)),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if failure_wait in done:
                ready_group.cancel()
                await asyncio.gather(ready_group, return_exceptions=True)
                await failure_wait
            await ready_group
            await asyncio.sleep(0)
            if any(task.done() for task in self._failure_tasks):
                await self.wait_failed()
        finally:
            if not failure_wait.done():
                failure_wait.cancel()
            await asyncio.gather(failure_wait, return_exceptions=True)
            for task in ready_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*ready_tasks, return_exceptions=True)

    async def wait_failed(self) -> None:
        """Raise the first participant failure, including an early one."""
        pending = list(self._failure_tasks)
        while pending:
            done, _ = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            pending = [task for task in pending if task not in done]
            for task in self._failure_tasks:
                if task not in done:
                    continue
                if task.cancelled() and self._orderly_shutdown:
                    continue
                await task
                raise AssertionError("a completed failure supervisor returned normally")
        await asyncio.Future()

    # @spec ING-VEH-016
    async def mark_serving(self) -> None:
        """Notify every selected plugin that host admission is open."""
        for item in self._entered:
            await item.participant.mark_serving()

    # @spec ING-VEH-017
    async def quiesce_and_drain(self) -> None:
        """Cancel supervision, then independently bound every reverse drain."""
        if self._drained:
            return
        self._drained = True
        await self._stop_supervision()
        errors: list[BaseException] = []
        for item in reversed(self._entered):
            try:
                await _wait_for_with_timeout(
                    item.participant.quiesce_and_drain(),
                    item.shutdown_grace,
                )
            except BaseException as error:
                errors.append(error)
        if errors:
            raise ApplicationPluginError(
                "application plugin drain failed",
                errors,
            )

    @property
    def shutdown_grace(self) -> float:
        """Use the longest finite selected-plugin grace for HTTP draining."""
        if not self._entered:
            return 0.0
        return max(item.shutdown_grace for item in self._entered)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        errors = await self._exit_all(exc_type, exc_value, traceback)
        if errors:
            if exc_value is not None:
                _retain_secondary_errors(exc_value, errors)
                return
            raise ApplicationPluginError(
                "application plugin exit failed",
                errors,
            )

    async def _stop_supervision(self) -> None:
        self._orderly_shutdown = True
        for task in self._failure_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._failure_tasks, return_exceptions=True)

    async def _exit_all(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> list[BaseException]:
        if self._exited:
            return []
        self._exited = True
        await self._stop_supervision()
        errors: list[BaseException] = []
        for item in reversed(self._entered):
            try:
                await _wait_for_with_timeout(
                    item.manager.__aexit__(
                        exc_type,
                        exc_value,
                        traceback,
                    ),
                    item.shutdown_grace,
                )
            except BaseException as error:
                errors.append(error)
        return errors


def _retain_secondary_errors(
    primary: BaseException,
    errors: Sequence[BaseException],
) -> None:
    """Attach cleanup diagnostics without replacing the primary failure."""
    retained = tuple(getattr(primary, "application_plugin_secondary_errors", ()))
    try:
        setattr(
            primary,
            "application_plugin_secondary_errors",
            retained + tuple(errors),
        )
    except (AttributeError, TypeError):
        pass


async def _wait_for_with_timeout(
    awaitable: Awaitable[Any],
    timeout: float,
) -> None:
    """Normalize Python 3.10's asyncio timeout to built-in TimeoutError."""
    try:
        await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("application plugin lifecycle operation timed out") from exc


def _validate_shutdown_grace(value: float) -> float:
    """Validate and normalize one participant's shutdown budget."""
    try:
        grace = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("application plugin shutdown grace must be finite and positive") from exc
    if grace <= 0 or not math.isfinite(grace):
        raise ValueError("application plugin shutdown grace must be finite and positive")
    return grace
