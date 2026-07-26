# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic application-plugin contracts for the OpenAI API server.

This module deliberately contains no compatibility-frontend behavior. It
provides the generic host-owned lifecycle hook used by explicitly selected
application plugins.
"""

from __future__ import annotations

import argparse
import math
import threading
from builtins import BaseExceptionGroup
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from typing import Any, Protocol, Self

APPLICATION_PLUGIN_ENTRY_POINT_GROUP = "vllm_omni.application_plugins"
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


class ApplicationAdmission(Protocol):
    """The host-owned linearization point exposed to application plugins."""

    @property
    def state(self) -> ApplicationAdmissionState:
        """Return the current admission state."""

    def is_open(self) -> bool:
        """Return whether application inference can create an owner."""

    async def open_after_http_listener_bound(self) -> None:
        """Make the one host-owned readiness transition after HTTP binds."""

    async def close_before_owner_drain(self) -> None:
        """Close admission before the process drains application owners."""

    def close_from_launcher_thread(self) -> None:
        """Close admission synchronously when the HTTP launcher gets signal."""


@dataclass(frozen=True)
class ApplicationPluginHostContext:
    """Generic host values shared by all explicitly selected plugins.

    The host owns all values and preserves object identity for ``app``,
    ``engine_client``, and ``session_factory``.
    """

    app: Any
    engine_client: Any
    session_factory: Any
    serve_args: Any
    admission: ApplicationAdmission
    install_asgi_wrapper: ApplicationASGIInstaller


@dataclass(frozen=True)
class ApplicationPluginContext(ApplicationPluginHostContext):
    """One selected plugin's host context and opaque configuration value."""

    plugin_name: str
    config: str | None


class ApplicationPlugin(Protocol):
    """A lifecycle-aware application plugin entry point."""

    config_optional: bool

    def __call__(self, context: ApplicationPluginContext) -> ApplicationPluginParticipant:
        """Return the selected plugin's async lifetime context manager."""


class ApplicationPluginParticipant(
    AbstractAsyncContextManager["ApplicationPluginParticipant"],
    Protocol,
):
    """One selected plugin's generic readiness and drain participant."""

    async def mark_serving(self) -> None:
        """Observe that host application admission is now open."""

    async def quiesce_and_drain(self) -> None:
        """Stop new work and drain this plugin's owners before host exit."""

    @property
    def shutdown_grace(self) -> float:
        """Return this participant's finite shutdown grace in seconds."""


ApplicationPluginEntryPoint = Callable[[ApplicationPluginContext], ApplicationPluginParticipant]


@dataclass(frozen=True)
class SelectedApplicationPlugin:
    """One discovered entry point paired with its validated opaque config."""

    name: str
    entry_point: ApplicationPluginEntryPoint
    config: str | None


class ApplicationPluginLifetime(AbstractAsyncContextManager["ApplicationPluginLifetime"], Protocol):
    """Selected-plugin lifetime controlled by the generic API-server host."""

    async def __aenter__(self) -> Self:
        """Enter selected plugins in explicit CLI order."""

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Unwind every entered plugin in reverse order."""

    async def mark_serving(self) -> None:
        """Tell selected plugins that host admission is now open."""

    async def quiesce_and_drain(self) -> None:
        """Quiesce selected plugins and drain their owners before exit."""

    @property
    def shutdown_grace(self) -> float:
        """Return the finite shared HTTP/plugin shutdown grace."""


def add_application_plugin_args(parser: argparse.ArgumentParser) -> None:
    """Add repeatable generic application-plugin selection CLI arguments."""
    parser.add_argument(
        "--application-plugin",
        action="append",
        default=[],
        metavar="NAME",
        help="Explicitly enable one application plugin entry point (repeatable).",
    )
    parser.add_argument(
        "--application-plugin-config",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Opaque configuration for one selected application plugin (repeatable).",
    )
    parser.add_argument(
        "--ws-max-size",
        type=_positive_integer,
        default=DEFAULT_WS_MAX_SIZE,
        metavar="BYTES",
        help="Maximum accepted WebSocket message size in bytes.",
    )


def _positive_integer(value: str) -> int:
    """Parse one finite positive byte count for the HTTP server."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def validate_application_plugin_options(
    selected_names: Sequence[str],
    config_values: Sequence[str],
    *,
    api_server_worker_count: int,
) -> Mapping[str, str | None]:
    """Validate explicit plugin selection and opaque keyed configuration.

    The contract rejects duplicate selected names, duplicate config keys,
    config for unselected names, and a selected plugin with any API-worker
    count other than one.
    """
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
        entries = entry_points.get(APPLICATION_PLUGIN_ENTRY_POINT_GROUP, ())
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
    return _ApplicationAdmission()


def defer_engine_shutdown_until_application_drain(
    engine: Any,
    admission: ApplicationAdmission,
) -> Any:
    """Proxy launcher shutdown so the engine context exits after owner drain."""
    return _DrainOrderedEngineClient(engine, admission)


# @spec ING-VEH-003, ING-VEH-010, ING-VEH-016
def prepare_application_plugin_asgi_wrappers(
    app: Any,
) -> ApplicationASGIComposition:
    """Prepare one stable plugin slot inside the host middleware boundary."""
    router = getattr(app, "router", None)
    if not callable(router):
        raise TypeError("application host router must be callable")
    middleware_stack = getattr(app, "middleware_stack", None)
    slot = _ApplicationASGICompositionSlot(router)

    try:
        app.router = slot
        if middleware_stack is not None:
            build_middleware_stack = getattr(app, "build_middleware_stack", None)
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

    def __init__(self, host_router: ApplicationASGI) -> None:
        self._host_router = host_router
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
        installed = self._host_router
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
    """Enter plugins in CLI order and unwind them in reverse order.

    The host nests this lifetime inside the engine context, closes admission
    before the shared owner drain, and opens admission only after every
    selected listener and the host HTTP listener are bound.
    """
    return _ManagedApplicationPluginLifetime(plugins, host_context)


class _ApplicationAdmission:
    """Single-process linearization point for application inference admission."""

    def __init__(self) -> None:
        self._state = ApplicationAdmissionState.CLOSED
        self._opened_once = False
        self._lock = threading.Lock()

    @property
    def state(self) -> ApplicationAdmissionState:
        with self._lock:
            return self._state

    def is_open(self) -> bool:
        return self.state is ApplicationAdmissionState.OPEN

    # @spec ING-VEH-016
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
        """Perform the synchronous OPEN→CLOSED shutdown transition."""
        with self._lock:
            self._state = ApplicationAdmissionState.CLOSED


class _DrainOrderedEngineClient:
    """Delegate engine work while suppressing launcher's eager signal stop."""

    def __init__(
        self,
        engine: Any,
        admission: ApplicationAdmission,
    ) -> None:
        self._engine = engine
        self._admission = admission
        self.shutdown_requested = False
        self.shutdown_timeout: float | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def shutdown(self, timeout: float | None = None) -> None:
        """Record shutdown; the enclosing engine context performs it later."""
        self._admission.close_from_launcher_thread()
        self.shutdown_requested = True
        self.shutdown_timeout = timeout


class _ManagedApplicationPluginLifetime:
    """Enter selected plugins in order and coordinate their generic lifecycle."""

    def __init__(
        self,
        plugins: Sequence[SelectedApplicationPlugin],
        host_context: ApplicationPluginHostContext,
    ) -> None:
        self._plugins = plugins
        self._host_context = host_context
        self._participants: list[ApplicationPluginParticipant] = []
        self._exit_stack = AsyncExitStack()

    # @spec ING-VEH-003, ING-VEH-010
    async def __aenter__(self) -> Self:
        try:
            for selected in self._plugins:
                scoped_installer = _EntryScopedApplicationASGIInstaller(self._host_context.install_asgi_wrapper)
                try:
                    context = ApplicationPluginContext(
                        app=self._host_context.app,
                        engine_client=self._host_context.engine_client,
                        session_factory=self._host_context.session_factory,
                        serve_args=self._host_context.serve_args,
                        admission=self._host_context.admission,
                        install_asgi_wrapper=scoped_installer,
                        plugin_name=selected.name,
                        config=selected.config,
                    )
                    participant = selected.entry_point(context)
                    await self._exit_stack.enter_async_context(participant)
                    self._participants.append(participant)
                finally:
                    scoped_installer.close()
        except BaseException:
            await self._exit_stack.aclose()
            raise
        return self

    # @spec ING-VEH-016
    async def mark_serving(self) -> None:
        """Notify every selected plugin that host admission is open."""
        for participant in self._participants:
            await participant.mark_serving()

    # @spec ING-VEH-017
    async def quiesce_and_drain(self) -> None:
        """Ask selected plugins to quiesce before reverse-order exit."""
        errors: list[BaseException] = []
        for participant in reversed(self._participants):
            try:
                await participant.quiesce_and_drain()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("application plugin drain failed", errors)

    @property
    def shutdown_grace(self) -> float:
        """Use the longest finite selected-plugin grace for HTTP draining."""
        if not self._participants:
            return 0.0
        values = [participant.shutdown_grace for participant in self._participants]
        if any(value <= 0 or not math.isfinite(value) for value in values):
            raise ValueError("application plugin shutdown grace must be finite and positive")
        return max(values)

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._exit_stack.__aexit__(exc_type, exc_value, traceback)
