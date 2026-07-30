# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first lifecycle contract for the OpenAI application-plugin call site."""

from __future__ import annotations

import ast
import asyncio
import inspect
from argparse import Namespace
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from vllm_omni.entrypoints.openai import api_server
from vllm_omni.utils.tracking_parser import TrackingNamespace

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[3]
_API_SERVER = _ROOT / "vllm_omni/entrypoints/openai/api_server.py"


class FakeServerSocket:
    def close(self) -> None:
        pass


class FakePluginLifetime:
    """Records the generic host lifecycle."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._failure = asyncio.Event()

    async def __aenter__(self):
        self.events.append("plugin-enter")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.events.append("plugin-exit")

    async def mark_serving(self) -> None:
        self.events.append("plugin-serving")

    async def wait_ready(self) -> None:
        self.events.append("plugin-ready")

    async def wait_failed(self):
        await self._failure.wait()
        raise RuntimeError("plugin failed")

    async def quiesce_and_drain(self) -> None:
        self.events.append("plugin-drain")

    @property
    def shutdown_grace(self) -> float:
        return 1.0


class FakeASGICompositionSlot:
    """Records host ownership of the prepare -> install -> seal transaction."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def install(self, wrapper) -> None:
        del wrapper
        self.events.append("asgi-install")

    def seal(self) -> None:
        self.events.append("asgi-seal")

    def fail(self) -> None:
        self.events.append("asgi-fail")


class NoopASGICompositionSlot:
    def install(self, wrapper) -> None:
        del wrapper

    def seal(self) -> None:
        pass

    def fail(self) -> None:
        pass


class FakeAdmission:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.open = False
        self.view = FakeAdmissionView(self)

    async def open_after_http_listener_bound(self) -> None:
        self.open = True
        self.events.append("admission-open")

    async def close_before_owner_drain(self) -> None:
        self.close_from_launcher_thread()

    def close_from_launcher_thread(self) -> None:
        if self.open:
            self.open = False
            self.events.append("admission-closed")

    def is_open(self) -> bool:
        return self.open


class FakeAdmissionLease:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._events.append("admission-lease-release")


class FakeAdmissionView:
    """Atomic acquisition capability given to selected plugins."""

    def __init__(self, controller: FakeAdmission) -> None:
        self._controller = controller

    def try_acquire(self):
        if not self._controller.open:
            return None
        return FakeAdmissionLease(self._controller.events)


def _args(*, selected_plugins: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        tool_parser_plugin="",
        reasoning_parser_plugin="",
        reasoning_parser=None,
        structured_outputs_config=SimpleNamespace(reasoning_parser=None),
        application_plugin=selected_plugins,
        application_plugin_config=[],
        api_server_worker_count=1,
        enable_ssl_refresh=False,
        host="127.0.0.1",
        port=0,
        uvicorn_log_level="info",
        disable_uvicorn_access_log=True,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=None,
        ssl_ciphers=None,
        h11_max_incomplete_event_size=None,
        h11_max_header_count=None,
        ws_max_size=2097152,
        application_plugin_startup_timeout=60.0,
    )


def _install_worker_basics(monkeypatch, events: list[str], *, serve):
    class FakeEngine:
        stage_configs = []

        async def get_supported_tasks(self):
            return ("generate",)

        async def check_health(self) -> None:
            return None

    fake_engine = FakeEngine()

    @asynccontextmanager
    async def fake_build_async_omni(*args, **kwargs):
        del args, kwargs
        events.append("engine-enter")
        try:
            yield fake_engine
        finally:
            events.append("engine-exit")

    async def fake_get_vllm_config(engine_client):
        del engine_client
        return None

    async def fake_init_app_state(engine_client, state, args):
        del args
        state.engine_client = engine_client
        events.append("app-state-init")

    async def fake_storage_start():
        pass

    monkeypatch.setattr(api_server, "build_async_omni", fake_build_async_omni)
    monkeypatch.setattr(api_server, "build_openai_app", lambda args, supported_tasks: FastAPI())
    monkeypatch.setattr(api_server, "serve_http", serve)
    monkeypatch.setattr(
        api_server,
        "prepare_application_plugin_asgi_wrappers",
        lambda app, **kwargs: NoopASGICompositionSlot(),
        raising=False,
    )
    monkeypatch.setattr(api_server.STORAGE_MANAGER, "start", fake_storage_start)
    monkeypatch.setattr(api_server, "_get_vllm_config", fake_get_vllm_config)
    monkeypatch.setattr(api_server, "omni_init_app_state", fake_init_app_state)
    monkeypatch.setattr(api_server, "get_uvicorn_log_config", lambda args: None)
    return fake_engine


def _assert_plugin_host_context(
    context,
    engine,
    admission: FakeAdmission,
    events: list[str],
) -> FakePluginLifetime:
    """Assert the host context is generic and admission is acquisition-only."""
    assert context.engine_client is engine
    assert context.admission is admission.view
    assert not hasattr(context.app.state, "application_admission")
    assert callable(context.install_asgi_wrapper)
    assert not hasattr(context, "session_factory")
    assert not hasattr(context, "serve_args")
    assert callable(context.admission.try_acquire)
    assert not hasattr(context, "host_internal")
    assert not hasattr(context, "host_options")
    assert not hasattr(context.admission, "is_open")
    assert not hasattr(context.admission, "open_after_http_listener_bound")
    assert not hasattr(context.admission, "close_before_owner_drain")
    assert not hasattr(context.admission, "close_from_launcher_thread")
    return FakePluginLifetime(events)


async def _wait_for_worker_event(
    worker_task: asyncio.Task,
    event: asyncio.Event,
    *,
    timeout: float,
) -> None:
    event_task = asyncio.create_task(event.wait())
    done, _ = await asyncio.wait(
        {worker_task, event_task},
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if worker_task in done:
        event_task.cancel()
        await worker_task
    if event_task not in done:
        event_task.cancel()
        raise TimeoutError("worker did not reach the expected event")
    await event_task


def test_api_server_delegates_listener_and_signal_ownership_to_launcher() -> None:
    """The API worker contains no polling/proxy shutdown workaround."""
    # @spec ING-VEH-017
    source = inspect.getsource(api_server.omni_run_server_worker)

    assert "_wait_for_http_listener_bound" not in source
    assert "server.should_exit" not in source


# @spec ING-VEH-003, ING-VEH-016, ING-VEH-017, ING-VEH-022
@pytest.mark.asyncio
async def test_selected_plugin_lifecycle_is_nested_around_serving_and_engine(monkeypatch) -> None:
    """The generic host owns composed readiness, grace, and shutdown."""
    # @spec ING-VEH-016, ING-VEH-017, ING-VEH-018, ING-VEH-022
    events: list[str] = []
    serve_started = asyncio.Event()
    http_shutdown = asyncio.Event()
    admission = FakeAdmission(events)

    async def fake_serve_http(*args, **kwargs):
        launcher_app = args[0]
        hook = kwargs.pop("lifecycle_hook")
        assert hook is not None
        assert kwargs["ws_max_size"] == 2097152
        assert kwargs["timeout_graceful_shutdown"] == 7.0
        events.append("http-bound")
        serve_started.set()
        await hook.on_bound()
        await http_shutdown.wait()
        hook.on_shutdown_requested()
        await hook.before_http_shutdown()
        events.append("http-shutdown")
        await hook.before_engine_shutdown()

        async def wait_for_shutdown() -> None:
            assert launcher_app.state.engine_client is engine

        return wait_for_shutdown()

    engine = _install_worker_basics(monkeypatch, events, serve=fake_serve_http)

    def optional_entry_point(context):
        del context
        raise AssertionError("test lifetime replaces the plugin entry point")

    optional_entry_point.config_optional = True

    monkeypatch.setattr(api_server, "create_application_admission", lambda: admission, raising=False)
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point for _ in selected],
        raising=False,
    )
    monkeypatch.setattr(
        api_server,
        "application_plugin_lifetime",
        lambda plugins, context: _assert_plugin_host_context(
            context,
            engine,
            admission,
            events,
        ),
        raising=False,
    )

    worker_task = asyncio.create_task(
        api_server.omni_run_server_worker(
            "127.0.0.1:0",
            FakeServerSocket(),
            _args(selected_plugins=["example"]),
            timeout_graceful_shutdown=7.0,
        )
    )
    await _wait_for_worker_event(
        worker_task,
        serve_started,
        timeout=2,
    )
    http_shutdown.set()
    await asyncio.wait_for(worker_task, timeout=2)

    assert events.index("engine-enter") < events.index("plugin-enter")
    assert events.index("app-state-init") < events.index("plugin-enter")
    assert events.index("plugin-enter") < events.index("plugin-ready")
    assert events.index("plugin-ready") < events.index("http-bound")
    assert events.index("plugin-enter") < events.index("http-bound")
    assert events.index("http-bound") < events.index("admission-open")
    assert events.index("admission-open") < events.index("plugin-serving")
    assert events.index("plugin-serving") < events.index("admission-closed")
    assert events.index("admission-closed") < events.index("plugin-drain")
    assert events.index("plugin-drain") < events.index("http-shutdown")
    assert events.index("plugin-drain") < events.index("plugin-exit")
    assert events.index("plugin-exit") < events.index("engine-exit")


# @spec ING-VEH-003, ING-VEH-016
@pytest.mark.asyncio
async def test_selected_plugin_prepares_eager_asgi_slot_before_state_init_and_seals_before_http(
    monkeypatch,
) -> None:
    """ING-VEH-003/016: host prepares early, entries install, then host seals."""
    events: list[str] = []
    serve_started = asyncio.Event()
    http_shutdown = asyncio.Event()
    slot = FakeASGICompositionSlot(events)

    async def fake_serve_http(*args, **kwargs):
        del args
        hook = kwargs.pop("lifecycle_hook")
        assert hook is not None
        events.append("http-bound")
        serve_started.set()
        await hook.on_bound()
        await http_shutdown.wait()
        hook.on_shutdown_requested()
        await hook.before_http_shutdown()
        events.append("http-shutdown")
        await hook.before_engine_shutdown()

        async def shutdown() -> None:
            pass

        return shutdown()

    _install_worker_basics(monkeypatch, events, serve=fake_serve_http)
    eager_app = FastAPI()
    eager_app.middleware_stack = eager_app.build_middleware_stack()
    monkeypatch.setattr(
        api_server,
        "build_openai_app",
        lambda args, supported_tasks: eager_app,
    )

    def prepare(app, *, admission=None):
        del admission
        assert app is eager_app
        assert app.middleware_stack is not None
        events.append("asgi-prepare")
        return slot

    class InstallingLifetime(FakePluginLifetime):
        def __init__(self, context) -> None:
            super().__init__(events)
            self._context = context

        async def __aenter__(self):
            self._context.install_asgi_wrapper(lambda inner: inner)
            return await super().__aenter__()

    def optional_entry_point(context):
        del context
        raise AssertionError("test lifetime replaces the plugin entry point")

    optional_entry_point.config_optional = True
    monkeypatch.setattr(
        api_server,
        "prepare_application_plugin_asgi_wrappers",
        prepare,
        raising=False,
    )
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point],
    )
    monkeypatch.setattr(
        api_server,
        "application_plugin_lifetime",
        lambda plugins, context: InstallingLifetime(context),
    )

    worker_task = asyncio.create_task(
        api_server.omni_run_server_worker(
            "127.0.0.1:0",
            FakeServerSocket(),
            _args(selected_plugins=["example"]),
        )
    )
    await _wait_for_worker_event(
        worker_task,
        serve_started,
        timeout=2,
    )
    http_shutdown.set()
    await asyncio.wait_for(worker_task, timeout=2)

    assert events.index("asgi-prepare") < events.index("app-state-init")
    assert events.index("app-state-init") < events.index("asgi-install")
    assert events.index("asgi-install") < events.index("asgi-seal")
    assert events.index("asgi-seal") < events.index("http-bound")
    assert "asgi-fail" not in events


@pytest.mark.asyncio
async def test_post_bind_plugin_failure_closes_admission_drains_and_cancels_http(monkeypatch) -> None:
    """ING-VEH-010/017: post-bind startup failure leaves no live HTTP task."""
    events: list[str] = []
    admission = FakeAdmission(events)

    async def fake_serve_http(*args, **kwargs):
        del args
        hook = kwargs.pop("lifecycle_hook")
        assert hook is not None
        events.append("http-bound")
        try:
            await hook.on_bound()
        finally:
            hook.on_shutdown_requested()
            await hook.before_http_shutdown()
            events.append("http-cancelled")
            await hook.before_engine_shutdown()

        async def shutdown() -> None:
            pass

        return shutdown()

    class FailingMarkLifetime(FakePluginLifetime):
        async def mark_serving(self) -> None:
            self.events.append("plugin-serving")
            raise RuntimeError("plugin readiness failed")

    def optional_entry_point(context):
        del context
        raise AssertionError("test lifetime replaces the plugin entry point")

    optional_entry_point.config_optional = True

    _install_worker_basics(monkeypatch, events, serve=fake_serve_http)
    monkeypatch.setattr(api_server, "create_application_admission", lambda: admission, raising=False)
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point],
        raising=False,
    )
    monkeypatch.setattr(
        api_server,
        "application_plugin_lifetime",
        lambda plugins, context: FailingMarkLifetime(events),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="plugin readiness failed"):
        await api_server.omni_run_server_worker("127.0.0.1:0", FakeServerSocket(), _args(selected_plugins=["example"]))

    assert events.index("http-bound") < events.index("admission-open")
    assert events.index("admission-open") < events.index("plugin-serving")
    assert events.index("plugin-serving") < events.index("admission-closed")
    assert events.index("admission-closed") < events.index("http-cancelled")
    assert events.index("admission-closed") < events.index("plugin-drain")
    assert events.index("plugin-drain") < events.index("plugin-exit")
    assert events.index("plugin-exit") < events.index("engine-exit")


# @spec ING-VEH-010
@pytest.mark.asyncio
async def test_plugin_start_failure_prevents_http_serving(monkeypatch) -> None:
    """ING-VEH-010: plugin startup failure is atomic and rolls back pre-serve."""
    events: list[str] = []

    async def must_not_serve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("HTTP serving started after plugin startup failure")

    class FailingLifetime(FakePluginLifetime):
        async def __aenter__(self):
            self.events.append("plugin-enter")
            raise RuntimeError("plugin bind failed")

    def optional_entry_point(context):
        del context
        raise AssertionError("test lifetime replaces the plugin entry point")

    optional_entry_point.config_optional = True

    _install_worker_basics(monkeypatch, events, serve=must_not_serve)
    slot = FakeASGICompositionSlot(events)
    monkeypatch.setattr(
        api_server,
        "prepare_application_plugin_asgi_wrappers",
        lambda app, **kwargs: slot,
        raising=False,
    )
    monkeypatch.setattr(api_server, "create_application_admission", lambda: FakeAdmission(events), raising=False)
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point],
        raising=False,
    )
    monkeypatch.setattr(
        api_server,
        "application_plugin_lifetime",
        lambda plugins, context: FailingLifetime(events),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="plugin bind failed"):
        await api_server.omni_run_server_worker("127.0.0.1:0", FakeServerSocket(), _args(selected_plugins=["example"]))

    assert "engine-enter" in events
    assert "asgi-fail" in events
    assert "engine-exit" in events


# @spec ING-VEH-009, ING-VEH-010
@pytest.mark.asyncio
async def test_missing_required_config_fails_before_entry_or_http(
    monkeypatch,
) -> None:
    """An absent optionality declaration defaults to required config."""
    events: list[str] = []

    async def must_not_serve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("HTTP started after config validation failed")

    def required_entry(context):
        del context
        raise AssertionError("entry ran without its required configuration")

    _install_worker_basics(monkeypatch, events, serve=must_not_serve)
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [required_entry],
        raising=False,
    )

    with pytest.raises(ValueError, match="requires configuration"):
        await api_server.omni_run_server_worker(
            "127.0.0.1:0",
            FakeServerSocket(),
            _args(selected_plugins=["required"]),
        )

    assert "plugin-enter" not in events
    assert "http-bound" not in events


# @spec ING-VEH-003
@pytest.mark.asyncio
async def test_no_selected_plugin_does_not_discover_entry_points(monkeypatch) -> None:
    """ING-VEH-003/009: no selection preserves the ordinary host path."""
    events: list[str] = []
    discovered = False

    async def immediate_shutdown(*args, **kwargs):
        del kwargs
        launcher_app = args[0]
        assert not hasattr(
            launcher_app.state,
            "application_admission",
        )
        assert not any(getattr(route, "path", None) == "/live" for route in launcher_app.routes)
        response = await api_server.health(SimpleNamespace(app=launcher_app))
        assert response.status_code == 200
        return asyncio.create_task(asyncio.sleep(0))

    def fail_if_discovered(selected):
        del selected
        nonlocal discovered
        discovered = True
        raise AssertionError("unselected entry points must not be discovered")

    def fail_if_prepared(app):
        del app
        raise AssertionError("unselected plugins must not alter the ASGI stack")

    _install_worker_basics(monkeypatch, events, serve=immediate_shutdown)
    monkeypatch.setattr(api_server, "discover_application_plugins", fail_if_discovered, raising=False)
    monkeypatch.setattr(
        api_server,
        "prepare_application_plugin_asgi_wrappers",
        fail_if_prepared,
        raising=False,
    )

    await api_server.omni_run_server_worker("127.0.0.1:0", FakeServerSocket(), _args(selected_plugins=[]))

    assert not discovered


# @spec ING-VEH-001, ING-VEH-003, ING-VEH-014
def test_api_server_has_no_model_or_frontend_specific_plugin_dependency() -> None:
    """The generic host imports neither frontend nor model-specific seams."""
    tree = ast.parse(_API_SERVER.read_text())
    imported_modules = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    imported_symbols = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names
    }

    assert not [name for name in imported_modules if name.startswith(("grpc", "riva", "nvidia_riva"))]
    assert "vllm_omni.entrypoints.nemotron_session" not in imported_modules
    assert "NemotronSessionFactory" not in imported_symbols


@pytest.mark.asyncio
async def test_endpoint_plugin_phase_b_uses_retained_instances_without_application_selection(
    monkeypatch,
) -> None:
    """Omni completes upstream phase B on the exact phase-A objects."""
    # @spec ING-VEH-023
    events: list[str] = []

    class EndpointPlugin:
        def __init__(self, name: str) -> None:
            self.name = name

        async def init_state(self, engine_client, state, args) -> None:
            del args
            assert engine_client is engine
            assert state.ordinary_state_ready
            events.append(f"endpoint-state:{self.name}:{id(self)}")

    first = EndpointPlugin("first")
    second = EndpointPlugin("second")
    app = FastAPI()
    app.state.endpoint_plugins = [first, second]

    async def immediate_shutdown(*args, **kwargs):
        del args, kwargs
        events.append("http")
        return asyncio.create_task(asyncio.sleep(0))

    engine = _install_worker_basics(
        monkeypatch,
        events,
        serve=immediate_shutdown,
    )
    monkeypatch.setattr(
        api_server,
        "build_openai_app",
        lambda args, supported_tasks: app,
    )

    async def init_ordinary_state(engine_client, state, args) -> None:
        del args
        assert engine_client is engine
        state.ordinary_state_ready = True
        events.append("ordinary-state")

    monkeypatch.setattr(api_server, "omni_init_app_state", init_ordinary_state)

    await api_server.omni_run_server_worker(
        "127.0.0.1:0",
        FakeServerSocket(),
        _args(selected_plugins=[]),
    )

    assert events.count(f"endpoint-state:first:{id(first)}") == 1
    assert events.count(f"endpoint-state:second:{id(second)}") == 1
    assert events.index("ordinary-state") < events.index(f"endpoint-state:first:{id(first)}")
    assert events.index(f"endpoint-state:first:{id(first)}") < events.index(f"endpoint-state:second:{id(second)}")
    assert events.index(f"endpoint-state:second:{id(second)}") < events.index("http")


@pytest.mark.asyncio
async def test_application_server_arguments_do_not_enter_async_omni(
    monkeypatch,
) -> None:
    """Application-plugin and WebSocket host policy stay API-server-only."""
    # @spec ING-VEH-003
    events: list[str] = []
    observed_engine_args = None

    async def immediate_shutdown(*args, **kwargs):
        del args
        hook = kwargs.pop("lifecycle_hook", None)
        if hook is None:
            await asyncio.sleep(0.05)
        else:
            await hook.on_bound()
            hook.on_shutdown_requested(RuntimeError("test shutdown"))
            await hook.before_http_shutdown()
            await hook.before_engine_shutdown()

        async def shutdown() -> None:
            pass

        return shutdown()

    engine = _install_worker_basics(
        monkeypatch,
        events,
        serve=immediate_shutdown,
    )

    @asynccontextmanager
    async def capture_build_async_omni(args, **kwargs):
        del kwargs
        nonlocal observed_engine_args
        observed_engine_args = args
        yield engine

    monkeypatch.setattr(
        api_server,
        "build_async_omni",
        capture_build_async_omni,
    )

    def optional_entry_point(context):
        del context
        return FakePluginLifetime(events)

    optional_entry_point.config_optional = True
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point],
    )
    args = _args(selected_plugins=["application"])

    await api_server.omni_run_server_worker(
        "127.0.0.1:0",
        FakeServerSocket(),
        args,
    )

    assert observed_engine_args is not None
    assert not hasattr(observed_engine_args, "application_plugin")
    assert not hasattr(observed_engine_args, "application_plugin_config")
    assert not hasattr(
        observed_engine_args,
        "application_plugin_startup_timeout",
    )
    assert not hasattr(observed_engine_args, "ws_max_size")
    assert args.application_plugin == ["application"]


def test_application_server_argument_projection_supports_tracking_namespace() -> None:
    """The real CLI namespace remains usable after server-only filtering."""
    # @spec ING-VEH-003
    args = TrackingNamespace(
        Namespace(
            model="example/model",
            application_plugin=["application"],
            application_plugin_config=["application=config.json"],
            application_plugin_startup_timeout=60.0,
            ws_max_size=2097152,
        ),
        frozenset(
            {
                "model",
                "application_plugin",
                "application_plugin_config",
                "application_plugin_startup_timeout",
                "ws_max_size",
            }
        ),
    )

    engine_args = api_server._engine_args_without_application_options(args)

    assert isinstance(engine_args, TrackingNamespace)
    assert engine_args.get_explicit_kwargs_dict() == {"model": "example/model"}
    assert args.application_plugin == ["application"]
    assert args.application_plugin_config == ["application=config.json"]


@pytest.mark.asyncio
async def test_endpoint_plugin_phase_b_failure_aborts_before_application_discovery_or_http(
    monkeypatch,
) -> None:
    """A phase-B failure is an ordinary startup failure."""
    # @spec ING-VEH-010, ING-VEH-023
    events: list[str] = []
    app = FastAPI()

    class BrokenEndpointPlugin:
        async def init_state(self, engine_client, state, args) -> None:
            del engine_client, state, args
            events.append("endpoint-state")
            raise RuntimeError("endpoint state failed")

    app.state.endpoint_plugins = [BrokenEndpointPlugin()]

    async def must_not_serve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("HTTP started after endpoint phase-B failure")

    _install_worker_basics(monkeypatch, events, serve=must_not_serve)
    monkeypatch.setattr(
        api_server,
        "build_openai_app",
        lambda args, supported_tasks: app,
    )

    discovered = False

    def must_not_discover(selected):
        del selected
        nonlocal discovered
        discovered = True
        raise AssertionError("application discovery preceded endpoint phase B")

    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        must_not_discover,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="endpoint state failed"):
        await api_server.omni_run_server_worker(
            "127.0.0.1:0",
            FakeServerSocket(),
            _args(selected_plugins=["application"]),
        )

    assert events == ["engine-enter", "app-state-init", "endpoint-state", "engine-exit"]
    assert not discovered


@pytest.mark.asyncio
async def test_host_operations_route_collision_fails_before_plugin_entry_or_http(
    monkeypatch,
) -> None:
    """Endpoint routes cannot shadow host readiness, liveness, or metrics."""
    # @spec ING-VEH-010, ING-VEH-020, ING-VEH-023
    events: list[str] = []
    app = FastAPI()

    @app.get("/live")
    async def shadow_live():
        return {"shadowed": True}

    async def must_not_serve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("HTTP started with a shadowed operations route")

    _install_worker_basics(monkeypatch, events, serve=must_not_serve)
    monkeypatch.setattr(
        api_server,
        "build_openai_app",
        lambda args, supported_tasks: app,
    )

    def optional_entry_point(context):
        del context
        raise AssertionError("participant entered after route collision")

    optional_entry_point.config_optional = True
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point],
    )

    with pytest.raises(RuntimeError, match="operations route"):
        await api_server.omni_run_server_worker(
            "127.0.0.1:0",
            FakeServerSocket(),
            _args(selected_plugins=["application"]),
        )

    assert "plugin-enter" not in events
    assert "http-bound" not in events


def test_builtin_health_replacement_preserves_plugin_collision_detection() -> None:
    """Only vLLM's built-in health route may be replaced by Omni health."""
    # @spec ING-VEH-010, ING-VEH-020, ING-VEH-023
    app = FastAPI()
    app.add_api_route(
        "/health",
        api_server.vllm_health,
        methods=["GET"],
    )

    api_server._remove_route_from_app(
        app,
        "/health",
        {"GET"},
        endpoint=api_server.vllm_health,
    )
    app.include_router(api_server.router)
    api_server._install_and_validate_application_operations_routes(app)

    health_routes = [
        route
        for route in app.routes
        if getattr(route, "path", None) == "/health"
    ]
    assert len(health_routes) == 1
    assert health_routes[0].endpoint is api_server.health

    shadowed_app = FastAPI()
    shadowed_app.add_api_route(
        "/health",
        api_server.vllm_health,
        methods=["GET"],
    )

    @shadowed_app.get("/health")
    async def shadow_health():
        return {"shadowed": True}

    api_server._remove_route_from_app(
        shadowed_app,
        "/health",
        {"GET"},
        endpoint=api_server.vllm_health,
    )
    shadowed_app.include_router(api_server.router)

    with pytest.raises(RuntimeError, match="operations route collision: /health"):
        api_server._install_and_validate_application_operations_routes(
            shadowed_app
        )


@pytest.mark.asyncio
async def test_observer_and_orchestrator_sink_exist_before_participant_entry(
    monkeypatch,
) -> None:
    """Ordinary state installation remains before downstream inheritance."""
    # @spec ING-VEH-018, PORT-OBS-003, PORT-OBS-009
    events: list[str] = []
    app = FastAPI()

    async def immediate_shutdown(*args, **kwargs):
        del args
        events.append("http")
        hook = kwargs.pop("lifecycle_hook", None)
        if hook is None:
            await asyncio.sleep(0.05)
        else:
            await hook.on_bound()
            hook.on_shutdown_requested(RuntimeError("test shutdown"))
            await hook.before_http_shutdown()
            await hook.before_engine_shutdown()

        async def shutdown() -> None:
            pass

        return shutdown()

    engine = _install_worker_basics(
        monkeypatch,
        events,
        serve=immediate_shutdown,
    )
    monkeypatch.setattr(
        api_server,
        "build_openai_app",
        lambda args, supported_tasks: app,
    )

    async def init_ordinary_state(engine_client, state, args) -> None:
        del args
        assert engine_client is engine
        state.streaming_observer = object()
        state.streaming_batch_sink = object()
        events.append("ordinary-state")

    monkeypatch.setattr(api_server, "omni_init_app_state", init_ordinary_state)

    def optional_entry_point(context):
        assert context.app.state.streaming_observer is not None
        assert context.app.state.streaming_batch_sink is not None
        return FakePluginLifetime(events)

    optional_entry_point.config_optional = True
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point],
    )

    await api_server.omni_run_server_worker(
        "127.0.0.1:0",
        FakeServerSocket(),
        _args(selected_plugins=["application"]),
    )

    assert events.index("ordinary-state") < events.index("plugin-enter")


@pytest.mark.asyncio
async def test_one_startup_deadline_covers_readiness_and_prevents_http(
    monkeypatch,
) -> None:
    """An unready participant exhausts one server-owned startup budget."""
    # @spec ING-VEH-010, ING-VEH-018
    events: list[str] = []

    async def must_not_serve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("HTTP started before participant readiness")

    class NeverReady(FakePluginLifetime):
        async def wait_ready(self) -> None:
            self.events.append("plugin-ready-wait")
            await asyncio.Future()

    def optional_entry_point(context):
        del context
        return NeverReady(events)

    optional_entry_point.config_optional = True
    _install_worker_basics(monkeypatch, events, serve=must_not_serve)
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point],
    )
    args = _args(selected_plugins=["application"])
    args.application_plugin_startup_timeout = 0.01

    with pytest.raises(TimeoutError):
        await api_server.omni_run_server_worker(
            "127.0.0.1:0",
            FakeServerSocket(),
            args,
        )

    assert "plugin-ready-wait" in events
    assert "plugin-exit" in events
    assert "http-bound" not in events


@pytest.mark.asyncio
async def test_startup_deadline_is_not_reset_for_each_sequential_entry(
    monkeypatch,
) -> None:
    """Two individually-fast entries may still exhaust the one deadline."""
    # @spec ING-VEH-010, ING-VEH-018
    events: list[str] = []

    async def must_not_serve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("HTTP started after the aggregate budget expired")

    class SlowEntry(FakePluginLifetime):
        async def __aenter__(self):
            self.events.append("entry-start")
            await asyncio.sleep(0.04)
            return await super().__aenter__()

    def entry_point(context):
        del context
        return SlowEntry(events)

    entry_point.config_optional = True
    _install_worker_basics(monkeypatch, events, serve=must_not_serve)
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [entry_point, entry_point],
    )
    args = _args(selected_plugins=["first", "second"])
    args.application_plugin_startup_timeout = 0.06

    with pytest.raises(TimeoutError):
        await api_server.omni_run_server_worker(
            "127.0.0.1:0",
            FakeServerSocket(),
            args,
        )

    assert events.count("entry-start") == 2
    assert events.count("plugin-enter") == 1
    assert events.count("plugin-exit") == 1


@pytest.mark.asyncio
async def test_failure_wins_the_readiness_race_and_rolls_back(
    monkeypatch,
) -> None:
    """A participant death cannot hide behind the startup timeout."""
    # @spec ING-VEH-010, ING-VEH-017, ING-VEH-018
    events: list[str] = []

    async def must_not_serve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("HTTP started after a readiness-time failure")

    class FailsWhileWaiting(FakePluginLifetime):
        async def wait_ready(self) -> None:
            self.events.append("plugin-ready-wait")
            await asyncio.Future()

        async def wait_failed(self):
            await asyncio.sleep(0)
            raise RuntimeError("failed before ready")

    def entry_point(context):
        del context
        return FailsWhileWaiting(events)

    entry_point.config_optional = True
    _install_worker_basics(monkeypatch, events, serve=must_not_serve)
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [entry_point],
    )
    args = _args(selected_plugins=["application"])
    args.application_plugin_startup_timeout = 1.0

    with pytest.raises(RuntimeError, match="failed before ready"):
        await api_server.omni_run_server_worker(
            "127.0.0.1:0",
            FakeServerSocket(),
            args,
        )

    assert "plugin-ready-wait" in events
    assert "plugin-exit" in events
    assert "http-bound" not in events


@pytest.mark.asyncio
async def test_post_start_failure_closes_admission_and_stops_http(
    monkeypatch,
) -> None:
    """The armed failure awaitable drives coordinated shutdown."""
    # @spec ING-VEH-017, ING-VEH-020, ING-VEH-022
    events: list[str] = []
    admission = FakeAdmission(events)

    class FailsAfterServing(FakePluginLifetime):
        def __init__(self, events: list[str]) -> None:
            super().__init__(events)
            self._fail = asyncio.Event()

        async def mark_serving(self) -> None:
            await super().mark_serving()
            self._fail.set()

        async def wait_failed(self):
            await self._fail.wait()
            raise RuntimeError("listener died")

    async def serve_until_cancelled(*args, **kwargs):
        del args
        hook = kwargs.pop("lifecycle_hook")
        assert hook is not None
        events.append("http-bound")
        try:
            await hook.on_bound()
            await hook.wait_failed()
        finally:
            hook.on_shutdown_requested()
            await hook.before_http_shutdown()
            events.append("http-cancelled")
            await hook.before_engine_shutdown()

    def optional_entry_point(context):
        del context
        return FailsAfterServing(events)

    optional_entry_point.config_optional = True
    _install_worker_basics(monkeypatch, events, serve=serve_until_cancelled)
    monkeypatch.setattr(
        api_server,
        "create_application_admission",
        lambda: admission,
        raising=False,
    )
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point],
    )

    with pytest.raises(RuntimeError, match="listener died"):
        await asyncio.wait_for(
            api_server.omni_run_server_worker(
                "127.0.0.1:0",
                FakeServerSocket(),
                _args(selected_plugins=["application"]),
            ),
            timeout=0.5,
        )

    assert events.index("plugin-serving") < events.index("admission-closed")
    assert events.index("admission-closed") < events.index("plugin-drain")
    assert events.index("plugin-drain") < events.index("http-cancelled")
    assert events.index("http-cancelled") < events.index("plugin-exit")
    assert events.index("plugin-exit") < events.index("engine-exit")


@pytest.mark.asyncio
async def test_launcher_hook_orders_shutdown_without_polling_server_state(
    monkeypatch,
) -> None:
    """The owning launcher exposes the two structural shutdown barriers."""
    # @spec ING-VEH-017, ING-VEH-022
    events: list[str] = []
    admission = FakeAdmission(events)

    async def lifecycle_aware_serve(*args, **kwargs):
        launcher_app = args[0]
        hook = kwargs.pop("lifecycle_hook")
        assert hook is not None
        live_route = next(route for route in launcher_app.routes if getattr(route, "path", None) == "/live")
        before_ready = await api_server.health(SimpleNamespace(app=launcher_app))
        assert before_ready.status_code == 503
        events.append("http-bound")
        await hook.on_bound()
        ready = await api_server.health(SimpleNamespace(app=launcher_app))
        assert ready.status_code == 200
        events.append("shutdown-signal")
        hook.on_shutdown_requested(RuntimeError("signal"))
        draining = await api_server.health(SimpleNamespace(app=launcher_app))
        assert draining.status_code == 503
        live = await live_route.endpoint()
        assert live.status_code == 200
        events.append("shutdown-requested-return")
        await hook.before_http_shutdown()
        events.append("http-shutdown")
        await hook.before_engine_shutdown()
        events.append("pre-engine-released")

        async def already_shutdown() -> None:
            pass

        return already_shutdown()

    class CoordinatedLifetime(FakePluginLifetime):
        async def __aexit__(self, exc_type, exc_value, traceback) -> None:
            await super().__aexit__(exc_type, exc_value, traceback)

    def optional_entry_point(context):
        del context
        raise AssertionError("test lifetime replaces the entry point")

    optional_entry_point.config_optional = True
    _install_worker_basics(
        monkeypatch,
        events,
        serve=lifecycle_aware_serve,
    )
    monkeypatch.setattr(
        api_server,
        "create_application_admission",
        lambda: admission,
        raising=False,
    )
    monkeypatch.setattr(
        api_server,
        "discover_application_plugins",
        lambda selected: [optional_entry_point],
    )
    monkeypatch.setattr(
        api_server,
        "application_plugin_lifetime",
        lambda plugins, context: CoordinatedLifetime(events),
    )

    await api_server.omni_run_server_worker(
        "127.0.0.1:0",
        FakeServerSocket(),
        _args(selected_plugins=["application"]),
    )

    assert events.index("plugin-ready") < events.index("http-bound")
    assert events.index("http-bound") < events.index("admission-open")
    assert events.index("admission-open") < events.index("plugin-serving")
    assert events.index("shutdown-signal") < events.index("admission-closed")
    assert events.index("admission-closed") < events.index("shutdown-requested-return")
    assert events.index("admission-closed") < events.index("plugin-drain")
    assert events.index("plugin-drain") < events.index("http-shutdown")
    assert events.index("http-shutdown") < events.index("plugin-exit")
    assert events.index("plugin-exit") < events.index("pre-engine-released")
    assert events.index("pre-engine-released") < events.index("engine-exit")
