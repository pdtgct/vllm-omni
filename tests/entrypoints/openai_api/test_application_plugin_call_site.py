# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first lifecycle contract for the OpenAI application-plugin call site."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from vllm_omni.entrypoints.openai import api_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeServerSocket:
    def close(self) -> None:
        pass


class FakePluginLifetime:
    """Records the generic host lifecycle."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def __aenter__(self):
        self.events.append("plugin-enter")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.events.append("plugin-exit")

    async def mark_serving(self) -> None:
        self.events.append("plugin-serving")

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


class FakeAdmissionView:
    """Read-only projection given to app state and selected plugins."""

    def __init__(self, controller: FakeAdmission) -> None:
        self._controller = controller

    def is_open(self) -> bool:
        return self._controller.open


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
    )


def _install_worker_basics(monkeypatch, events: list[str], *, serve):
    class FakeEngine:
        stage_configs = []

        async def get_supported_tasks(self):
            return ("generate",)

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
        del engine_client, state, args
        events.append("app-state-init")

    async def fake_storage_start():
        pass

    async def fake_listener_bound(*args, **kwargs):
        del args, kwargs
        await asyncio.sleep(0)

    monkeypatch.setattr(api_server, "build_async_omni", fake_build_async_omni)
    monkeypatch.setattr(api_server, "build_openai_app", lambda args, supported_tasks: FastAPI())
    monkeypatch.setattr(api_server, "serve_http", serve)
    monkeypatch.setattr(api_server.STORAGE_MANAGER, "start", fake_storage_start)
    monkeypatch.setattr(api_server, "_get_vllm_config", fake_get_vllm_config)
    monkeypatch.setattr(api_server, "omni_init_app_state", fake_init_app_state)
    monkeypatch.setattr(api_server, "get_uvicorn_log_config", lambda args: None)
    monkeypatch.setattr(
        api_server,
        "_wait_for_http_listener_bound",
        fake_listener_bound,
        raising=False,
    )
    return fake_engine


def _assert_plugin_host_context(
    context,
    engine,
    admission: FakeAdmission,
    events: list[str],
) -> FakePluginLifetime:
    """Assert the host context is generic and admission is read-only."""
    assert context.engine_client is engine
    assert context.admission is admission.view
    assert context.app.state.application_admission is admission.view
    assert callable(context.install_asgi_wrapper)
    assert not hasattr(context, "host_internal")
    assert not hasattr(context, "host_options")
    assert not hasattr(context.admission, "open_after_http_listener_bound")
    assert not hasattr(context.admission, "close_before_owner_drain")
    assert not hasattr(context.admission, "close_from_launcher_thread")
    return FakePluginLifetime(events)


@pytest.mark.asyncio
async def test_listener_bound_signal_is_distinct_from_serving_lifetime() -> None:
    """readiness does not await the server's full lifetime."""
    app = FastAPI()
    serving = asyncio.Event()

    async def serve_forever() -> None:
        await serving.wait()

    serve_task = asyncio.create_task(serve_forever())
    app.state.server = SimpleNamespace(started=False)

    wait_task = asyncio.create_task(api_server._wait_for_http_listener_bound(app, serve_task, timeout=1))
    await asyncio.sleep(0)
    assert not wait_task.done()
    app.state.server.started = True
    await asyncio.wait_for(wait_task, timeout=1)
    assert not serve_task.done()
    serving.set()
    await serve_task


@pytest.mark.asyncio
async def test_selected_plugin_lifecycle_is_nested_around_serving_and_engine(monkeypatch) -> None:
    """The generic host owns composed readiness, grace, and shutdown."""
    events: list[str] = []
    serve_started = asyncio.Event()
    http_shutdown = asyncio.Event()
    admission = FakeAdmission(events)

    async def fake_serve_http(*args, **kwargs):
        launcher_app = args[0]
        assert kwargs["ws_max_size"] == 2097152
        assert kwargs["timeout_graceful_shutdown"] == 7.0
        events.append("http-bound")
        serve_started.set()
        await http_shutdown.wait()
        launcher_app.state.engine_client.shutdown(timeout=1.0)

        async def wait_for_shutdown() -> None:
            events.append("http-shutdown")

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
    await asyncio.wait_for(serve_started.wait(), timeout=2)
    http_shutdown.set()
    await asyncio.wait_for(worker_task, timeout=2)

    assert events.index("engine-enter") < events.index("plugin-enter")
    assert events.index("app-state-init") < events.index("plugin-enter")
    assert events.index("plugin-enter") < events.index("http-bound")
    assert events.index("http-bound") < events.index("admission-open")
    assert events.index("admission-open") < events.index("plugin-serving")
    assert events.index("plugin-serving") < events.index("admission-closed")
    assert events.index("admission-closed") < events.index("plugin-drain")
    assert events.index("plugin-drain") < events.index("http-shutdown")
    assert events.index("plugin-drain") < events.index("plugin-exit")
    assert events.index("plugin-exit") < events.index("engine-exit")


@pytest.mark.asyncio
async def test_selected_plugin_prepares_eager_asgi_slot_before_state_init_and_seals_before_http(
    monkeypatch,
) -> None:
    """host prepares early, entries install, then host seals."""
    events: list[str] = []
    serve_started = asyncio.Event()
    http_shutdown = asyncio.Event()
    slot = FakeASGICompositionSlot(events)

    async def fake_serve_http(*args, **kwargs):
        del args, kwargs
        events.append("http-bound")
        serve_started.set()
        await http_shutdown.wait()

        async def shutdown() -> None:
            events.append("http-shutdown")

        return shutdown()

    _install_worker_basics(monkeypatch, events, serve=fake_serve_http)
    eager_app = FastAPI()
    eager_app.middleware_stack = eager_app.build_middleware_stack()
    monkeypatch.setattr(
        api_server,
        "build_openai_app",
        lambda args, supported_tasks: eager_app,
    )

    def prepare(app):
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
    await asyncio.wait_for(serve_started.wait(), timeout=2)
    http_shutdown.set()
    await asyncio.wait_for(worker_task, timeout=2)

    assert events.index("asgi-prepare") < events.index("app-state-init")
    assert events.index("app-state-init") < events.index("asgi-install")
    assert events.index("asgi-install") < events.index("asgi-seal")
    assert events.index("asgi-seal") < events.index("http-bound")
    assert "asgi-fail" not in events


@pytest.mark.asyncio
async def test_post_bind_plugin_failure_closes_admission_drains_and_cancels_http(monkeypatch) -> None:
    """post-bind startup failure leaves no live HTTP task."""
    events: list[str] = []
    admission = FakeAdmission(events)

    async def fake_serve_http(*args, **kwargs):
        del args, kwargs
        events.append("http-bound")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            events.append("http-cancelled")

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


@pytest.mark.asyncio
async def test_plugin_start_failure_prevents_http_serving(monkeypatch) -> None:
    """plugin startup failure is atomic and rolls back pre-serve."""
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
        lambda app: slot,
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


@pytest.mark.asyncio
async def test_no_selected_plugin_does_not_discover_entry_points(monkeypatch) -> None:
    """no selection preserves the ordinary host path."""
    events: list[str] = []
    discovered = False

    async def immediate_shutdown(*args, **kwargs):
        del args, kwargs
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
