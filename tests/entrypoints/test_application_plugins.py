# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the generic OpenAI application-plugin hook."""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import sys

if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup
from contextlib import asynccontextmanager
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE_PATH = Path(__file__).resolve().parents[2] / "vllm_omni/entrypoints/openai/application_plugins.py"
_SERVE_PATH = Path(__file__).resolve().parents[2] / "vllm_omni/entrypoints/cli/serve.py"
_SPEC = importlib.util.spec_from_file_location(
    "_application_plugins_under_test",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
application_plugins = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = application_plugins
_SPEC.loader.exec_module(application_plugins)

ApplicationAdmissionState = application_plugins.ApplicationAdmissionState
ApplicationPluginContext = application_plugins.ApplicationPluginContext
SelectedApplicationPlugin = application_plugins.SelectedApplicationPlugin


def _host_context(admission):
    app = object()
    engine_client = object()
    session_factory = object()

    def install_asgi_wrapper(wrapper) -> None:
        del wrapper

    return (
        SimpleNamespace(
            app=app,
            engine_client=engine_client,
            session_factory=session_factory,
            serve_args=SimpleNamespace(),
            admission=admission,
            install_asgi_wrapper=install_asgi_wrapper,
        ),
        app,
        engine_client,
        session_factory,
    )


# @spec ING-VEH-003, ING-VEH-016
def test_per_entry_context_preserves_host_identity_and_scopes_opaque_config() -> None:
    """Each plugin gets only generic values and a read-only admission view."""

    class AdmissionView:
        def is_open(self) -> bool:
            return False

    admission = AdmissionView()
    app = object()
    engine_client = object()

    context = ApplicationPluginContext(
        app=app,
        engine_client=engine_client,
        admission=admission,
        install_asgi_wrapper=lambda wrapper: None,
        plugin_name="first",
        config="first.toml",
    )

    assert context.app is app
    assert context.engine_client is engine_client
    assert context.admission is admission
    assert context.plugin_name == "first"
    assert context.config == "first.toml"
    assert {field.name for field in fields(ApplicationPluginContext)} == {
        "app",
        "engine_client",
        "admission",
        "install_asgi_wrapper",
        "plugin_name",
        "config",
    }
    assert not hasattr(context, "session_factory")
    assert not hasattr(context, "serve_args")
    assert not hasattr(context.admission, "open_after_http_listener_bound")
    assert not hasattr(context.admission, "close_before_owner_drain")
    assert not hasattr(context.admission, "close_from_launcher_thread")


# @spec ING-VEH-003
def test_prepared_asgi_slot_repairs_eager_stack_inside_host_boundary_once() -> None:
    """ING-VEH-001/003: an eager stack is rebuilt once around a stable slot."""
    events: list[str] = []

    class Router:
        marker = object()

        async def __call__(self, scope, receive, send) -> None:
            del scope, receive, send
            events.append("router")

    class App:
        def __init__(self) -> None:
            self.router = Router()
            self.build_count = 0
            self.middleware_stack = self.build_middleware_stack()

        def build_middleware_stack(self):
            self.build_count += 1
            inner = self.router

            async def host_middleware(scope, receive, send) -> None:
                events.append("host-middleware")
                await inner(scope, receive, send)

            return host_middleware

    def wrapper(inner):
        async def installed(scope, receive, send) -> None:
            events.append("plugin")
            await inner(scope, receive, send)

        return installed

    app = App()
    original_stack = app.middleware_stack
    slot = application_plugins.prepare_application_plugin_asgi_wrappers(app)
    slot.install(wrapper)
    slot.seal()

    asyncio.run(app.middleware_stack({}, None, None))

    assert app.build_count == 2
    assert app.middleware_stack is not original_stack
    assert events == ["host-middleware", "plugin", "router"]
    assert app.router.marker is Router.marker


# @spec ING-VEH-003, ING-VEH-016
def test_prepared_asgi_slot_preserves_lazy_stack_and_cli_wrapper_order() -> None:
    """ING-VEH-003/016: [A, B] composes as host -> A -> B -> router."""
    events: list[str] = []

    class Router:
        async def __call__(self, scope, receive, send) -> None:
            del scope, receive, send
            events.append("router")

    class App:
        def __init__(self) -> None:
            self.router = Router()
            self.middleware_stack = None
            self.build_count = 0

        def build_middleware_stack(self):
            self.build_count += 1
            inner = self.router

            async def host_middleware(scope, receive, send) -> None:
                events.append("host-middleware")
                await inner(scope, receive, send)

            return host_middleware

    def wrapper(name):
        def wrap(inner):
            async def installed(scope, receive, send) -> None:
                events.append(name)
                await inner(scope, receive, send)

            return installed

        return wrap

    app = App()
    slot = application_plugins.prepare_application_plugin_asgi_wrappers(app)
    slot.install(wrapper("A"))
    slot.install(wrapper("B"))
    slot.seal()
    app.middleware_stack = app.build_middleware_stack()

    asyncio.run(app.middleware_stack({}, None, None))

    assert app.build_count == 1
    assert events == ["host-middleware", "A", "B", "router"]


# @spec ING-VEH-010, ING-VEH-016, ING-VEH-017
def test_prepared_asgi_slot_rejects_late_registration_and_failed_seal() -> None:
    """ING-VEH-010/016/017: sealing is terminal and failure is irreversible."""

    class Router:
        async def __call__(self, scope, receive, send) -> None:
            del scope, receive, send

    app = SimpleNamespace(router=Router(), middleware_stack=None)
    slot = application_plugins.prepare_application_plugin_asgi_wrappers(app)
    slot.seal()
    with pytest.raises(RuntimeError, match="sealed"):
        slot.install(lambda inner: inner)

    failed_app = SimpleNamespace(router=Router(), middleware_stack=None)
    failed_slot = application_plugins.prepare_application_plugin_asgi_wrappers(failed_app)

    def broken_wrapper(inner):
        del inner
        raise RuntimeError("wrapper construction failed")

    failed_slot.install(broken_wrapper)
    with pytest.raises(RuntimeError, match="wrapper construction failed"):
        failed_slot.seal()
    with pytest.raises(RuntimeError, match="failed"):
        failed_slot.install(lambda inner: inner)
    with pytest.raises(RuntimeError, match="failed"):
        failed_slot.seal()


# @spec ING-VEH-010
def test_prepared_asgi_slot_restores_host_if_eager_rebuild_fails() -> None:
    """ING-VEH-010: preparation failure restores the untouched host app."""

    class Router:
        async def __call__(self, scope, receive, send) -> None:
            del scope, receive, send

    class App:
        def __init__(self) -> None:
            self.router = Router()
            self.middleware_stack = object()

        def build_middleware_stack(self):
            raise RuntimeError("host rebuild failed")

    app = App()
    original_router = app.router
    original_stack = app.middleware_stack

    with pytest.raises(RuntimeError, match="host rebuild failed"):
        application_plugins.prepare_application_plugin_asgi_wrappers(app)

    assert app.router is original_router
    assert app.middleware_stack is original_stack


# @spec ING-VEH-016, ING-VEH-019
@pytest.mark.asyncio
async def test_closed_admission_guard_blocks_native_work_but_keeps_ops_live() -> None:
    """The innermost guard projects 503/1013 before the host router."""
    routed: list[tuple[str, str]] = []
    wrapped: list[str] = []
    sent: list[dict[str, object]] = []

    class Router:
        async def __call__(self, scope, receive, send) -> None:
            del receive, send
            routed.append((scope["type"], scope["path"]))

    admission = application_plugins.create_application_admission()
    app = SimpleNamespace(
        router=Router(),
        middleware_stack=None,
        state=SimpleNamespace(application_admission=admission),
    )
    slot = application_plugins.prepare_application_plugin_asgi_wrappers(app)

    def wrapper(inner):
        async def installed(scope, receive, send) -> None:
            wrapped.append(scope["path"])
            await inner(scope, receive, send)

        return installed

    slot.install(wrapper)
    slot.seal()

    async def receive() -> dict[str, object]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    for path in ("/health", "/metrics"):
        await app.router(
            {"type": "http", "method": "GET", "path": path},
            receive,
            send,
        )
    assert routed == [("http", "/health"), ("http", "/metrics")]

    await app.router(
        {"type": "http", "method": "GET", "path": "/v1/models"},
        receive,
        send,
    )
    assert routed == [("http", "/health"), ("http", "/metrics")]
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 503

    sent.clear()
    await app.router(
        {"type": "websocket", "path": "/v1/realtime"},
        receive,
        send,
    )
    assert routed == [("http", "/health"), ("http", "/metrics")]
    assert sent == [{"type": "websocket.close", "code": 1013}]

    await admission.open_after_http_listener_bound()
    await app.router(
        {"type": "http", "method": "GET", "path": "/v1/models"},
        receive,
        send,
    )
    assert routed[-1] == ("http", "/v1/models")
    assert wrapped == [
        "/health",
        "/metrics",
        "/v1/models",
        "/v1/realtime",
        "/v1/models",
    ]


def test_selection_requires_explicit_unique_names_and_keyed_config() -> None:
    """ING-VEH-003/009: selection is explicit, ordered, and duplicate-free."""
    config = application_plugins.validate_application_plugin_options(
        ["first", "second"],
        ["first=first.toml", "second="],
        api_server_worker_count=1,
    )

    assert config == {"first": "first.toml", "second": ""}
    with pytest.raises(ValueError, match="duplicate selected"):
        application_plugins.validate_application_plugin_options(["first", "first"], [], api_server_worker_count=1)
    with pytest.raises(ValueError, match="duplicate config"):
        application_plugins.validate_application_plugin_options(
            ["first"], ["first=a", "first=b"], api_server_worker_count=1
        )
    with pytest.raises(ValueError, match="unselected"):
        application_plugins.validate_application_plugin_options(["first"], ["second=b"], api_server_worker_count=1)


def test_cli_arguments_preserve_repeatable_selection_and_keyed_config() -> None:
    """ING-VEH-009: plugin selection/config is explicit at the CLI boundary."""
    parser = argparse.ArgumentParser()

    application_plugins.add_application_plugin_args(parser)

    args = parser.parse_args(
        [
            "--application-plugin",
            "first",
            "--application-plugin",
            "second",
            "--application-plugin-config",
            "first=first.toml",
            "--application-plugin-config",
            "second=",
            "--ws-max-size",
            "2097152",
        ]
    )
    assert args.application_plugin == ["first", "second"]
    assert args.application_plugin_config == ["first=first.toml", "second="]
    assert args.ws_max_size == 2097152


def test_installation_does_not_activate_an_unselected_entry_point(monkeypatch) -> None:
    """ING-VEH-001/009: discovery never activates an unselected installed plugin."""
    loaded: list[str] = []

    def selected(context):
        del context
        loaded.append("selected")
        return _lifetime()

    def unselected(context):
        del context
        loaded.append("unselected")
        return _lifetime()

    monkeypatch.setattr(
        application_plugins.metadata,
        "entry_points",
        lambda: FakeEntryPoints({"selected": selected, "unselected": unselected}),
    )

    assert application_plugins.discover_application_plugins(["selected"]) == [selected]
    assert loaded == []


def test_discovery_preserves_explicit_cli_order(monkeypatch) -> None:
    """ING-VEH-003: discovery and plugin entry follow CLI selection order."""

    def first(context):
        del context
        return _lifetime()

    def second(context):
        del context
        return _lifetime()

    monkeypatch.setattr(
        application_plugins.metadata,
        "entry_points",
        lambda: FakeEntryPoints({"first": first, "second": second}),
    )

    assert application_plugins.discover_application_plugins(["second", "first"]) == [second, first]


def test_discovery_rejects_ambiguous_installed_entry_point_name(
    monkeypatch,
) -> None:
    """ING-VEH-009: package collisions never depend on discovery order."""

    def plugin(context):
        del context
        return _lifetime()

    class AmbiguousEntryPoints:
        def select(self, *, group: str):
            assert group == application_plugins.APPLICATION_PLUGIN_ENTRY_POINT_GROUP
            return [
                FakeEntryPoint("same", plugin),
                FakeEntryPoint("same", plugin),
            ]

    monkeypatch.setattr(
        application_plugins.metadata,
        "entry_points",
        AmbiguousEntryPoints,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        application_plugins.discover_application_plugins(["same"])


def test_selected_plugin_rejects_multi_worker_before_any_listener_starts() -> None:
    """ING-VEH-015: selected plugins require exactly one API-server worker."""
    with pytest.raises(ValueError, match="one API-server worker"):
        application_plugins.validate_application_plugin_options(["first"], [], api_server_worker_count=2)


def test_omni_cli_keeps_plugin_mode_single_process_and_self_describing() -> None:
    """ING-VEH-009/015: the CLI rejects bad modes but permits explicit one."""
    source = _SERVE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    messages = {
        value.value for value in ast.walk(tree) if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    selected_branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(isinstance(name, ast.Name) and name.id == "selected_plugins" for name in ast.walk(node.test))
    ]

    assert "--application-plugin is unavailable with --headless" in messages
    assert "--application-plugin requires --omni" in messages
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "pop"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "api_server_count"
        for branch in selected_branches
        for statement in branch.body
        for call in ast.walk(statement)
    )


@pytest.mark.asyncio
async def test_host_opens_admission_once_after_composed_listener_readiness() -> None:
    """ING-VEH-016: a host-owned CLOSED->OPEN transition linearizes readiness."""
    admission = application_plugins.create_application_admission()

    assert admission.state is ApplicationAdmissionState.CLOSED
    assert not admission.is_open()
    await admission.open_after_http_listener_bound()
    assert admission.state is ApplicationAdmissionState.OPEN
    assert admission.is_open()


@pytest.mark.asyncio
async def test_lifetime_scopes_config_signals_serving_then_drains_before_reverse_unwind() -> None:
    """ING-VEH-003/010/016/017: host owns per-entry context and lifecycle signals."""
    events: list[str] = []
    admission = FakeAdmission(events)
    host_context, _, _, _ = _host_context(admission)

    def first(context):
        assert context.plugin_name == "first"
        assert context.config == "first.toml"
        assert context.app is host_context.app
        return FakeParticipant(events, "first")

    def second(context):
        assert context.plugin_name == "second"
        assert context.config is None
        assert context.engine_client is host_context.engine_client
        return FakeParticipant(events, "second")

    plugins = [
        SelectedApplicationPlugin("first", first, "first.toml"),
        SelectedApplicationPlugin("second", second, None),
    ]
    async with application_plugins.application_plugin_lifetime(plugins, host_context) as lifetime:
        await host_context.admission.open_after_http_listener_bound()
        await lifetime.mark_serving()
        events.append("serving")
        await host_context.admission.close_before_owner_drain()
        await lifetime.quiesce_and_drain()

    assert events == [
        "enter:first",
        "enter:second",
        "admission:open",
        "serving:first",
        "serving:second",
        "serving",
        "admission:closed",
        "drain:second",
        "drain:first",
        "exit:second",
        "exit:first",
    ]


# @spec ING-VEH-010, ING-VEH-020
@pytest.mark.asyncio
async def test_mark_serving_failure_drains_every_entered_participant() -> None:
    """A failed advertisement drains all entries in reverse CLI order."""
    events: list[str] = []
    admission = FakeAdmission(events)
    host_context, _, _, _ = _host_context(admission)

    class FailingAdvertisement(FakeParticipant):
        async def mark_serving(self) -> None:
            self.events.append(f"serving:{self.name}")
            raise RuntimeError("advertisement failed")

    def first(context):
        del context
        return FakeParticipant(events, "first")

    def second(context):
        del context
        return FailingAdvertisement(events, "second")

    def third(context):
        del context
        return FakeParticipant(events, "third")

    selected = [
        SelectedApplicationPlugin("first", first, None),
        SelectedApplicationPlugin("second", second, None),
        SelectedApplicationPlugin("third", third, None),
    ]
    async with application_plugins.application_plugin_lifetime(
        selected,
        host_context,
    ) as lifetime:
        with pytest.raises(RuntimeError, match="advertisement failed"):
            await lifetime.mark_serving()
        await lifetime.quiesce_and_drain()

    assert "serving:first" in events
    assert "serving:second" in events
    assert "serving:third" not in events
    assert events[5:8] == [
        "drain:third",
        "drain:second",
        "drain:first",
    ]
    assert events[-3:] == ["exit:third", "exit:second", "exit:first"]


# @spec ING-VEH-003, ING-VEH-016
@pytest.mark.asyncio
async def test_lifetime_scopes_each_asgi_installer_to_its_entry() -> None:
    """ING-VEH-003/016: retained installers expire after their plugin enters."""
    retained_installers = []
    events: list[str] = []

    class Router:
        async def __call__(self, scope, receive, send) -> None:
            del scope, receive, send
            events.append("router")

    class InstallingParticipant(FakeParticipant):
        def __init__(self, context) -> None:
            super().__init__(events, context.plugin_name)
            self._context = context

        async def __aenter__(self):
            retained_installers.append(self._context.install_asgi_wrapper)

            def wrapper(inner):
                async def installed(scope, receive, send) -> None:
                    events.append(self.name)
                    await inner(scope, receive, send)

                return installed

            self._context.install_asgi_wrapper(wrapper)
            return await super().__aenter__()

    app = SimpleNamespace(router=Router(), middleware_stack=None)
    slot = application_plugins.prepare_application_plugin_asgi_wrappers(app)
    host_context = SimpleNamespace(
        app=app,
        engine_client=object(),
        session_factory=object(),
        serve_args=SimpleNamespace(),
        admission=FakeAdmission(events),
        install_asgi_wrapper=slot.install,
    )
    plugins = [
        SelectedApplicationPlugin("first", InstallingParticipant, None),
        SelectedApplicationPlugin("second", InstallingParticipant, None),
    ]

    async with application_plugins.application_plugin_lifetime(plugins, host_context):
        assert retained_installers[0] is not retained_installers[1]
        for installer in retained_installers:
            with pytest.raises(RuntimeError, match="entry"):
                installer(lambda inner: inner)
        slot.seal()
        await app.router({}, None, None)

    assert events[:4] == ["enter:first", "enter:second", "first", "second"]
    assert events[4] == "router"


@pytest.mark.asyncio
async def test_startup_failure_rolls_back_entered_plugins_before_engine_exit() -> None:
    """ING-VEH-010/017: failed startup leaves no partially entered plugin."""
    events: list[str] = []
    host_context, _, _, _ = _host_context(FakeAdmission(events))

    def first(context):
        assert context.plugin_name == "first"
        return FakeParticipant(events, "first")

    def broken(context):
        assert context.plugin_name == "broken"
        raise RuntimeError("bind failed")

    plugins = [
        SelectedApplicationPlugin("first", first, None),
        SelectedApplicationPlugin("broken", broken, None),
    ]
    with pytest.raises(RuntimeError, match="bind failed"):
        async with application_plugins.application_plugin_lifetime(plugins, host_context):
            pytest.fail("lifetime must not yield after startup failure")

    assert events == ["enter:first", "exit:first"]


@pytest.mark.asyncio
async def test_no_plugin_is_a_compatibility_noop() -> None:
    """ING-VEH-003: no selection leaves the existing host serving behavior unchanged."""
    events: list[str] = []
    host_context, _, _, _ = _host_context(FakeAdmission(events))

    async with application_plugins.application_plugin_lifetime([], host_context):
        events.append("host-serving")

    assert events == ["host-serving"]


# @spec ING-VEH-017, ING-VEH-022
def test_drain_attempts_every_participant_and_grace_is_finite() -> None:
    """One failed drain cannot strand a later participant."""

    class FailingParticipant(FakeParticipant):
        async def quiesce_and_drain(self) -> None:
            self.events.append(f"drain:{self.name}")
            raise RuntimeError(self.name)

    async def exercise() -> None:
        events: list[str] = []
        admission = FakeAdmission(events)
        host_context, _, _, _ = _host_context(admission)

        def first(context):
            del context
            return FailingParticipant(events, "first")

        def second(context):
            del context
            return FailingParticipant(events, "second")

        plugins = [
            SelectedApplicationPlugin("first", first, None),
            SelectedApplicationPlugin("second", second, None),
        ]
        async with application_plugins.application_plugin_lifetime(plugins, host_context) as lifetime:
            assert lifetime.shutdown_grace == 1.0
            with pytest.raises(BaseExceptionGroup) as error:
                await lifetime.quiesce_and_drain()
            assert len(error.value.exceptions) == 2
        assert events[2:4] == ["drain:second", "drain:first"]

    asyncio.run(exercise())


# @spec ING-VEH-010, ING-VEH-018
def test_nonfinite_participant_shutdown_grace_is_rejected_on_entry() -> None:
    """An invalid grace rolls back before another participant enters."""

    class InfiniteParticipant(FakeParticipant):
        @property
        def shutdown_grace(self) -> float:
            return float("inf")

    async def exercise() -> None:
        events: list[str] = []
        host_context, _, _, _ = _host_context(FakeAdmission(events))

        def invalid(context):
            del context
            return InfiniteParticipant(events, "infinite")

        def must_not_enter(context):
            del context
            return FakeParticipant(events, "later")

        selected = [
            SelectedApplicationPlugin("infinite", invalid, None),
            SelectedApplicationPlugin("later", must_not_enter, None),
        ]
        with pytest.raises(ValueError, match="finite and positive"):
            async with application_plugins.application_plugin_lifetime(
                selected, host_context
            ):
                pytest.fail("invalid grace must fail before the lifetime yields")

        assert events == ["enter:infinite", "exit:infinite"]

    asyncio.run(exercise())


# @spec ING-VEH-017, ING-VEH-022
def test_each_participant_drain_is_timeboxed_and_all_are_attempted() -> None:
    """A hung reverse-first drain cannot prevent remaining attempts."""

    class HangingParticipant(FakeParticipant):
        @property
        def shutdown_grace(self) -> float:
            return 0.01

        async def quiesce_and_drain(self) -> None:
            self.events.append(f"drain:{self.name}")
            await asyncio.Future()

    class FailingParticipant(FakeParticipant):
        @property
        def shutdown_grace(self) -> float:
            return 0.02

        async def quiesce_and_drain(self) -> None:
            self.events.append(f"drain:{self.name}")
            raise RuntimeError(self.name)

    async def exercise() -> None:
        events: list[str] = []
        host_context, _, _, _ = _host_context(FakeAdmission(events))

        def failing(context):
            del context
            return FailingParticipant(events, "failing")

        def hanging(context):
            del context
            return HangingParticipant(events, "hanging")

        selected = [
            SelectedApplicationPlugin("failing", failing, None),
            SelectedApplicationPlugin("hanging", hanging, None),
        ]
        async with application_plugins.application_plugin_lifetime(
            selected, host_context
        ) as lifetime:
            with pytest.raises(BaseExceptionGroup) as error:
                await asyncio.wait_for(
                    lifetime.quiesce_and_drain(),
                    timeout=0.25,
                )
            assert any(
                isinstance(item, asyncio.TimeoutError)
                for item in error.value.exceptions
            )
            assert any(
                isinstance(item, RuntimeError)
                for item in error.value.exceptions
            )

        assert events[2:4] == ["drain:hanging", "drain:failing"]

    asyncio.run(exercise())


def test_plugin_defers_launcher_engine_shutdown_until_context_exit() -> None:
    """ING-VEH-017: signals cannot stop the engine before owner drain."""

    class Engine:
        marker = object()

        def __init__(self) -> None:
            self.calls: list[float | None] = []

        def shutdown(self, timeout: float | None = None) -> None:
            self.calls.append(timeout)

    engine = Engine()
    admission = application_plugins.create_application_admission()
    asyncio.run(admission.open_after_http_listener_bound())
    proxy = application_plugins.defer_engine_shutdown_until_application_drain(engine, admission)
    proxy.shutdown(timeout=12.0)

    assert proxy.marker is engine.marker
    assert proxy.shutdown_requested
    assert proxy.shutdown_timeout == 12.0
    assert engine.calls == []
    assert admission.is_open() is False


class FakeAdmission:
    """Records the host lifecycle ordering."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def open_after_http_listener_bound(self) -> None:
        self.events.append("admission:open")

    async def close_before_owner_drain(self) -> None:
        self.events.append("admission:closed")


class FakeEntryPoint:
    """Small public-entry-point stand-in used to pin selected-only loading."""

    def __init__(self, name: str, plugin) -> None:
        self.name = name
        self._plugin = plugin

    def load(self):
        return self._plugin


class FakeEntryPoints:
    def __init__(self, plugins) -> None:
        self._plugins = plugins

    def select(self, *, group: str):
        assert group == application_plugins.APPLICATION_PLUGIN_ENTRY_POINT_GROUP
        return [FakeEntryPoint(name, plugin) for name, plugin in self._plugins.items()]


class FakeParticipant:
    """Generic selected-plugin participant used to pin host signal order."""

    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    async def __aenter__(self):
        self.events.append(f"enter:{self.name}")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.events.append(f"exit:{self.name}")

    async def mark_serving(self) -> None:
        self.events.append(f"serving:{self.name}")

    async def quiesce_and_drain(self) -> None:
        self.events.append(f"drain:{self.name}")

    @property
    def shutdown_grace(self) -> float:
        return 1.0


@asynccontextmanager
async def _lifetime():
    yield
