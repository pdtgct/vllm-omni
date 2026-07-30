# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the generic OpenAI application-plugin hook."""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import sys
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
    """Each plugin gets generic values and atomic admission acquisition."""
    # @spec ING-VEH-003, ING-VEH-019

    class Admission:
        def try_acquire(self):
            return None

    admission = Admission()
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
    assert not hasattr(context, "host_internal")
    assert not hasattr(context, "host_options")
    assert callable(context.admission.try_acquire)
    assert not hasattr(context.admission, "is_open")
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
    """The guard acquires atomically and preserves operational scopes."""
    # @spec ING-VEH-019, ING-VEH-020, ING-VEH-024
    routed: list[tuple[str, str]] = []
    wrapped: list[str] = []
    sent: list[dict[str, object]] = []
    acquired: list[str] = []
    released: list[str] = []

    class Router:
        async def __call__(self, scope, receive, send) -> None:
            del receive, send
            if scope["path"] not in {"/health", "/live", "/metrics"}:
                assert acquired == [scope["path"]]
                assert released == []
            routed.append((scope["type"], scope["path"]))

    class Lease:
        def __init__(self, path: str) -> None:
            self._path = path

        def release(self) -> None:
            released.append(self._path)

    class Admission:
        def __init__(self) -> None:
            self.open = False
            self.path = ""

        def try_acquire(self):
            if not self.open:
                return None
            acquired.append(self.path)
            return Lease(self.path)

    admission = Admission()
    app = SimpleNamespace(
        router=Router(),
        middleware_stack=None,
        state=SimpleNamespace(),
    )
    slot = application_plugins.prepare_application_plugin_asgi_wrappers(
        app,
        admission=admission,
    )

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

    for path in ("/health", "/live", "/metrics"):
        await app.router(
            {"type": "http", "method": "GET", "path": path},
            receive,
            send,
        )
    assert routed == [
        ("http", "/health"),
        ("http", "/live"),
        ("http", "/metrics"),
    ]

    await app.router(
        {"type": "http", "method": "GET", "path": "/v1/models"},
        receive,
        send,
    )
    assert routed == [
        ("http", "/health"),
        ("http", "/live"),
        ("http", "/metrics"),
    ]
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 503

    sent.clear()
    await app.router(
        {
            "type": "websocket",
            "path": "/v1/realtime",
            "extensions": {"websocket.http.response": {}},
        },
        receive,
        send,
    )
    assert routed == [
        ("http", "/health"),
        ("http", "/live"),
        ("http", "/metrics"),
    ]
    assert sent[0]["type"] == "websocket.http.response.start"
    assert sent[0]["status"] == 503
    assert sent[-1]["type"] == "websocket.http.response.body"

    admission.open = True
    admission.path = "/v1/models"
    acquired.clear()
    released.clear()
    await app.router(
        {"type": "http", "method": "GET", "path": "/v1/models"},
        receive,
        send,
    )
    assert routed[-1] == ("http", "/v1/models")
    assert acquired == ["/v1/models"]
    assert released == ["/v1/models"]
    assert wrapped == [
        "/health",
        "/live",
        "/metrics",
        "/v1/models",
        "/v1/realtime",
        "/v1/models",
    ]


@pytest.mark.asyncio
async def test_native_guard_releases_lease_when_the_inner_scope_fails() -> None:
    """A handler exception cannot leak an admitted owner."""
    # @spec ING-VEH-019
    released = 0

    class Lease:
        def release(self) -> None:
            nonlocal released
            released += 1

    class Admission:
        def try_acquire(self):
            return Lease()

    class BrokenRouter:
        async def __call__(self, scope, receive, send) -> None:
            del scope, receive, send
            raise RuntimeError("handler failed")

    app = SimpleNamespace(
        router=BrokenRouter(),
        middleware_stack=None,
        state=SimpleNamespace(),
    )
    slot = application_plugins.prepare_application_plugin_asgi_wrappers(
        app,
        admission=Admission(),
    )
    slot.seal()

    with pytest.raises(RuntimeError, match="handler failed"):
        await app.router(
            {"type": "http", "method": "POST", "path": "/v1/work"},
            None,
            None,
        )

    assert released == 1


@pytest.mark.asyncio
async def test_raw_wrapper_claims_or_delegates_without_double_acquisition() -> None:
    """Claimed and passthrough scopes each acquire exactly once."""
    # @spec ING-VEH-019, ING-VEH-021, ING-VEH-024
    acquired: list[str] = []
    released: list[str] = []
    routed: list[str] = []

    class Lease:
        def __init__(self, path: str) -> None:
            self._path = path

        def release(self) -> None:
            released.append(self._path)

    class Admission:
        def __init__(self) -> None:
            self.path = ""

        def try_acquire(self):
            acquired.append(self.path)
            return Lease(self.path)

    admission = Admission()

    class Router:
        async def __call__(self, scope, receive, send) -> None:
            del receive, send
            routed.append(scope["path"])

    app = SimpleNamespace(
        router=Router(),
        middleware_stack=None,
        state=SimpleNamespace(),
    )
    slot = application_plugins.prepare_application_plugin_asgi_wrappers(
        app,
        admission=admission,
    )

    def wrapper(inner):
        async def dispatch(scope, receive, send) -> None:
            admission.path = scope["path"]
            if scope["path"] == "/claimed":
                lease = admission.try_acquire()
                assert lease is not None
                try:
                    return
                finally:
                    lease.release()
            await inner(scope, receive, send)

        return dispatch

    slot.install(wrapper)
    slot.seal()

    admission.path = "/claimed"
    await app.router(
        {"type": "http", "path": "/claimed"},
        None,
        None,
    )
    admission.path = "/passthrough"
    await app.router(
        {"type": "http", "path": "/passthrough"},
        None,
        None,
    )
    admission.path = "/health"
    await app.router(
        {"type": "http", "path": "/health"},
        None,
        None,
    )

    assert acquired == ["/claimed", "/passthrough"]
    assert released == ["/claimed", "/passthrough"]
    assert routed == ["/passthrough", "/health"]


@pytest.mark.asyncio
async def test_close_preserves_admitted_scope_but_rejects_the_next_scope() -> None:
    """Closure blocks new owners without cancelling an admitted lease."""
    # @spec ING-VEH-016, ING-VEH-019
    entered = asyncio.Event()
    release_handler = asyncio.Event()
    completed = False
    sent: list[dict[str, object]] = []

    class Router:
        async def __call__(self, scope, receive, send) -> None:
            del scope, receive, send
            nonlocal completed
            entered.set()
            await release_handler.wait()
            completed = True

    admission = application_plugins.create_application_admission()
    await admission.open_after_http_listener_bound()
    app = SimpleNamespace(
        router=Router(),
        middleware_stack=None,
        state=SimpleNamespace(),
    )
    slot = application_plugins.prepare_application_plugin_asgi_wrappers(
        app,
        admission=admission.view,
    )
    slot.seal()

    admitted = asyncio.create_task(
        app.router(
            {"type": "http", "method": "POST", "path": "/v1/work"},
            None,
            None,
        )
    )
    await entered.wait()
    admission.close_from_launcher_thread()
    release_handler.set()
    await admitted
    assert completed

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    await app.router(
        {"type": "http", "method": "POST", "path": "/v1/later"},
        None,
        send,
    )
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 503


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
    """plugin selection/config is explicit at the CLI boundary."""
    # @spec ING-VEH-003, ING-VEH-018
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
            "--application-plugin-startup-timeout",
            "12.5",
        ]
    )
    assert args.application_plugin == ["first", "second"]
    assert args.application_plugin_config == ["first=first.toml", "second="]
    assert args.ws_max_size == 2097152
    assert args.application_plugin_startup_timeout == 12.5

    defaults = parser.parse_args([])
    assert defaults.application_plugin_startup_timeout == 60.0
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--application-plugin-startup-timeout", "0"],
        )


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
    """Only an atomic lease can authorize work after the host transition."""
    # @spec ING-VEH-016, ING-VEH-019
    admission = application_plugins.create_application_admission()

    assert admission.state is ApplicationAdmissionState.CLOSED
    assert admission.view.try_acquire() is None
    assert not hasattr(admission.view, "is_open")
    await admission.open_after_http_listener_bound()
    assert admission.state is ApplicationAdmissionState.OPEN
    lease = admission.view.try_acquire()
    assert lease is not None
    lease.release()
    lease.release()
    await admission.close_before_owner_drain()
    assert admission.view.try_acquire() is None


def test_admission_close_linearizes_before_all_later_acquisitions() -> None:
    """A completed close makes every later owner registration fail."""
    # @spec ING-VEH-016, ING-VEH-019
    admission = application_plugins.create_application_admission()
    asyncio.run(admission.open_after_http_listener_bound())

    admitted_before_close = admission.view.try_acquire()
    assert admitted_before_close is not None
    admission.close_from_launcher_thread()

    later = [admission.view.try_acquire() for _ in range(64)]
    assert later == [None] * 64
    admitted_before_close.release()
    admitted_before_close.release()

    with pytest.raises(RuntimeError, match="reopen"):
        asyncio.run(admission.open_after_http_listener_bound())


@pytest.mark.asyncio
async def test_lifetime_scopes_config_signals_serving_then_drains_before_reverse_unwind() -> None:
    """host owns per-entry context and lifecycle signals."""
    # @spec ING-VEH-018, ING-VEH-022
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
        await lifetime.wait_ready()
        events.append("admission:open")
        await lifetime.mark_serving()
        events.append("serving")
        events.append("admission:closed")
        await lifetime.quiesce_and_drain()

    assert events == [
        "enter:first",
        "enter:second",
        "ready:first",
        "ready:second",
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
async def test_readiness_is_concurrent_after_all_participants_enter() -> None:
    """Readiness cannot serialize listeners that mutually await startup."""
    # @spec ING-VEH-018
    events: list[str] = []
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    host_context, _, _, _ = _host_context(FakeAdmission(events))

    class ConcurrentReadyParticipant(FakeParticipant):
        def __init__(
            self,
            events: list[str],
            name: str,
            mine: asyncio.Event,
            peer: asyncio.Event,
        ) -> None:
            super().__init__(events, name)
            self._mine = mine
            self._peer = peer

        async def wait_ready(self) -> None:
            self.events.append(f"ready-start:{self.name}")
            self._mine.set()
            await self._peer.wait()
            self.events.append(f"ready-done:{self.name}")

    selected = [
        SelectedApplicationPlugin(
            "first",
            lambda context: ConcurrentReadyParticipant(
                events,
                context.plugin_name,
                first_started,
                second_started,
            ),
            None,
        ),
        SelectedApplicationPlugin(
            "second",
            lambda context: ConcurrentReadyParticipant(
                events,
                context.plugin_name,
                second_started,
                first_started,
            ),
            None,
        ),
    ]

    async with application_plugins.application_plugin_lifetime(
        selected,
        host_context,
    ) as lifetime:
        await asyncio.wait_for(lifetime.wait_ready(), timeout=0.25)

    assert events[:2] == ["enter:first", "enter:second"]
    assert {"ready-start:first", "ready-start:second"} <= set(events)
    assert {"ready-done:first", "ready-done:second"} <= set(events)


@pytest.mark.asyncio
async def test_preawait_failure_is_latched_and_normal_return_is_failure() -> None:
    """Failure supervision cannot miss an early task death."""
    # @spec ING-VEH-017, ING-VEH-018
    events: list[str] = []
    host_context, _, _, _ = _host_context(FakeAdmission(events))

    class EarlyFailure(FakeParticipant):
        async def wait_failed(self):
            raise RuntimeError("listener failed")

    selected = [
        SelectedApplicationPlugin(
            "failed",
            lambda context: EarlyFailure(events, context.plugin_name),
            None,
        )
    ]
    async with application_plugins.application_plugin_lifetime(
        selected,
        host_context,
    ) as lifetime:
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="listener failed"):
            await lifetime.wait_failed()

    class UnexpectedReturn(FakeParticipant):
        async def wait_failed(self):
            return None

    selected = [
        SelectedApplicationPlugin(
            "returned",
            lambda context: UnexpectedReturn(events, context.plugin_name),
            None,
        )
    ]
    async with application_plugins.application_plugin_lifetime(
        selected,
        host_context,
    ) as lifetime:
        with pytest.raises(RuntimeError, match="returned"):
            await lifetime.wait_failed()


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
            with pytest.raises(BaseException) as error:
                await lifetime.quiesce_and_drain()
            assert len(getattr(error.value, "exceptions", ())) == 2
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
            async with application_plugins.application_plugin_lifetime(selected, host_context):
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
        async with application_plugins.application_plugin_lifetime(selected, host_context) as lifetime:
            with pytest.raises(BaseException) as error:
                await asyncio.wait_for(
                    lifetime.quiesce_and_drain(),
                    timeout=0.25,
                )
            errors = getattr(error.value, "exceptions", ())
            assert any(isinstance(item, TimeoutError) for item in errors)
            assert any(isinstance(item, RuntimeError) for item in errors)

        assert events[2:4] == ["drain:hanging", "drain:failing"]

    asyncio.run(exercise())


def test_each_participant_exit_is_timeboxed_and_all_are_attempted() -> None:
    """A hung context exit cannot strand an earlier participant."""
    # @spec ING-VEH-017, ING-VEH-022

    class FailingExit(FakeParticipant):
        async def __aexit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback
            self.events.append(f"exit:{self.name}")
            raise RuntimeError(self.name)

    class HangingExit(FakeParticipant):
        @property
        def shutdown_grace(self) -> float:
            return 0.01

        async def __aexit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback
            self.events.append(f"exit:{self.name}")
            await asyncio.Future()

    async def exercise() -> None:
        events: list[str] = []
        host_context, _, _, _ = _host_context(FakeAdmission(events))
        selected = [
            SelectedApplicationPlugin(
                "failing",
                lambda context: FailingExit(events, context.plugin_name),
                None,
            ),
            SelectedApplicationPlugin(
                "hanging",
                lambda context: HangingExit(events, context.plugin_name),
                None,
            ),
        ]

        async def run_lifetime() -> None:
            async with application_plugins.application_plugin_lifetime(
                selected,
                host_context,
            ):
                pass

        with pytest.raises(BaseException) as error:
            await asyncio.wait_for(run_lifetime(), timeout=0.25)

        errors = getattr(error.value, "exceptions", ())
        assert any(isinstance(item, TimeoutError) for item in errors)
        assert any(isinstance(item, RuntimeError) for item in errors)
        assert events[-2:] == ["exit:hanging", "exit:failing"]

    asyncio.run(exercise())


def test_supervision_module_remains_python_310_compatible() -> None:
    """The host protocol must not import Python-3.11-only runtime APIs."""
    # @spec ING-VEH-018
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported_names = {
        (node.module if isinstance(node, ast.ImportFrom) else None, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    attributes = {
        (node.value.id, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }

    assert ("builtins", "BaseExceptionGroup") not in imported_names
    assert ("asyncio", "timeout") not in attributes
    assert ("typing", "Self") not in imported_names


class FakeAdmission:
    """Atomic admission capability used by lifetime-only tests."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def try_acquire(self):
        return None


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

    async def wait_ready(self) -> None:
        self.events.append(f"ready:{self.name}")

    async def wait_failed(self):
        await asyncio.Future()

    async def quiesce_and_drain(self) -> None:
        self.events.append(f"drain:{self.name}")

    @property
    def shutdown_grace(self) -> float:
        return 1.0


@asynccontextmanager
async def _lifetime():
    yield
