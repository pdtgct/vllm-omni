# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract: every shutdown cause runs the same ordered barriers.

ING-VEH-017 names five shutdown causes — process signal, watchdog/engine
failure, participant failure, unexpected HTTP serve-task completion, and
programmatic stop — and requires each to close admission synchronously, then
drain participants, then stop HTTP, then exit participant contexts, and only
then shut the engine down. Each test here drives one cause through a real
Uvicorn server and asserts the exact journal order, including the real HTTP
shutdown observed via `uvicorn.Server.shutdown`. The signal cause runs in a
subprocess so SIGTERM never reaches the pytest worker.

Faults inside the chain must not skip later phases: a raising notification,
drain, HTTP shutdown, or context exit still ends in exactly one engine
shutdown, and the first cause remains the primary result. `wait_failed()`
is a failure in all three forms — exception, unexpected normal return, and
unexpected cancellation.

The watchdog test patches the module-level ``_WATCHDOG_INTERVAL_S`` seam;
the launcher must hoist its poll interval there (upstream keeps it local to
the loop body, which makes the failure path untestable in bounded time).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER_PATH = _ROOT / "vllm_omni/entrypoints/launcher.py"

_ORDERED_JOURNAL = [
    "shutdown-requested",
    "participant-drain",
    "http-shutdown",
    "participant-exit",
    "engine-shutdown",
]


def _load_launcher():
    assert _LAUNCHER_PATH.exists(), "Omni-owned launcher is not implemented"
    spec = importlib.util.spec_from_file_location(
        "_omni_launcher_shutdown_causes_under_test",
        _LAUNCHER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_journal(events: list[str], *, expect_http_shutdown: bool = True) -> None:
    expected = [name for name in _ORDERED_JOURNAL if expect_http_shutdown or name != "http-shutdown"]
    positions = [events.index(name) for name in expected]
    assert positions == sorted(positions), events
    assert events.count("shutdown-requested") == 1, events
    assert events.count("engine-shutdown") == 1, events


class _RecordingEngine:
    def __init__(self, events: list[str]) -> None:
        self.errored = False
        self.is_running = True
        self.vllm_config = SimpleNamespace(shutdown_timeout=0.01)
        self._events = events

    def shutdown(self, timeout=None) -> None:
        assert timeout == 0.01
        self._events.append("engine-shutdown")


class _RecordingHook:
    def __init__(
        self,
        events: list[str],
        bound: asyncio.Event,
        fault_site: str | None = None,
    ) -> None:
        self._events = events
        self._bound = bound
        self._fault_site = fault_site
        self.causes: list[BaseException | None] = []
        self.failure: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def _record(self, site: str) -> None:
        self._events.append(site)
        if self._fault_site == site:
            raise RuntimeError(f"fault at {site}")

    async def on_bound(self) -> None:
        self._events.append("bound")
        self._bound.set()

    def on_shutdown_requested(self, cause=None) -> None:
        self.causes.append(cause)
        self._record("shutdown-requested")

    async def before_http_shutdown(self) -> None:
        self._record("participant-drain")

    async def before_engine_shutdown(self) -> None:
        self._record("participant-exit")

    async def wait_failed(self) -> None:
        await self.failure


def _listening_socket() -> tuple[socket.socket, str, int]:
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(("127.0.0.1", 0))
    listen_socket.listen()
    listen_socket.setblocking(False)
    host, port = listen_socket.getsockname()
    return listen_socket, host, port


async def _run_cause(
    monkeypatch,
    trigger,
    *,
    launcher=None,
    fault_site: str | None = None,
    http_fault: bool = False,
    expect_http_shutdown: bool = True,
):
    """Boot the hooked launcher, fire one shutdown cause, return the journal.

    ``trigger(app, hook)`` runs once the listener is bound. Returns
    ``(events, hook, outcome)`` where outcome is the serve result or the
    raised exception. Always asserts the ING-VEH-017 barrier order and that
    admission closure and engine shutdown each happen exactly once.
    """
    if launcher is None:
        launcher = _load_launcher()
    events: list[str] = []
    bound = asyncio.Event()

    original_http_shutdown = uvicorn.Server.shutdown

    async def recording_http_shutdown(self, sockets=None):
        events.append("http-shutdown")
        if http_fault:
            raise RuntimeError("fault at http-shutdown")
        return await original_http_shutdown(self, sockets=sockets)

    monkeypatch.setattr(uvicorn.Server, "shutdown", recording_http_shutdown)

    app = FastAPI()
    app.state.engine_client = _RecordingEngine(events)
    hook = _RecordingHook(events, bound, fault_site=fault_site)
    listen_socket, host, port = _listening_socket()

    serve_task = asyncio.create_task(
        launcher.serve_http(
            app,
            listen_socket,
            lifecycle_hook=hook,
            host=host,
            port=port,
            log_level="error",
            lifespan="off",
        )
    )
    outcome: object = None
    try:
        await asyncio.wait_for(bound.wait(), timeout=2)
        await trigger(app, hook)
        try:
            shutdown = await asyncio.wait_for(serve_task, timeout=5)
            outcome = await shutdown
        except asyncio.TimeoutError:
            raise
        except BaseException as error:  # noqa: BLE001 - the contract result
            outcome = error
    finally:
        if not serve_task.done():
            serve_task.cancel()
            await asyncio.gather(serve_task, return_exceptions=True)
        listen_socket.close()

    _assert_journal(events, expect_http_shutdown=expect_http_shutdown)
    return events, hook, outcome


_SIGNAL_CHILD = """
import asyncio
import importlib.util
import json
import os
import signal
import socket
import sys
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI

launcher_path = sys.argv[1]
spec = importlib.util.spec_from_file_location(
    "_omni_launcher_signal_child", launcher_path
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

events = []
_original_http_shutdown = uvicorn.Server.shutdown


async def _recording_http_shutdown(self, sockets=None):
    events.append("http-shutdown")
    return await _original_http_shutdown(self, sockets=sockets)


uvicorn.Server.shutdown = _recording_http_shutdown


class Engine:
    errored = False
    is_running = True
    vllm_config = SimpleNamespace(shutdown_timeout=0.01)

    def shutdown(self, timeout=None):
        events.append("engine-shutdown")


class Hook:
    async def on_bound(self):
        events.append("bound")
        os.kill(os.getpid(), signal.SIGTERM)

    def on_shutdown_requested(self, cause=None):
        events.append("shutdown-requested")

    async def before_http_shutdown(self):
        events.append("participant-drain")

    async def before_engine_shutdown(self):
        events.append("participant-exit")

    async def wait_failed(self):
        await asyncio.get_running_loop().create_future()


async def main():
    app = FastAPI()
    app.state.engine_client = Engine()
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(("127.0.0.1", 0))
    listen_socket.listen()
    listen_socket.setblocking(False)
    host, port = listen_socket.getsockname()
    shutdown = await module.serve_http(
        app,
        listen_socket,
        lifecycle_hook=Hook(),
        host=host,
        port=port,
        log_level="error",
        lifespan="off",
    )
    await shutdown
    listen_socket.close()


asyncio.run(main())
print(json.dumps(events))
"""


def test_process_signal_orders_the_full_shutdown_barrier_chain(
    tmp_path,
) -> None:
    """SIGTERM closes admission first and stops the engine last.

    Runs in a subprocess so the signal never reaches the pytest worker.
    """
    # @spec ING-VEH-017, ING-VEH-022
    child = tmp_path / "signal_child.py"
    child.write_text(_SIGNAL_CHILD, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(child), str(_LAUNCHER_PATH)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    events = json.loads(result.stdout.strip().splitlines()[-1])
    _assert_journal(events)


@pytest.mark.asyncio
async def test_watchdog_engine_failure_orders_the_full_barrier_chain(
    monkeypatch,
) -> None:
    """An errored engine enters the same coordinated path, cause retained."""
    # @spec ING-VEH-017, ING-VEH-022
    launcher = _load_launcher()
    assert hasattr(launcher, "_WATCHDOG_INTERVAL_S"), (
        "the watchdog poll interval must be a module seam so the failure path is testable in bounded time"
    )
    monkeypatch.setattr(launcher, "_WATCHDOG_INTERVAL_S", 0.05)
    monkeypatch.setattr(sys.modules["vllm.envs"], "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", False)

    async def trigger(app, hook) -> None:
        del hook
        app.state.engine_client.errored = True
        app.state.engine_client.is_running = False

    events, _, outcome = await _run_cause(monkeypatch, trigger, launcher=launcher)
    assert isinstance(outcome, RuntimeError), events
    assert "watchdog" in str(outcome)


@pytest.mark.asyncio
async def test_participant_exception_orders_the_full_barrier_chain(
    monkeypatch,
) -> None:
    """A post-start participant failure drains, stops HTTP, then the engine."""

    # @spec ING-VEH-017, ING-VEH-022
    async def trigger(app, hook) -> None:
        del app
        hook.failure.set_exception(ValueError("participant exploded"))

    events, _, outcome = await _run_cause(monkeypatch, trigger)
    assert isinstance(outcome, ValueError), events
    assert "participant exploded" in str(outcome)


@pytest.mark.asyncio
async def test_participant_normal_return_is_a_failure(monkeypatch) -> None:
    """`wait_failed()` returning normally still closes and shuts down."""

    # @spec ING-VEH-017
    async def trigger(app, hook) -> None:
        del app
        hook.failure.set_result(None)

    events, _, outcome = await _run_cause(monkeypatch, trigger)
    assert isinstance(outcome, BaseException), events


@pytest.mark.asyncio
async def test_participant_cancellation_is_a_failure(monkeypatch) -> None:
    """`wait_failed()` cancelled from outside still closes and shuts down."""

    # @spec ING-VEH-017
    async def trigger(app, hook) -> None:
        del app
        hook.failure.cancel()

    events, _, outcome = await _run_cause(monkeypatch, trigger)
    assert isinstance(outcome, BaseException), events


@pytest.mark.asyncio
async def test_unexpected_http_exit_still_drains_before_engine_shutdown(
    monkeypatch,
) -> None:
    """An HTTP server that dies on its own is a failure, not a bypass."""
    # @spec ING-VEH-017, ING-VEH-022
    die = asyncio.Event()
    original_serve = uvicorn.Server.serve

    async def dying_serve(self, sockets=None):
        inner = asyncio.create_task(original_serve(self, sockets=sockets))
        die_wait = asyncio.create_task(die.wait())
        done, _ = await asyncio.wait({inner, die_wait}, return_when=asyncio.FIRST_COMPLETED)
        if die_wait in done:
            inner.cancel()
            await asyncio.gather(inner, return_exceptions=True)
            return None
        die_wait.cancel()
        return inner.result()

    monkeypatch.setattr(uvicorn.Server, "serve", dying_serve)

    async def trigger(app, hook) -> None:
        del app, hook
        die.set()

    events, _, outcome = await _run_cause(monkeypatch, trigger, expect_http_shutdown=False)
    assert isinstance(outcome, RuntimeError), events
    assert "unexpectedly" in str(outcome)


@pytest.mark.asyncio
async def test_programmatic_stop_orders_the_full_barrier_chain(
    monkeypatch,
) -> None:
    """`should_exit` set by the host enters the same coordinated path."""

    # @spec ING-VEH-017, ING-VEH-022
    async def trigger(app, hook) -> None:
        del hook
        app.state.server.should_exit = True

    events, _, outcome = await _run_cause(monkeypatch, trigger)
    assert not isinstance(outcome, BaseException), events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_site",
    [
        "shutdown-requested",
        "participant-drain",
        "http-shutdown",
        "participant-exit",
    ],
)
async def test_fault_inside_the_chain_cannot_skip_later_phases(monkeypatch, fault_site) -> None:
    """A raising phase surfaces as the result; the engine still stops once."""

    # @spec ING-VEH-017, ING-VEH-022
    async def trigger(app, hook) -> None:
        del hook
        app.state.server.should_exit = True

    events, _, outcome = await _run_cause(
        monkeypatch,
        trigger,
        fault_site=None if fault_site == "http-shutdown" else fault_site,
        http_fault=fault_site == "http-shutdown",
    )
    assert isinstance(outcome, BaseException), events
    assert f"fault at {fault_site}" in str(outcome), events


@pytest.mark.asyncio
async def test_first_cause_stays_primary_across_a_two_cause_race(
    monkeypatch,
) -> None:
    """The first cause is the notified and raised one; close happens once."""

    # @spec ING-VEH-017
    boom = ValueError("first cause")

    async def trigger(app, hook) -> None:
        hook.failure.set_exception(boom)
        await asyncio.sleep(0)
        app.state.server.should_exit = True

    events, hook, outcome = await _run_cause(monkeypatch, trigger)
    assert outcome is boom, events
    assert hook.causes == [boom]


@pytest.mark.asyncio
async def test_host_request_closes_admission_before_publishing_the_flag(
    monkeypatch,
) -> None:
    """The conforming programmatic stop is synchronous, not a polled flag."""

    # @spec ING-VEH-017
    async def trigger(app, hook) -> None:
        request_shutdown = app.state.request_application_shutdown
        assert hook.causes == [], "admission closed before the host asked"

        request_shutdown(None)

        # Both assertions are about the same instant: the notification has
        # already happened and the flag the watcher polls has not been set,
        # so admission closure cannot trail the stop by a poll interval.
        assert hook.causes == [None]
        assert app.state.server.should_exit is False

    events, hook, outcome = await _run_cause(monkeypatch, trigger)
    assert not isinstance(outcome, BaseException), events
    assert events.count("shutdown-requested") == 1, events


@pytest.mark.asyncio
async def test_failure_latched_while_binding_never_opens_admission(
    monkeypatch,
) -> None:
    """A failure that beats the listener must not be followed by serving."""
    # @spec ING-VEH-016, ING-VEH-017
    launcher = _load_launcher()
    events: list[str] = []

    original_http_shutdown = uvicorn.Server.shutdown

    async def recording_http_shutdown(self, sockets=None):
        events.append("http-shutdown")
        return await original_http_shutdown(self, sockets=sockets)

    monkeypatch.setattr(uvicorn.Server, "shutdown", recording_http_shutdown)

    app = FastAPI()
    app.state.engine_client = _RecordingEngine(events)
    hook = _RecordingHook(events, asyncio.Event())
    latched = ValueError("failed while binding")
    hook.failure.set_exception(latched)
    listen_socket, host, port = _listening_socket()

    outcome: object = None
    try:
        try:
            shutdown = await asyncio.wait_for(
                launcher.serve_http(
                    app,
                    listen_socket,
                    lifecycle_hook=hook,
                    host=host,
                    port=port,
                    log_level="error",
                    lifespan="off",
                ),
                timeout=5,
            )
            outcome = await shutdown
        except asyncio.TimeoutError:
            raise
        except BaseException as error:  # noqa: BLE001 - the contract result
            outcome = error
    finally:
        listen_socket.close()

    assert outcome is latched, events
    assert "bound" not in events, events
    assert hook.causes == [latched]
    _assert_journal(events)


@pytest.mark.asyncio
async def test_engine_shutdown_is_awaited_to_completion_across_cancellation(
    monkeypatch,
) -> None:
    """Cancelling the await cannot orphan the executor thread's stop."""
    # @spec ING-VEH-017, ING-VEH-022
    launcher = _load_launcher()
    events: list[str] = []
    started = threading.Event()
    release = threading.Event()

    class BlockingEngine(_RecordingEngine):
        def shutdown(self, timeout=None) -> None:
            self._events.append("engine-shutdown")
            started.set()
            assert release.wait(timeout=10), "engine stop was never released"
            self._events.append("engine-shutdown-returned")

    app = FastAPI()
    app.state.engine_client = BlockingEngine(events)
    bound = asyncio.Event()
    hook = _RecordingHook(events, bound)
    listen_socket, host, port = _listening_socket()

    serve_task = asyncio.create_task(
        launcher.serve_http(
            app,
            listen_socket,
            lifecycle_hook=hook,
            host=host,
            port=port,
            log_level="error",
            lifespan="off",
        )
    )
    try:
        await asyncio.wait_for(bound.wait(), timeout=2)
        app.state.request_application_shutdown(None)
        for _ in range(500):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), events

        # Cancel while the stop is in flight on the executor thread.
        serve_task.cancel()
        for _ in range(20):
            await asyncio.sleep(0.01)
        # The decisive assertion: serve has not finished, because the engine
        # stop it started has not finished.  A launcher that treated the
        # cancelled await as the phase being over would already be done here,
        # leaving the executor thread running behind it.
        assert not serve_task.done(), events

        release.set()
        await asyncio.gather(serve_task, return_exceptions=True)
    finally:
        release.set()
        listen_socket.close()

    assert events.count("engine-shutdown") == 1, events
    assert "engine-shutdown-returned" in events, events


@pytest.mark.asyncio
async def test_a_late_supervisor_failure_is_retained_as_the_result(
    monkeypatch,
) -> None:
    """A watchdog that fires mid-drain is evidence, not a discarded task."""
    # @spec ING-VEH-017, ING-VEH-022
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "_WATCHDOG_INTERVAL_S", 0.02)
    monkeypatch.setattr(sys.modules["vllm.envs"], "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", False)

    class DyingDuringDrain(_RecordingHook):
        def __init__(self, events, bound, engine) -> None:
            super().__init__(events, bound)
            self._engine = engine

        async def before_http_shutdown(self) -> None:
            self._record("participant-drain")
            self._engine.errored = True
            self._engine.is_running = False
            await asyncio.sleep(0.2)

    events: list[str] = []
    engine = _RecordingEngine(events)

    async def trigger(app, hook) -> None:
        del hook
        app.state.request_application_shutdown(None)

    original_http_shutdown = uvicorn.Server.shutdown

    async def recording_http_shutdown(self, sockets=None):
        events.append("http-shutdown")
        return await original_http_shutdown(self, sockets=sockets)

    monkeypatch.setattr(uvicorn.Server, "shutdown", recording_http_shutdown)

    app = FastAPI()
    app.state.engine_client = engine
    bound = asyncio.Event()
    hook = DyingDuringDrain(events, bound, engine)
    listen_socket, host, port = _listening_socket()

    serve_task = asyncio.create_task(
        launcher.serve_http(
            app,
            listen_socket,
            lifecycle_hook=hook,
            host=host,
            port=port,
            log_level="error",
            lifespan="off",
        )
    )
    outcome: object = None
    try:
        await asyncio.wait_for(bound.wait(), timeout=2)
        await trigger(app, hook)
        try:
            shutdown = await asyncio.wait_for(serve_task, timeout=5)
            outcome = await shutdown
        except asyncio.TimeoutError:
            raise
        except BaseException as error:  # noqa: BLE001 - the contract result
            outcome = error
    finally:
        if not serve_task.done():
            serve_task.cancel()
            await asyncio.gather(serve_task, return_exceptions=True)
        listen_socket.close()

    # No cause was named at the linearization point, so the only failure the
    # run produced is the one that arrived after it — and it survived.
    assert hook.causes == [None], events
    assert isinstance(outcome, RuntimeError), outcome
    assert "watchdog" in str(outcome), outcome
    _assert_journal(events)


def test_completed_causes_orders_every_finished_supervisor() -> None:
    """Concurrent causes are all named, with a fixed primary."""
    # @spec ING-VEH-017
    launcher = _load_launcher()

    async def exercise() -> None:
        participant = ValueError("participant")
        watchdog = RuntimeError("engine client failed its HTTP watchdog")
        http = RuntimeError("serve raised")

        async def raise_it(error):
            raise error

        failure_task = asyncio.ensure_future(raise_it(participant))
        watchdog_task = asyncio.ensure_future(raise_it(watchdog))
        server_task = asyncio.ensure_future(raise_it(http))
        await asyncio.gather(
            failure_task,
            watchdog_task,
            server_task,
            return_exceptions=True,
        )
        server = SimpleNamespace(should_exit=False)

        causes = launcher._completed_causes(
            server,
            failure_task=failure_task,
            watchdog_task=watchdog_task,
            server_task=server_task,
        )
        assert causes == [participant, watchdog, http]

        # Once the stop is under way, a finished serve task is the shutdown
        # working rather than an unexpected exit.
        server.should_exit = True
        assert launcher._completed_causes(
            server,
            failure_task=failure_task,
            watchdog_task=watchdog_task,
            server_task=server_task,
        ) == [participant, watchdog]

    asyncio.run(exercise())
