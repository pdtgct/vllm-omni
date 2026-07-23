"""Provider selection and the remote fast-fail gate.

ING-CORE-002 (binding is a value), ING-ADM-003 (remote gate: same W,
fast-fail, never queues, no connection for a rejected session, and
``ADMITTED`` only on the upstream's awaited acknowledgement — every
negative or failure path releases the projected count and disposes
the connection).
"""

from typing import Any

import pytest

from nemotron_asr_ingress.events import AdmissionOutcome
from nemotron_asr_ingress.provider import (
    InProcessProvider,
    RemoteGate,
    RemoteProvider,
    select_provider,
)

URL = "ws://upstream:8000/v1/realtime"

VALUES: dict[str, Any] = {
    "provider": "in-process",
    "watermark": 2,
    "realtime_url": URL,
}


class FakeUpstream:
    """Recorded connect/ack/close seams with scripted admission acks."""

    def __init__(self, acks: list[AdmissionOutcome] | None = None) -> None:
        self.acks: list[AdmissionOutcome] = list(acks or [])
        self.connects: list[str] = []
        self.acked: list[str] = []
        self.closed: list[str] = []

    def connect(self, url: str) -> str:
        self.connects.append(url)
        return f"connection-{len(self.connects)}"

    async def await_admission(self, connection: str) -> AdmissionOutcome:
        self.acked.append(connection)
        return self.acks.pop(0)

    def close(self, connection: str) -> None:
        self.closed.append(connection)


def make_remote(
    upstream: FakeUpstream, watermark: int = 1
) -> tuple[RemoteProvider, RemoteGate]:
    gate = RemoteGate(watermark=watermark)
    provider = RemoteProvider(
        url=URL,
        gate=gate,
        connect=upstream.connect,
        await_admission=upstream.await_admission,
        close=upstream.close,
    )
    return provider, gate


# @spec ING-CORE-002
def test_in_process_provider_selected_by_value() -> None:
    provider = select_provider(VALUES)
    assert isinstance(provider, InProcessProvider)
    assert provider.kind == "in-process"


# @spec ING-CORE-002
def test_remote_provider_selected_by_value() -> None:
    provider = select_provider({**VALUES, "provider": "remote"})
    assert isinstance(provider, RemoteProvider)
    assert provider.kind == "remote"


# @spec ING-CORE-002
def test_unknown_provider_value_is_never_defaulted() -> None:
    with pytest.raises(ValueError, match="provider"):
        select_provider({**VALUES, "provider": "sidecar"})


# @spec ING-ADM-003
def test_remote_gate_fast_fails_at_the_watermark() -> None:
    gate = RemoteGate(watermark=2)
    assert gate.request() is AdmissionOutcome.ADMITTED
    assert gate.request() is AdmissionOutcome.ADMITTED
    assert gate.request() is AdmissionOutcome.BUSY
    gate.release()
    assert gate.request() is AdmissionOutcome.ADMITTED


# @spec ING-ADM-003, ING-CORE-003
def test_remote_gate_never_queues() -> None:
    gate = RemoteGate(watermark=1)
    outcomes = [gate.request() for _ in range(5)]
    assert outcomes[0] is AdmissionOutcome.ADMITTED
    assert all(o is AdmissionOutcome.BUSY for o in outcomes[1:])
    assert AdmissionOutcome.QUEUED not in outcomes


# @spec ING-ADM-003
async def test_no_upstream_connection_for_a_rejected_session() -> None:
    upstream = FakeUpstream(acks=[AdmissionOutcome.ADMITTED])
    provider, _ = make_remote(upstream, watermark=1)
    assert await provider.open_session() is AdmissionOutcome.ADMITTED
    assert upstream.connects == [URL]
    assert await provider.open_session() is AdmissionOutcome.BUSY
    # The rejection opened nothing upstream and awaited no ack.
    assert upstream.connects == [URL]
    assert upstream.acked == ["connection-1"]


# @spec ING-ADM-003
async def test_admitted_is_reported_only_after_the_upstream_ack() -> None:
    order: list[str] = []
    gate = RemoteGate(watermark=1)

    def connect(url: str) -> str:
        order.append("connect")
        return "connection"

    async def await_admission(connection: Any) -> AdmissionOutcome:
        order.append("ack")
        # While the ack is being awaited, nothing is registered yet:
        # the local count is only a projection, never an admission.
        assert provider.connections == []
        return AdmissionOutcome.ADMITTED

    provider = RemoteProvider(
        url=URL, gate=gate, connect=connect, await_admission=await_admission
    )
    assert await provider.open_session() is AdmissionOutcome.ADMITTED
    assert order == ["connect", "ack"]
    assert provider.connections == ["connection"]


# @spec ING-ADM-003
async def test_busy_ack_releases_the_count_and_disposes() -> None:
    upstream = FakeUpstream(
        acks=[AdmissionOutcome.BUSY, AdmissionOutcome.ADMITTED]
    )
    provider, gate = make_remote(upstream, watermark=1)
    assert await provider.open_session() is AdmissionOutcome.BUSY
    assert upstream.closed == ["connection-1"]
    assert provider.connections == []
    assert gate.active == 0
    # The released count admits a follow-up open below the bound.
    assert await provider.open_session() is AdmissionOutcome.ADMITTED
    assert provider.connections == ["connection-2"]


# @spec ING-ADM-003
async def test_connect_failure_releases_the_count_and_reraises() -> None:
    gate = RemoteGate(watermark=1)
    closed: list[Any] = []

    def connect(url: str) -> str:
        raise OSError("upstream unreachable")

    async def never_acks(connection: Any) -> AdmissionOutcome:
        raise AssertionError("ack awaited without a connection")

    provider = RemoteProvider(
        url=URL,
        gate=gate,
        connect=connect,
        await_admission=never_acks,
        close=closed.append,
    )
    with pytest.raises(OSError, match="unreachable"):
        await provider.open_session()
    assert gate.active == 0
    # No connection ever existed, so nothing was disposed.
    assert closed == []


# @spec ING-ADM-003
async def test_ack_failure_disposes_releases_and_reraises() -> None:
    gate = RemoteGate(watermark=1)
    closed: list[Any] = []

    def connect(url: str) -> str:
        return "connection"

    async def failing_ack(connection: Any) -> AdmissionOutcome:
        raise OSError("upstream went away mid-ack")

    provider = RemoteProvider(
        url=URL,
        gate=gate,
        connect=connect,
        await_admission=failing_ack,
        close=closed.append,
    )
    with pytest.raises(OSError, match="mid-ack"):
        await provider.open_session()
    assert closed == ["connection"]
    assert provider.connections == []
    assert gate.active == 0


# @spec ING-ADM-003
async def test_close_session_releases_the_count_and_disposes() -> None:
    upstream = FakeUpstream(
        acks=[AdmissionOutcome.ADMITTED, AdmissionOutcome.ADMITTED]
    )
    provider, gate = make_remote(upstream, watermark=1)
    assert await provider.open_session() is AdmissionOutcome.ADMITTED
    (connection,) = provider.connections
    provider.close_session(connection)
    assert provider.connections == []
    assert upstream.closed == [connection]
    assert gate.active == 0
    # The freed slot admits again below the bound.
    assert await provider.open_session() is AdmissionOutcome.ADMITTED


# @spec ING-ADM-003
async def test_select_provider_default_seams_fail_closed() -> None:
    provider = select_provider(
        {**VALUES, "provider": "remote", "watermark": 1}
    )
    assert isinstance(provider, RemoteProvider)
    with pytest.raises(RuntimeError, match="connect factory"):
        await provider.open_session()
    # Fail-closed released the projected count: the retry reaches the
    # transport seam again instead of fast-failing BUSY at the bound.
    with pytest.raises(RuntimeError, match="connect factory"):
        await provider.open_session()


# @spec ING-ADM-003
def test_remote_gate_enforces_the_same_watermark_value() -> None:
    # Both gates are built from the one values mapping: the remote
    # count is a proxy, but the bound is the same W (never a separate
    # remote-only tunable).
    values = {**VALUES, "provider": "remote", "watermark": 1}
    gate = RemoteGate(watermark=int(values["watermark"]))
    assert gate.request() is AdmissionOutcome.ADMITTED
    assert gate.request() is AdmissionOutcome.BUSY
