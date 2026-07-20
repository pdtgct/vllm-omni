"""Provider selection and the remote fast-fail gate.

ING-CORE-002 (binding is a value), ING-ADM-003 (remote gate: same W,
fast-fail, never queues, no connection for a rejected session).
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

VALUES: dict[str, Any] = {
    "provider": "in-process",
    "watermark": 2,
    "realtime_url": "ws://upstream:8000/v1/realtime",
}


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
def test_no_upstream_connection_for_a_rejected_session() -> None:
    calls: list[str] = []

    def connect(url: str) -> str:
        calls.append(url)
        return "connection"

    gate = RemoteGate(watermark=1)
    provider = RemoteProvider(
        url="ws://upstream:8000/v1/realtime", gate=gate, connect=connect
    )
    assert provider.open_session() is AdmissionOutcome.ADMITTED
    assert calls == ["ws://upstream:8000/v1/realtime"]
    assert provider.open_session() is AdmissionOutcome.BUSY
    # The rejection opened nothing upstream.
    assert calls == ["ws://upstream:8000/v1/realtime"]


# @spec ING-ADM-003
def test_remote_gate_enforces_the_same_watermark_value() -> None:
    # Both gates are built from the one values mapping: the remote
    # count is a proxy, but the bound is the same W (never a separate
    # remote-only tunable).
    values = {**VALUES, "provider": "remote", "watermark": 1}
    gate = RemoteGate(watermark=int(values["watermark"]))
    assert gate.request() is AdmissionOutcome.ADMITTED
    assert gate.request() is AdmissionOutcome.BUSY
