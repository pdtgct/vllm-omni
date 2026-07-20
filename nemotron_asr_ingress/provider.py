"""Provider selection: how a dialect adapter binds the session core.

The binding is a value, never structure (ING-CORE-002): ``in-process``
binds the serving core in the same process (the β tier, also the
GPU-free test binding); ``remote`` speaks the vLLM ``/v1/realtime``
dialect to any vllm-omni server carrying the model. The remote gate is
a fast-fail counter on the same watermark value — its count is a proxy
that cannot see engine-side eviction, so it never queues (ledger A11,
ING-ADM-003).
"""

from collections.abc import Callable, Mapping
from typing import Any

from nemotron_asr_ingress.events import AdmissionOutcome


# @spec ING-ADM-003
class RemoteGate:
    """Fast-fail watermark counter for the remote provider.

    Enforces the same W value as the in-process gate, answers ``BUSY``
    the moment its live count reaches W, and has no queue by
    construction.
    """

    def __init__(self, watermark: int) -> None:
        """Set W (the shared ENV value)."""
        if watermark < 1:
            raise ValueError(f"watermark must be >= 1, got {watermark}")
        self._watermark = watermark
        self._active = 0

    @property
    def active(self) -> int:
        """Sessions this adapter currently has open upstream."""
        return self._active

    def request(self) -> AdmissionOutcome:
        """``ADMITTED`` below W; ``BUSY`` at W — immediately, no queue."""
        if self._active >= self._watermark:
            return AdmissionOutcome.BUSY
        self._active += 1
        return AdmissionOutcome.ADMITTED

    def release(self) -> None:
        """Free one counted session.

        Raises:
            RuntimeError: If nothing is counted — a release without an
                admission is adapter bookkeeping gone wrong, never
                ignored.
        """
        if self._active == 0:
            raise RuntimeError("release() without a matching admission")
        self._active -= 1


class InProcessProvider:
    """Session-core binding to the serving core in the same process."""

    kind = "in-process"

    def __init__(self, values: Mapping[str, Any]) -> None:
        """Bind with the shared ingress values."""
        self.values: dict[str, Any] = dict(values)


class RemoteProvider:
    """Session-core binding over the vLLM realtime dialect.

    ``connect`` is the transport factory (injected so the GPU-free
    tier can observe that no upstream connection is ever opened for a
    rejected session — ING-ADM-003).
    """

    kind = "remote"

    def __init__(
        self,
        url: str,
        gate: RemoteGate,
        connect: Callable[[str], Any],
    ) -> None:
        """Bind the upstream endpoint behind the fast-fail gate."""
        self.url = url
        self._gate = gate
        self._connect = connect
        self.connections: list[Any] = []

    def open_session(self) -> AdmissionOutcome:
        """Gate first: ``BUSY`` opens nothing; ``ADMITTED`` connects."""
        outcome = self._gate.request()
        if outcome is AdmissionOutcome.ADMITTED:
            self.connections.append(self._connect(self.url))
        return outcome


def _pod_side_connect(url: str) -> Any:
    """The default transport factory: the WebSocket shell is pod-side.

    The sans-IO client core (:mod:`nemotron_asr_ingress.client`)
    carries the dialect handling; the socket itself is injected where
    a transport exists.
    """
    raise RuntimeError(
        "no WebSocket transport is bound in this environment; "
        "construct RemoteProvider with an explicit connect factory"
    )


# @spec ING-CORE-002
def select_provider(
    values: Mapping[str, Any],
) -> InProcessProvider | RemoteProvider:
    """Build the provider the ``ingress.provider`` value names.

    Raises:
        ValueError: For a provider value that is neither
            ``in-process`` nor ``remote`` — never a silent default.
    """
    kind = values.get("provider")
    if kind == "in-process":
        return InProcessProvider(values)
    if kind == "remote":
        return RemoteProvider(
            url=str(values["realtime_url"]),
            gate=RemoteGate(watermark=int(values["watermark"])),
            connect=_pod_side_connect,
        )
    raise ValueError(
        f"unknown ingress provider {kind!r}: expected 'in-process' or 'remote'"
    )
