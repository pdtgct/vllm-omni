"""Provider selection: how a dialect adapter binds the session core.

The binding is a value, never structure (ING-CORE-002): ``in-process``
binds the serving core in the same process (the β tier, also the
GPU-free test binding); ``remote`` speaks the vLLM ``/v1/realtime``
dialect to any vllm-omni server carrying the model. The remote gate is
a fast-fail counter on the same watermark value — its count is a proxy
that cannot see engine-side eviction, so it never queues (ledger A11,
ING-ADM-003).
"""

from collections.abc import Awaitable, Callable, Mapping
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

    The local counter is only a conservative fast-fail projection
    (ING-ADM-003): ``ADMITTED`` is reported ONLY on the upstream
    provider's authoritative acknowledgement, awaited through the
    injected ``await_admission`` seam. Every negative or failure path
    releases the projected count and disposes the connection — the
    count can never leak. ``connect``/``await_admission``/``close``
    are injected so the GPU-free tier can observe that no upstream
    connection is opened for a rejected session and that no path
    leaks the count.
    """

    kind = "remote"

    def __init__(
        self,
        url: str,
        gate: RemoteGate,
        connect: Callable[[str], Any],
        await_admission: Callable[[Any], Awaitable[AdmissionOutcome]],
        close: Callable[[Any], None] | None = None,
    ) -> None:
        """Bind the upstream endpoint behind the fast-fail gate."""
        self.url = url
        self._gate = gate
        self._connect = connect
        self._await_admission = await_admission
        self._close = close
        self.connections: list[Any] = []

    async def open_session(self) -> AdmissionOutcome:
        """Project first; ``ADMITTED`` only on the upstream ack.

        ``BUSY`` at the local bound opens nothing. Below it, the
        upstream request opens and the authoritative admitted/busy
        answer is awaited (ING-ADM-003); an unavailable answer fails
        closed — the exception propagates with the projected count
        released, never left counted.
        """
        if self._gate.request() is AdmissionOutcome.BUSY:
            return AdmissionOutcome.BUSY
        connection: Any = None
        try:
            connection = self._connect(self.url)
            outcome = await self._await_admission(connection)
        except BaseException:
            self._dispose(connection)
            self._gate.release()
            raise
        if outcome is not AdmissionOutcome.ADMITTED:
            self._dispose(connection)
            self._gate.release()
            return AdmissionOutcome.BUSY
        self.connections.append(connection)
        return AdmissionOutcome.ADMITTED

    def close_session(self, connection: Any) -> None:
        """Release one admitted session: dispose and free the count."""
        self.connections.remove(connection)
        self._dispose(connection)
        self._gate.release()

    def _dispose(self, connection: Any) -> None:
        if connection is not None and self._close is not None:
            self._close(connection)


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


async def _pod_side_await_admission(connection: Any) -> AdmissionOutcome:
    """The default acknowledgement seam: pod-side, like the transport."""
    raise RuntimeError(
        "no admission-acknowledgement seam is bound in this environment; "
        "construct RemoteProvider with an explicit await_admission"
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
            await_admission=_pod_side_await_admission,
        )
    raise ValueError(
        f"unknown ingress provider {kind!r}: expected 'in-process' or 'remote'"
    )
