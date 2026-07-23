"""Provider selection: how a dialect adapter binds the session core.

The binding is a value, never structure (ING-CORE-002): ``in-process``
binds the serving core in the same process (the β tier, also the
GPU-free test binding); ``remote`` speaks the vLLM ``/v1/realtime``
dialect to any vllm-omni server carrying the model. The remote gate is
a fast-fail counter on the same watermark value — its count is a proxy
that cannot see engine-side eviction, so it never queues (ledger A11,
ING-ADM-003).
"""

import inspect
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


class RemoteLease:
    """One admitted upstream session, owned by whoever holds it.

    Admission hands the connection to exactly one owner instead of
    parking it on a shared registry: the lease is the only handle,
    and ``close`` is the only release path — idempotent, so a
    teardown raced by an error path frees the gate count exactly
    once (ING-ADM-003).
    """

    def __init__(
        self,
        connection: Any,
        dispose: Callable[[Any], Awaitable[None]],
        gate: RemoteGate,
    ) -> None:
        """Take ownership of one admitted, counted connection."""
        self._connection = connection
        self._dispose = dispose
        self._gate = gate
        self._closed = False

    @property
    def connection(self) -> Any:
        """The owned upstream connection."""
        return self._connection

    @property
    def closed(self) -> bool:
        """Whether the lease has been released."""
        return self._closed

    async def close(self) -> None:
        """Dispose the connection and free the count, exactly once.

        A repeated close is a no-op. The count is released even if
        disposal raises — a failed close must not strand a slot.
        """
        if self._closed:
            return
        self._closed = True
        try:
            await self._dispose(self._connection)
        finally:
            self._gate.release()


class RemoteProvider:
    """Session-core binding over the vLLM realtime dialect.

    The local counter is only a conservative fast-fail projection
    (ING-ADM-003): admission is reported ONLY on the upstream
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
        close: Callable[[Any], Any] | None = None,
    ) -> None:
        """Bind the upstream endpoint behind the fast-fail gate.

        ``close`` may be sync or return an awaitable (a real
        WebSocket closure normally awaits); both are honored.
        """
        self.url = url
        self._gate = gate
        self._connect = connect
        self._await_admission = await_admission
        self._close = close

    async def open_session(self) -> AdmissionOutcome | RemoteLease:
        """Project first; a lease only on the upstream ack.

        Returns the :class:`RemoteLease` owning the connection when
        the upstream acknowledges admission, and a bare
        ``AdmissionOutcome`` otherwise — the union keeps the two
        answers structurally distinct: a negative outcome carries
        nothing to close, so no half-alive lease can exist, and the
        admitted path has exactly one owner for the connection
        (no shared registry to desynchronize).

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
            if connection is not None:
                await self._dispose(connection)
            self._gate.release()
            raise
        if outcome is not AdmissionOutcome.ADMITTED:
            await self._dispose(connection)
            self._gate.release()
            return AdmissionOutcome.BUSY
        return RemoteLease(connection, self._dispose, self._gate)

    async def _dispose(self, connection: Any) -> None:
        if self._close is not None:
            result = self._close(connection)
            if inspect.isawaitable(result):
                await result


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
