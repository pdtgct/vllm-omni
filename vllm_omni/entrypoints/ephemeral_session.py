# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The canonical ephemeral-transcription orchestrator (PORT-EPH-001/002).

The ONE serving-layer orchestrator every single-shot transcription
surface delegates to: the OpenAI ``/v1/audio/transcriptions`` adapter,
the Riva ``Recognize`` codec, and the NIM HTTP shim transcription
handler each hand their decoded clip here and shape the response, but
none of them owns admission, the feed -> flush -> finish/abort
lifecycle, final-tail draining, or lease release. That lifecycle lives
here exactly once (PORT-EPH-001), lifted from the RFC-2 ``Recognize``
body minus the wire.

Two layers, one published seam. This module owns the *serving-layer*
concerns -- provider leasing (behind the factory), deadline
composition, the terminal-call/release invariant. Everything
model-specific -- per-stream state, prompt/locale selection, geometry,
carrier minting, and above all cadence segmentation (PORT owns it,
ING-FE-006) -- lives behind the :class:`SessionFactory` /
:class:`SessionLease` protocols. The orchestrator consumes those
protocols and nothing else: it holds NO cadence or chunk arithmetic,
never converts milliseconds to samples, and never consumes RFC-2's
``IngressValues`` (PORT-EPH-004). The only sample accounting it does is
a pre-submit slice: it hands each feed call at most
``submit_bound_samples`` samples -- a bound the serving layer computes
and passes in, so the model's cadence grid stays entirely on the model
side of the seam (ING-FE-005/ING-FE-006).

Engine-free (stdlib + protocol shapes only -- no ``torch``, no
``vllm``, no numpy at runtime), so it loads and runs on the macOS
loader path alongside the model package's ``session.py``.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    import numpy.typing as npt

    #: One contiguous block of decoded float32 PCM. Type-parity with the
    #: seam's ``feed`` argument; the orchestrator only ever slices it.
    FloatSamples = npt.NDArray[np.float32]


@runtime_checkable
class SessionLease(Protocol):
    """The orchestrator's handle on one admitted session.

    Structural by intent (PORT-EPH-004): an implementor -- the model
    package's engine-bound transcriber, RFC-2's ``EngineTranscriber``,
    or a test fake -- satisfies this by *shape*, never by importing or
    subclassing it. The async lifecycle surface is exactly the RFC-2
    ``Transcriber`` protocol (``feed``/``flush``/``update_locale``/
    ``abort``/``finish``) plus lease :meth:`release`.

    The terminal invariant the orchestrator upholds on every path:
    exactly ONE terminal engine-cleanup call -- :meth:`finish` on the
    success path, :meth:`abort` on the error/cancel path, never both --
    followed by exactly one :meth:`release` that frees the provider
    slot AFTER that cleanup, never before. Release is a distinct call,
    not folded into finish/abort, precisely so the orchestrator can
    guarantee the slot is freed on every path in a single ``finally``
    even if the terminal cleanup call itself raises, while preserving
    the cleanup-then-release ordering.
    """

    async def feed(self, samples: FloatSamples) -> list[str]:
        """Advance one accepted piece; one hypothesis per CHUNK done."""
        ...

    async def flush(self) -> str:
        """Drain the final-tail (zero-sample included); final transcript."""
        ...

    async def update_locale(self, locale: str) -> None:
        """Forward a validated mid-session locale change."""
        ...

    async def abort(self) -> None:
        """Free engine state immediately on a non-normal end."""
        ...

    async def finish(self) -> None:
        """Free engine state on the normal end, flushed or not."""
        ...

    async def release(self) -> None:
        """Free the provider slot; idempotent, called after cleanup."""
        ...


@runtime_checkable
class SessionFactory(Protocol):
    """Mints one admitted :class:`SessionLease` for model-typed params.

    Admission and provider leasing live BEHIND this seam (PORT-EPH-004):
    the orchestrator never imports a concrete gate, never sees a
    watermark, and never counts slots. ``cadence`` and ``locale`` are
    opaque model-typed strings -- the factory (implemented by the model
    package) validates them against the served checkpoint's own
    authority; the orchestrator neither interprets nor defaults them.

    Admission-busy is surfaced by raising :class:`AdmissionBusyError`, not by
    returning ``None``: ``open`` keeps a total ``SessionLease`` return
    so no caller threads an Optional or risks proceeding on a ``None``;
    the busy condition is exceptional (the shared pool is full) and each
    transport already translates errors to its own busy status at its
    boundary; and because a raised ``AdmissionBusyError`` means no lease was
    ever minted, the orchestrator's "release only what you leased"
    invariant stays trivially correct -- there is nothing to release.
    """

    async def open(self, *, cadence: str, locale: str) -> SessionLease:
        """Lease one admitted session, or raise :class:`AdmissionBusyError`."""
        ...


class AdmissionBusyError(Exception):
    """No slot was available to admit an ephemeral session.

    Raised by :meth:`SessionFactory.open`; no lease is minted, so the
    orchestrator has nothing to release. Each transport projects this to
    its own busy status (Riva ``BUSY`` / HTTP 429 / NIM shape).
    """


class FinalizationTimeoutError(Exception):
    """The finalization drain exceeded the composed deadline.

    Distinct from an engine-originated ``TimeoutError``: this is raised
    only when the orchestrator's own composed deadline expired around
    ``flush`` (``asyncio.Timeout.expired()`` is true). A ``TimeoutError``
    the engine itself raises is a FAILED finalization and propagates
    unchanged -- never re-labelled as deadline expiry.
    """

    def __init__(self, drain_timeout_s: float) -> None:
        super().__init__(
            f"finalization drain exceeded {drain_timeout_s} s; "
            "engine state aborted, no final result emitted"
        )
        self.drain_timeout_s = drain_timeout_s


@dataclass(frozen=True)
class TranscriptionResult:
    """The result of one ephemeral transcription (PORT-EPH-002).

    Minimal by intent: ``text`` is the exactly-one final transcript. The
    deferred structured-output scope (verbose_json segments / word
    offsets, detected language, duration) lands later as OPTIONAL fields
    on this frozen class -- widening room, never a signature break.
    """

    text: str


def _compose_deadline(
    finalization_timeout_s: float, deadline: float | None
) -> float:
    """Compose the effective finalization bound (A2: serving owns it).

    ``finalization_timeout_s`` is the mandatory finite bound
    (ING-LIFE-005); ``deadline`` is the transport's remaining time
    budget in seconds (e.g. ``grpc.aio`` ``time_remaining()``), or
    ``None`` when the transport set none. The drain runs under the
    earlier of the two, composed here in the ONE place that owns
    deadlines.
    """
    if not math.isfinite(finalization_timeout_s) or finalization_timeout_s <= 0:
        raise ValueError(
            "finalization_timeout_s must be finite and positive, "
            f"got {finalization_timeout_s}"
        )
    if deadline is None:
        return finalization_timeout_s
    return min(finalization_timeout_s, deadline)


async def _drain_within_deadline(lease: SessionLease, drain_timeout: float) -> str:
    """Flush under the composed deadline; translate only true expiry.

    ``asyncio.timeout`` (not ``wait_for``) so THIS deadline's expiry is
    distinguishable from a ``TimeoutError`` the flush itself raises: on
    expiry (``bound.expired()``) a :class:`FinalizationTimeoutError` is
    raised; an engine-originated ``TimeoutError`` propagates unchanged.
    Both outcomes leave the failure to the caller's terminal handler --
    this helper never aborts or releases.
    """
    try:
        async with asyncio.timeout(drain_timeout) as bound:
            return await lease.flush()
    except TimeoutError:
        if bound.expired():
            raise FinalizationTimeoutError(drain_timeout) from None
        raise


async def transcribe_ephemeral(
    pieces: Iterable[FloatSamples],
    *,
    factory: SessionFactory,
    cadence: str,
    locale: str,
    finalization_timeout_s: float,
    submit_bound_samples: int,
    deadline: float | None = None,
) -> TranscriptionResult:
    """Run one canonical ephemeral transcription (PORT-EPH-001/002).

    Open a lease through ``factory`` for the opaque model-typed
    ``cadence``/``locale``; feed the ``pieces`` -- slicing each block so
    no single ``feed`` exceeds ``submit_bound_samples`` (the serving
    layer's pre-submit bound, ING-FE-005; NOT a cadence, ING-FE-006) --
    then drain the final-tail with ``flush`` under the composed
    deadline, take the single terminal ``finish``, free the slot, and
    return exactly one :class:`TranscriptionResult`.

    On ANY error or cancellation the lease is ``abort``-ed and
    ``release``-d idempotently and NO result is emitted: the terminal
    call is ``finish`` XOR ``abort``, never both, and the slot is freed
    only after that cleanup. A ``flush`` that hangs past the composed
    deadline aborts with :class:`FinalizationTimeoutError`; an
    engine-originated ``TimeoutError`` is NOT misclassified as deadline
    expiry.

    Args:
        pieces: The decoded clip as an iterable of contiguous float32
            blocks. A single clip is passed as a one-element iterable
            (``(clip,)``) -- a bare 1-D array must NOT be passed, as
            iterating it would yield samples, not blocks. An empty clip
            (zero blocks, or blocks of length zero) is valid: no feed
            runs and ``flush`` drains the zero-sample final tail
            (PORT-SESS-003).
        factory: The admission/leasing seam; see :class:`SessionFactory`.
        cadence: Opaque model-typed cadence label for the factory.
        locale: Opaque model-typed locale for the factory.
        finalization_timeout_s: The mandatory finite drain bound
            (ING-LIFE-005).
        submit_bound_samples: The maximum sample count handed to any one
            ``feed`` call. The serving layer computes it to respect the
            pre-submit bound; the orchestrator does no ms-to-samples or
            cadence math itself.
        deadline: The transport's remaining time budget in seconds, or
            ``None``; composed with ``finalization_timeout_s`` here.

    Returns:
        The single final transcription result.

    Raises:
        AdmissionBusyError: If no slot was available (no lease minted).
        FinalizationTimeoutError: If the drain exceeded the composed deadline.
        ValueError: On a non-finite/non-positive bound.
        Exception: Any feed/flush/finish failure or cancellation
            propagates after idempotent abort + release.
    """
    if submit_bound_samples < 1:
        raise ValueError(
            f"submit_bound_samples must be positive, got {submit_bound_samples}"
        )
    drain_timeout = _compose_deadline(finalization_timeout_s, deadline)

    # open() is before the lease exists: a raised AdmissionBusyError mints no
    # lease, so there is nothing to release -- it simply propagates.
    lease = await factory.open(cadence=cadence, locale=locale)

    try:
        for block in pieces:
            for start in range(0, len(block), submit_bound_samples):
                await lease.feed(block[start : start + submit_bound_samples])
        transcript = await _drain_within_deadline(lease, drain_timeout)
    except BaseException:
        # Feed failure, flush failure (incl. FinalizationTimeoutError and
        # engine TimeoutError), or cancellation: the terminal call is
        # abort, and the slot is freed after it -- even if abort raises.
        try:
            await lease.abort()
        finally:
            await lease.release()
        raise

    # Flush drained cleanly within the deadline: finish is the single
    # terminal call, and release frees the slot after it -- even if
    # finish raises. finish never runs alongside abort.
    try:
        await lease.finish()
    finally:
        await lease.release()
    return TranscriptionResult(text=transcript)
