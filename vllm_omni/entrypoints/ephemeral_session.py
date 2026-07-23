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
    returning ``None``: both ``open`` methods keep a total ``SessionLease``
    return so no caller threads an Optional or risks proceeding on a
    ``None``; the busy condition is exceptional (the shared pool is full)
    and each transport already translates errors to its own busy status
    at its boundary; and because a raised ``AdmissionBusyError`` means no
    lease was ever minted, the orchestrator's "release only what you
    leased" invariant stays trivially correct -- there is nothing to
    release.

    Two entry points, one per regime. ``open_ephemeral`` leases the
    single-shot session at the model's OWN canonical geometry
    (PORT-REGIME-002): the caller passes no cadence, so no transport can
    pick the wrong one and the canonical-geometry choice lives once, in
    the model-aware factory. ``open`` keeps the general cadence-selecting
    entry for realtime sessions, where the admitted cadence is a genuine
    per-session parameter.
    """

    async def open_ephemeral(self, *, locale: str) -> SessionLease:
        """Lease one canonical single-shot session (PORT-REGIME-002).

        The factory selects the model's canonical ephemeral geometry;
        the caller supplies only ``locale``. Raises
        :class:`AdmissionBusyError` when no slot is free.
        """
        ...

    async def open(self, *, cadence: str, locale: str) -> SessionLease:
        """Lease one realtime session at ``cadence``, or raise
        :class:`AdmissionBusyError`."""
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


def _remaining_drain_budget(
    finalization_timeout_s: float,
    transport_deadline_at: float | None,
    now: float,
) -> float:
    """The finalization bound evaluated FRESH at drain time (A2).

    ``finalization_timeout_s`` is the mandatory finite bound
    (ING-LIFE-005). ``transport_deadline_at`` is the transport's
    deadline as an ABSOLUTE monotonic-clock instant captured at entry
    (``None`` when the transport set none), and ``now`` is the current
    monotonic time. The drain runs under the earlier of the mandatory
    bound and the transport's REMAINING time -- computed here, after
    feeding, so a slow feed cannot hand finalization a stale budget. A
    transport already out of time yields a non-positive budget, which
    fires the timeout immediately.
    """
    if transport_deadline_at is None:
        return finalization_timeout_s
    return min(finalization_timeout_s, transport_deadline_at - now)


async def _finalize_within_deadline(
    lease: SessionLease, drain_timeout: float
) -> str:
    """Drain ``flush`` THEN ``finish`` under ONE deadline (PORT-EPH-002).

    Both terminal steps are bounded together: a stalled ``finish`` can
    no longer retain the slot after a clean ``flush``. ``asyncio.timeout``
    (not ``wait_for``) keeps THIS deadline's expiry distinguishable from
    a ``TimeoutError`` the engine itself raises -- on expiry
    (``bound.expired()``) a :class:`FinalizationTimeoutError` is raised;
    an engine-originated ``TimeoutError`` propagates unchanged. This
    helper never aborts or releases: any failure (expiry, or a raising
    flush/finish) leaves the terminal handling to the caller, which
    aborts then releases.
    """
    try:
        async with asyncio.timeout(drain_timeout) as bound:
            transcript = await lease.flush()
            await lease.finish()
            return transcript
    except TimeoutError:
        if bound.expired():
            raise FinalizationTimeoutError(drain_timeout) from None
        raise


async def transcribe_ephemeral(
    pieces: Iterable[FloatSamples],
    *,
    factory: SessionFactory,
    locale: str,
    finalization_timeout_s: float,
    submit_bound_samples: int,
    deadline: float | None = None,
) -> TranscriptionResult:
    """Run one canonical ephemeral transcription (PORT-EPH-001/002).

    Lease the model's canonical single-shot session through
    ``factory.open_ephemeral`` (the caller passes no cadence -- the
    canonical geometry is the model-aware factory's, not each
    transport's, PORT-REGIME-002); feed the ``pieces`` -- slicing each block so no
    single ``feed`` exceeds ``submit_bound_samples`` (the serving
    layer's pre-submit bound, ING-FE-005; NOT a cadence, ING-FE-006) --
    then drain the final-tail with ``flush`` AND take the terminal
    ``finish`` together under one deadline, free the slot, and return
    exactly one :class:`TranscriptionResult`.

    On ANY error or cancellation -- including a ``flush`` or ``finish``
    that hangs past the deadline, or a raising ``finish`` -- the lease is
    ``abort``-ed then ``release``-d and NO result is emitted: the
    successful terminal call is ``finish``, else ``abort`` recovers, and
    the slot is freed only after that cleanup on every path. A drain that
    outlives the deadline raises :class:`FinalizationTimeoutError`; an
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
        locale: Opaque model-typed locale for the factory.
        finalization_timeout_s: The mandatory finite drain bound
            (ING-LIFE-005).
        submit_bound_samples: The maximum sample count handed to any one
            ``feed`` call. The serving layer computes it to respect the
            pre-submit bound; the orchestrator does no ms-to-samples or
            cadence math itself.
        deadline: The transport's remaining time budget in seconds AT
            CALL TIME, or ``None``. Captured as an absolute monotonic
            instant so feeding does not eat into the finalization budget;
            composed with ``finalization_timeout_s`` fresh at drain time.

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
    if not math.isfinite(finalization_timeout_s) or finalization_timeout_s <= 0:
        raise ValueError(
            "finalization_timeout_s must be finite and positive, "
            f"got {finalization_timeout_s}"
        )
    # Anchor the transport deadline to an absolute monotonic instant NOW,
    # before admission and feeding, so its remaining budget is measured
    # from call time -- a slow feed shortens finalization's budget rather
    # than leaving it stale.
    loop = asyncio.get_running_loop()
    transport_deadline_at = None if deadline is None else loop.time() + deadline

    # open_ephemeral is before the lease exists: a raised AdmissionBusyError
    # mints no lease, so there is nothing to release -- it just propagates.
    lease = await factory.open_ephemeral(locale=locale)

    try:
        try:
            for block in pieces:
                for start in range(0, len(block), submit_bound_samples):
                    await lease.feed(block[start : start + submit_bound_samples])
            drain_timeout = _remaining_drain_budget(
                finalization_timeout_s, transport_deadline_at, loop.time()
            )
            transcript = await _finalize_within_deadline(lease, drain_timeout)
        except BaseException:
            # Any failure -- feed, a raising or timed-out flush/finish, or
            # cancellation -- recovers through abort; the original failure
            # dominates (a raising abort must not mask why we aborted).
            try:
                await lease.abort()
            except Exception:
                pass
            raise
    finally:
        # The slot is freed after terminal cleanup (finish on success,
        # abort on failure) on EVERY path.
        await lease.release()
    return TranscriptionResult(text=transcript)
