"""The session core: sans-IO behavior owner for every ingress surface.

One session core owns admission, buffering bounds, finalization, and
validation semantics exactly once (ING-CORE-001); adapters are dialect
codecs over :mod:`nemotron_asr_ingress.events`. Time is always
an explicit ``now`` argument (the :class:`IdleClock` pattern), so the
GPU-free tier tests every branch without sleeping.

This module is the durable home for the values every tier shares
(:data:`SAMPLE_RATE`, :data:`VALID_CHUNK_MS`, :class:`IdleClock`):
serve imports from ingress, never the reverse.
"""

import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.events import (
    AdmissionOutcome,
    Admitted,
    Busy,
    Configure,
    Event,
    Final,
    Partial,
    SessionError,
    UpdateAck,
)

FloatAudio = npt.NDArray[np.float32]

SAMPLE_RATE = 16000

#: The checkpoint's supported streaming chunk sizes.
VALID_CHUNK_MS = (80, 160, 320, 560, 1120)


class IdleClock:
    """Receive-cadence idle clock for one admitted session."""

    def __init__(self, ttl_s: float, now: float) -> None:
        """Start the clock at admission time ``now``."""
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be positive, got {ttl_s}")
        self.ttl_s = ttl_s
        self._last_receive = now

    def touch(self, now: float) -> None:
        """Record a wire receive."""
        self._last_receive = now

    def remaining(self, now: float) -> float:
        """Seconds until expiry (<= 0 when expired)."""
        return self._last_receive + self.ttl_s - now

    def expired(self, now: float) -> bool:
        """Whether the session has been idle past the TTL."""
        return self.remaining(now) <= 0


#: Mid-session updates touching these fields answer
#: ``config_change_rejected`` (ING-LIFE-008, PORT-SESS-002).
ADMISSION_FIXED_FIELDS = ("chunk_ms", "encoding", "sample_rate_hz")


class Transcriber(Protocol):
    """The compute seam the session core drives (PORT-owned behavior).

    Async because the in-process binding submits directly through the
    engine's own streaming-generation entry point (ING-VEH-004); a
    remote-dialect or offline implementation may still complete
    immediately, but the seam is colored async throughout so no
    caller bridges a thread or a queue to reach it (ING-VEH-004,
    ING-VEH-005). ``step`` consumes one exact admitted-config chunk
    and returns the cumulative hypothesis; ``flush`` consumes the raw
    un-padded sub-chunk residual (PORT-SESS-003 owns the tail
    transform) and returns the final transcript; ``update_locale``
    forwards a validated mid-session locale (PORT-LID-003 applies it
    at the next chunk boundary); ``abort`` frees engine state
    immediately on a non-normal end (client close, detected
    disconnect, idle-TTL); ``finish`` frees engine state on the
    normal end of a session, called exactly once by every
    ``finalize`` -- including the skip-flush path where ``flush``
    never runs (ING-LIFE-003) -- so an implementation holding a live
    resource (e.g. an open engine generation) always gets exactly one
    terminal call on every path, success or not (ING-VEH-004).
    Deliberately not named to match ``SessionCore.close`` (the
    client-disconnect path, which calls ``abort``): the two names
    would otherwise cross.
    """

    async def step(self, chunk: FloatAudio) -> str:
        """Advance one chunk; return the cumulative hypothesis."""
        ...

    async def flush(self, residual: FloatAudio) -> str:
        """Finalize with the raw residual; return the final transcript."""
        ...

    async def update_locale(self, target_lang: str) -> None:
        """Forward a validated locale change."""
        ...

    async def abort(self) -> None:
        """Free engine state immediately (close / idle abort)."""
        ...

    async def finish(self) -> None:
        """Free engine state on a normal finalize, flushed or not.

        Called exactly once by ``SessionCore.finalize`` regardless of
        whether ``flush`` ran; never called on the abort path (abort
        already frees state). A transcriber with no persistent
        resource (e.g. a synchronous stub) may treat this as a no-op.
        """
        ...


@dataclass(frozen=True)
class IngressValues:
    """ING-owned tuning values (carried by ENV per ENV-MOD-003).

    ``watermark`` is W (PORT-STATE-004 residency cap);
    ``admission_queue`` is the in-process gate's bounded queue length
    (0 = fast-fail); ``admission_wait_s`` is the ING-owned wait bound
    PORT-SESS-005 carves out; ``chunk_buffer_s`` bounds the re-chunk
    buffer in seconds of audio; ``pre_roll_bytes`` bounds the raw
    pre-admission buffer; ``idle_ttl_s`` is the PORT-SESS-005 TTL;
    ``locales`` is the checkpoint-derived valid locale set
    (PORT-LID-001).
    """

    watermark: int
    admission_queue: int
    admission_wait_s: float
    chunk_buffer_s: float
    pre_roll_bytes: int
    idle_ttl_s: float
    locales: tuple[str, ...]


# @spec ING-FE-005, ING-FE-006
class ChunkBuffer:
    """Bounded re-chunk buffer: float32 in, exact admitted chunks out.

    The single implementation of ING-FE-005/006's buffering: arbitrary
    -size decoded audio accumulates; exact ``chunk_samples`` chunks pop
    out; whatever remains at finalize is the raw un-padded residual.
    """

    def __init__(self, chunk_samples: int, max_seconds: float) -> None:
        """Bound the buffer at ``max_seconds`` of 16 kHz audio."""
        if chunk_samples <= 0:
            raise ValueError(
                f"chunk_samples must be positive, got {chunk_samples}"
            )
        if max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds}")
        self._chunk_samples = chunk_samples
        self._max_samples = int(max_seconds * SAMPLE_RATE)
        self._parts: list[FloatAudio] = []
        self._held = 0

    def append(self, samples: FloatAudio) -> bool:
        """Accumulate; return ``False`` when the bound would be exceeded."""
        if self._held + len(samples) > self._max_samples:
            return False
        if len(samples):
            self._parts.append(np.asarray(samples, dtype=np.float32))
            self._held += len(samples)
        return True

    def pop_chunks(self) -> list[FloatAudio]:
        """Drain every complete admitted-size chunk, in order."""
        n_ready = self._held // self._chunk_samples
        if n_ready == 0:
            return []
        joined = np.concatenate(self._parts)
        chunks = [
            joined[i * self._chunk_samples : (i + 1) * self._chunk_samples]
            for i in range(n_ready)
        ]
        rest = joined[n_ready * self._chunk_samples :]
        self._parts = [rest] if len(rest) else []
        self._held = len(rest)
        return chunks

    def residual(self) -> FloatAudio:
        """The sub-chunk remainder, raw and un-padded."""
        if not self._parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._parts)


PreRollItem = tuple[
    Literal["audio", "update", "finalize"],
    bytes | Mapping[str, Any] | None,
]


# @spec ING-ADM-005
class PreRollBuffer:
    """Bounded pre-admission buffer of raw client bytes (ING-ADM-005).

    Holds *undecoded* client payloads plus held ``finalize``/``update``
    events in arrival order while an admission outcome is pending; no
    decode work is spent on a never-admitted session. ``release``
    returns everything in order on ``admitted``; ``discard`` drops the
    whole buffer on ``busy``.

    Overflow doctrine (Pete, PR #13 review): pre-admission overflow is
    ``buffer_overflow`` — fatal to the attempt, session never admitted,
    no audio processed, safe to retry with backoff; avoid by waiting
    for the admission answer before streaming ahead of it. The whole
    buffer is discarded (never a processed prefix), the pending
    admission resolves negatively, and its queue slot frees; the
    watermark is untouched because nothing was ever admitted. Distinct
    from ``busy`` on purpose: ``busy`` is fixed by capacity/backoff,
    ``buffer_overflow`` by client pacing. This doctrine must reach the
    servicer's user-facing docs when the downstream package ships.
    """

    def __init__(self, max_bytes: int) -> None:
        """Bound the held audio at ``max_bytes`` of raw payload."""
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self._max_bytes = max_bytes
        self._held_bytes = 0
        self._items: list[PreRollItem] = []

    def hold_audio(self, raw: bytes) -> bool:
        """Hold one raw payload; ``False`` when the bound is exceeded."""
        if self._held_bytes + len(raw) > self._max_bytes:
            return False
        self._items.append(("audio", raw))
        self._held_bytes += len(raw)
        return True

    def hold_update(self, fields: Mapping[str, Any]) -> None:
        """Hold a pre-admission update for the outcome."""
        self._items.append(("update", dict(fields)))

    def hold_finalize(self) -> None:
        """Hold a pre-admission finalize for the outcome."""
        self._items.append(("finalize", None))

    def release(self) -> list[PreRollItem]:
        """Everything held, in arrival order (on ``admitted``)."""
        items = self._items
        self._items = []
        self._held_bytes = 0
        return items

    def discard(self) -> None:
        """Drop the whole buffer (on ``busy``); nothing half-processed."""
        self._items = []
        self._held_bytes = 0


# @spec ING-ADM-002, ING-ADM-004
class InProcessGate:
    """The authoritative admission point (ledger A11, ING-ADM-002).

    Wraps the watermark with a bounded admission queue: admission
    authority follows eviction visibility, so only this gate may
    queue. ``request`` answers ``ADMITTED``/``BUSY`` immediately or
    ``None`` when the session is queued; ``poll`` resolves a queued
    session to ``ADMITTED`` (a slot freed, in queue order) or
    ``BUSY``-shaped release via the admission-wait timeout — the
    caller distinguishes the timeout by ``waited_out``.

    Thread-safe by construction: the β shell runs each connection's
    sans-IO stack on an executor thread, so gate calls race — every
    compound check-then-mutate holds one internal lock.
    """

    def __init__(self, watermark: int, queue_len: int, wait_s: float) -> None:
        """Set W, the bounded queue length, and the wait bound."""
        if watermark < 1:
            raise ValueError(f"watermark must be >= 1, got {watermark}")
        if queue_len < 0:
            raise ValueError(f"queue_len must be >= 0, got {queue_len}")
        if wait_s <= 0:
            raise ValueError(f"wait_s must be positive, got {wait_s}")
        self._watermark = watermark
        self._queue_len = queue_len
        self._wait_s = wait_s
        self._lock = threading.Lock()
        self._admitted: set[str] = set()
        self._queue: list[str] = []
        self._enqueued_at: dict[str, float] = {}

    @property
    def active(self) -> int:
        """Currently admitted (resident) sessions, parked included."""
        with self._lock:
            return len(self._admitted)

    def request(self, session_id: str, now: float) -> AdmissionOutcome | None:
        """Admit, reject, or (``None``) enqueue a configure."""
        with self._lock:
            # A freed slot goes to the queue head, never to a newcomer.
            if not self._queue and len(self._admitted) < self._watermark:
                self._admitted.add(session_id)
                return AdmissionOutcome.ADMITTED
            if len(self._queue) < self._queue_len:
                self._queue.append(session_id)
                self._enqueued_at[session_id] = now
                return None
            return AdmissionOutcome.BUSY

    def poll(self, session_id: str, now: float) -> AdmissionOutcome | None:
        """Resolve a queued session; ``None`` while still waiting."""
        with self._lock:
            if session_id in self._admitted:
                return AdmissionOutcome.ADMITTED
            if session_id not in self._queue:
                return AdmissionOutcome.BUSY
            if self._waited_out(session_id, now):
                self._queue.remove(session_id)
                return AdmissionOutcome.BUSY
            if self._queue[0] == session_id and (
                len(self._admitted) < self._watermark
            ):
                self._queue.remove(session_id)
                del self._enqueued_at[session_id]
                self._admitted.add(session_id)
                return AdmissionOutcome.ADMITTED
            return None

    def waited_out(self, session_id: str, now: float) -> bool:
        """Whether a queued session exceeded the admission-wait bound."""
        with self._lock:
            return self._waited_out(session_id, now)

    def _waited_out(self, session_id: str, now: float) -> bool:
        """The unlocked check (callers hold the lock)."""
        enqueued_at = self._enqueued_at.get(session_id)
        return enqueued_at is not None and now - enqueued_at > self._wait_s

    def release(self, session_id: str) -> None:
        """Free one admitted session's slot (close/final/abort)."""
        with self._lock:
            if session_id in self._admitted:
                self._admitted.remove(session_id)
                return
            if session_id in self._enqueued_at:
                if session_id in self._queue:
                    self._queue.remove(session_id)
                del self._enqueued_at[session_id]
                return
            raise RuntimeError(
                f"release() for a session the gate never saw: {session_id}"
            )


# @spec ING-CORE-001, ING-ERR-003, ING-ERR-004
class SessionCore:
    """One session's behavior owner (sans-IO).

    Every method takes the event and explicit ``now`` and returns the
    ordered list of outbound events the adapter must surface. After a
    ``finalize`` is accepted the session is terminal: audio and
    finalize-class events answer ``session_terminal`` (ING-ERR-004),
    and exactly one ``Final`` ever emits (ING-LIFE-002).
    """

    def __init__(
        self,
        gate: InProcessGate,
        transcriber: Transcriber,
        values: IngressValues,
        provenance: Mapping[str, Any],
        session_id: str | None = None,
    ) -> None:
        """Bind one session to the shared gate and its transcriber."""
        self._gate = gate
        self._transcriber = transcriber
        self.values = values
        self._provenance = provenance
        self.session_id = (
            session_id if session_id is not None else f"sess-{uuid.uuid4().hex}"
        )
        self._configured: Configure | None = None
        self._queued = False
        self._slot_held = False
        self._terminal = False
        self._buffer: ChunkBuffer | None = None
        self._idle: IdleClock | None = None
        self._chunk_index = 0
        self._cumulative = ""

    @property
    def terminal(self) -> bool:
        """Whether the session takes no further input."""
        return self._terminal

    @property
    def config(self) -> Configure | None:
        """The session's config (set by ``configure``, admission-fixed).

        The kernel-side transcriber reads it at first step: the seam's
        callables carry no config, and the core is the config's owner.
        """
        return self._configured

    # @spec ING-ADM-001, ING-CORE-003, ING-CORE-004
    def configure(self, config: Configure, now: float) -> list[Event]:
        """Admission handshake: the outcome answers this configure."""
        if self._terminal:
            return [self._terminal_error()]
        if self._configured is not None:
            return [
                SessionError(
                    code=errors.PROTOCOL_ORDER,
                    detail="the session is already configured",
                )
            ]
        # Validation precedes the gate: a rejected config never takes
        # a slot (ING-ADM-001 answers only an admissible configure).
        if config.chunk_ms not in VALID_CHUNK_MS:
            self._terminal = True
            return [
                SessionError(
                    code=errors.INVALID_CONFIG_FIELD,
                    fields=("chunk_ms",),
                    detail=(
                        f"chunk_ms must be one of {VALID_CHUNK_MS}, "
                        f"got {config.chunk_ms}"
                    ),
                )
            ]
        if config.target_lang not in self.values.locales:
            self._terminal = True
            return [
                SessionError(
                    code=errors.UNKNOWN_LOCALE,
                    fields=("target_lang",),
                    detail=f"unknown locale {config.target_lang!r}",
                )
            ]
        self._configured = config
        outcome = self._gate.request(self.session_id, now)
        if outcome is None:
            self._queued = True
            return []
        if outcome is AdmissionOutcome.BUSY:
            self._terminal = True
            return [Busy(detail="admission watermark full")]
        return [self._admit(now)]

    # @spec ING-CORE-005, ING-FE-006
    async def receive_audio(self, samples: FloatAudio, now: float) -> list[Event]:
        """Accumulate decoded audio; step every ready chunk."""
        if self._terminal:
            return [self._terminal_error()]
        if self._buffer is None or self._idle is None:
            return [
                SessionError(
                    code=errors.PROTOCOL_ORDER,
                    detail="audio before the admission outcome",
                )
            ]
        self._idle.touch(now)
        events: list[Event] = []
        if not self._buffer.append(samples):
            events.append(
                SessionError(
                    code=errors.BUFFER_OVERFLOW,
                    detail=(
                        "re-chunk buffer bound exceeded "
                        f"({self.values.chunk_buffer_s} s); audio dropped"
                    ),
                )
            )
        for chunk in self._buffer.pop_chunks():
            self._cumulative = await self._transcriber.step(chunk)
            events.append(
                Partial(
                    cumulative=self._cumulative, chunk_index=self._chunk_index
                )
            )
            self._chunk_index += 1
        return events

    # @spec ING-LIFE-007, ING-LIFE-008, ING-LIFE-009
    async def update(self, fields: Mapping[str, Any], now: float) -> list[Event]:
        """Mid-session update: rejections first, truthful ack last."""
        if self._terminal:
            return [self._terminal_error()]
        if self._idle is None:
            return [
                SessionError(
                    code=errors.PROTOCOL_ORDER,
                    detail="update before the admission outcome",
                )
            ]
        self._idle.touch(now)
        rejections: list[Event] = []
        honored: dict[str, Any] = {}
        for key, value in fields.items():
            if key == "target_lang":
                if value not in self.values.locales:
                    rejections.append(
                        SessionError(
                            code=errors.UNKNOWN_LOCALE,
                            fields=("target_lang",),
                            detail=f"unknown locale {value!r}",
                        )
                    )
                else:
                    await self._transcriber.update_locale(value)
                    honored["target_lang"] = value
            elif key in ADMISSION_FIXED_FIELDS:
                rejections.append(
                    SessionError(
                        code=errors.CONFIG_CHANGE_REJECTED,
                        fields=(key,),
                        detail=f"{key} is fixed at admission",
                    )
                )
            else:
                # Never a silent drop (ING-ERR-001).
                rejections.append(
                    SessionError(
                        code=errors.INVALID_CONFIG_FIELD,
                        fields=(key,),
                        detail=f"unknown update field {key!r}",
                    )
                )
        # The ack is last and truthful (ING-LIFE-009): emitted when
        # anything was honored, or for a no-op update.
        if honored or not rejections:
            return rejections + [UpdateAck(honored=honored)]
        return rejections

    # @spec ING-LIFE-002, ING-LIFE-003
    async def finalize(self, now: float) -> list[Event]:
        """Client-driven end of audio; exactly one ``Final``."""
        if self._terminal:
            return [self._terminal_error()]
        if self._buffer is None:
            return [
                SessionError(
                    code=errors.PROTOCOL_ORDER,
                    detail="finalize before the admission outcome",
                )
            ]
        residual = self._buffer.residual()
        if len(residual):
            transcript = await self._transcriber.flush(residual)
        else:
            # Skip-flush: PORT's tail transform never runs on nothing
            # (ING-LIFE-003); an empty session finalizes to "".
            transcript = self._cumulative
        self._terminal = True
        self._release_slot()
        # Unconditional, flush-or-not: the one terminal call every
        # transcriber gets on the normal end of a session (never
        # skipped alongside flush -- see the Transcriber.finish
        # docstring for why the skip-flush path still needs this).
        await self._transcriber.finish()
        return [Final(transcript=transcript)]

    # @spec ING-LIFE-004
    async def close(self, now: float) -> list[Event]:
        """Immediate teardown (client close or detected disconnect)."""
        if self._queued:
            self._queued = False
            self._gate.release(self.session_id)
            self._terminal = True
            return []
        if self._slot_held:
            await self._transcriber.abort()
            self._release_slot()
        self._terminal = True
        return []

    # @spec ING-ADM-002, ING-LIFE-005, ING-LIFE-006
    async def poll(self, now: float) -> list[Event]:
        """Timer-driven events: queued admission, wait/idle timeouts."""
        if self._terminal:
            return []
        if self._queued:
            outcome = self._gate.poll(self.session_id, now)
            if outcome is None:
                return []
            self._queued = False
            if outcome is AdmissionOutcome.ADMITTED:
                self._slot_held = True
                return [self._finish_admission(now)]
            self._terminal = True
            if self._gate.waited_out(self.session_id, now):
                self._gate.release(self.session_id)
                return [
                    SessionError(
                        code=errors.ADMISSION_WAIT_TIMEOUT,
                        detail=(
                            "queued past the admission-wait bound "
                            f"({self.values.admission_wait_s} s)"
                        ),
                    )
                ]
            return [Busy(detail="admission watermark full")]
        if self._idle is not None and self._idle.expired(now):
            await self._transcriber.abort()
            self._release_slot()
            self._terminal = True
            return [
                SessionError(
                    code=errors.IDLE_TIMEOUT,
                    detail=(
                        f"no audio for {self.values.idle_ttl_s} s "
                        "(idle-TTL backstop)"
                    ),
                )
            ]
        return []

    def _admit(self, now: float) -> Admitted:
        """Record an already-granted slot and start session state."""
        self._slot_held = True
        return self._finish_admission(now)

    def _finish_admission(self, now: float) -> Admitted:
        """Build per-session state; the idle clock starts at admission."""
        config = self._configured
        if config is None:
            raise RuntimeError("admission finished without a configure")
        chunk_samples = config.chunk_ms * (SAMPLE_RATE // 1000)
        self._buffer = ChunkBuffer(
            chunk_samples=chunk_samples,
            max_seconds=self.values.chunk_buffer_s,
        )
        self._idle = IdleClock(ttl_s=self.values.idle_ttl_s, now=now)
        return Admitted(session_id=self.session_id, provenance=self._provenance)

    def _release_slot(self) -> None:
        """Free the gate slot exactly once per session."""
        if self._slot_held:
            self._slot_held = False
            self._gate.release(self.session_id)

    def _terminal_error(self) -> SessionError:
        """The one post-finalize answer (ING-ERR-004)."""
        return SessionError(
            code=errors.SESSION_TERMINAL,
            detail="the session is terminal; no further input is taken",
        )
