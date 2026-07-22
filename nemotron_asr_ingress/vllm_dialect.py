"""Server-side vLLM ``/v1/realtime`` dialect binding for the β server.

A sans-IO dialect codec (wire JSON dicts in, wire JSON dicts out) over
the session core — ING-1's convergence: the β server speaks the pin's
realtime dialect so one shared client drives both β and stock
upstream/omni. Wire shapes are those of
``vllm/entrypoints/speech_to_text/realtime/protocol.py`` @ the PIN
(``ee0da84ab``; verified byte-identical to ``702f4814f``).

Dialect mechanics honored from the pin: ``session.created`` is sent on
connect; ``session.update`` is mandatory before any commit
(``model_not_validated`` otherwise); append audio is base64 PCM16 @
16 kHz decoded int16 -> float32/32768. Deviations that are the point:
the pin's silently-ignored second commit surfaces a catalog error
(ING-ERR-004), and two additive extensions ride ``session.update``
(``chunk_ms``, ``target_lang`` — upstream ignores unknown fields
there) plus additive server events stock upstream simply never sends:
``session.admitted`` (provenance, ING-CORE-004) answering admission,
and ``session.updated`` acking the honored subset of a mid-session
update (ING-LIFE-009's truthful-ack rule needs a wire event to ride).

Admission answers ``session.update`` itself, never the first append:
the additive fields complete the config there (ADM-001's the-outcome-
answers-the-configure, and the OpenAI-Realtime ``transcription_
session.update -> updated`` ack pattern the RFC conformance note can
converge on). ``session.admitted`` stays distinct from the connect-
time ``session.created`` — a connect-time event cannot carry a
deferred outcome once the gate queues (A11 reserves that path).
Decided in PR #13 review (Pete, 2026-07-12). When the gate queues, the
answer is deferred to ``poll`` — never raced onto an append's reply —
and on ``admitted`` the pre-roll drains *after* the ``session.
admitted`` event, so provenance always precedes the first delta.
"""

import base64
import binascii
from collections.abc import Mapping
from typing import Any

import numpy as np

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.core import (
    FloatAudio,
    PreRollBuffer,
    SessionCore,
)
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

#: The two additive request fields riding ``session.update``.
_EXTENSION_FIELDS = ("chunk_ms", "target_lang")


# @spec ING-CORE-001, ING-LIFE-001
class VllmRealtimeAdapter:
    """One connection's dialect codec over one session core."""

    def __init__(self, core: SessionCore, model_name: str) -> None:
        """Bind a session core and the served model's name."""
        self._core = core
        self._model_name = model_name
        self._pre_roll = PreRollBuffer(core.values.pre_roll_bytes)
        self._configured = False
        self._queued = False
        self._nonfinal_commits = 0
        self._byte_tail = b""
        self._last_cumulative = ""

    @property
    def terminal(self) -> bool:
        """Whether the session takes no further input (core-owned)."""
        return self._core.terminal

    @property
    def should_close(self) -> bool:
        """Always ``False``: this dialect's errors are in-stream.

        The pin's realtime dialect never closes the transport on an
        error event — session termination drives the close (the
        shell's ``ShellAdapter`` seam carries the attribute so the
        NIM dialect's ``error+close`` projections can share the
        shell).
        """
        return False

    def on_connect(self) -> list[dict[str, Any]]:
        """Wire events sent on connect (``session.created``)."""
        return [{"type": "session.created", "id": self._core.session_id}]

    # @spec ING-LIFE-004
    async def on_disconnect(self, now: float) -> None:
        """A detected transport drop: immediate ``close``, no reply.

        Engine abort and slot free happen now, never left to the
        idle-TTL backstop; a queued attempt leaves the queue; the
        pre-roll drops whole.
        """
        self._queued = False
        self._pre_roll.discard()
        await self._core.close(now)

    async def on_event(
        self, event: Mapping[str, Any], now: float
    ) -> list[dict[str, Any]]:
        """Translate one client wire event; return wire events out."""
        event_type = event.get("type")
        if event_type == "session.update":
            return await self._on_session_update(event, now)
        if event_type == "input_audio_buffer.append":
            return await self._on_append(event, now)
        if event_type == "input_audio_buffer.commit":
            return await self._on_commit(event, now)
        return [
            self._error("unknown_event", f"unknown event type {event_type!r}")
        ]

    async def poll(self, now: float) -> list[dict[str, Any]]:
        """Surface timer-driven core events (idle/wait timeouts)."""
        core_events = await self._core.poll(now)
        wire: list[dict[str, Any]] = []
        admitted = False
        for core_event in core_events:
            wire.extend(self.project(core_event))
            if isinstance(core_event, Admitted):
                admitted = True
        if self._queued and core_events:
            self._queued = False
            if admitted:
                wire.extend(await self._drain_pre_roll(now))
            else:
                # Negative outcome: the whole pre-roll drops, nothing
                # half-processed (ING-ADM-005).
                self._pre_roll.discard()
        return wire

    # @spec ING-CORE-005, ING-ERR-001
    def project(self, event: Event) -> list[dict[str, Any]]:
        """Project one session-core stream event to wire events."""
        if isinstance(event, Admitted):
            return [
                {
                    "type": "session.admitted",
                    "id": event.session_id,
                    "provenance": dict(event.provenance),
                }
            ]
        if isinstance(event, Busy):
            return [
                self._error("busy", event.detail or "admission watermark full")
            ]
        if isinstance(event, Partial):
            delta = self._delta_from(event.cumulative)
            if not delta:
                # A chunk that added nothing (silence) sends nothing:
                # the pin emits deltas per generated token, so an
                # empty-payload delta is dialect noise it never sends.
                return []
            return [{"type": "transcription.delta", "delta": delta}]
        if isinstance(event, Final):
            return [{"type": "transcription.done", "text": event.transcript}]
        if isinstance(event, UpdateAck):
            return [{"type": "session.updated", "honored": dict(event.honored)}]
        detail = event.detail or event.code
        if event.fields:
            detail = f"{detail} (fields: {', '.join(event.fields)})"
        projection = errors.catalog()[event.code]
        return [self._error(projection.vllm or event.code, detail)]

    # @spec ING-CORE-003, ING-CORE-004
    def project_admission(
        self,
        outcome: AdmissionOutcome,
        provenance: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Project the admission outcome onto the wire.

        ``ADMITTED`` becomes the additive ``session.admitted`` event
        (provenance riding it, ING-CORE-004); ``BUSY`` becomes the
        ``busy`` error event.

        Raises:
            ValueError: For any outcome this v1 adapter does not know
                (``QUEUED`` included) — never defaulted to success
                (ING-CORE-003).
        """
        if outcome is AdmissionOutcome.ADMITTED:
            return [
                {
                    "type": "session.admitted",
                    "id": self._core.session_id,
                    "provenance": dict(provenance) if provenance else {},
                }
            ]
        if outcome is AdmissionOutcome.BUSY:
            return [self._error("busy", "admission watermark full")]
        raise ValueError(
            f"admission outcome {outcome!r} is not projectable by the "
            "v1 adapter"
        )

    # @spec ING-ADM-001
    async def _on_session_update(
        self, event: Mapping[str, Any], now: float
    ) -> list[dict[str, Any]]:
        """First update is the configure; later ones are mid-session."""
        if self._queued:
            self._pre_roll.hold_update(self._extension_fields(event))
            return []
        if self._configured:
            core_events = await self._core.update(
                self._extension_fields(event), now
            )
            return self._project_all(core_events)
        model = event.get("model")
        if model is None:
            return [
                self._error("invalid_event", "session.update requires model")
            ]
        if model != self._model_name:
            return [
                self._error("model_not_found", f"model {model!r} is not served")
            ]
        self._configured = True
        config = Configure(
            chunk_ms=event.get("chunk_ms", 560),
            target_lang=event.get("target_lang", "auto"),
        )
        answers = self._core.configure(config, now)
        if not answers:
            self._queued = True
            return []
        return self._project_all(answers)

    # @spec ING-ADM-005
    async def _on_append(
        self, event: Mapping[str, Any], now: float
    ) -> list[dict[str, Any]]:
        """Decode one append through the front-end into the core."""
        if not self._configured:
            return [
                self._error(
                    "model_not_validated",
                    "session.update must precede audio",
                )
            ]
        payload = event.get("audio")
        if not isinstance(payload, str):
            return [
                self._error(
                    "invalid_audio", "audio payload must be a base64 string"
                )
            ]
        if self._queued:
            if self._pre_roll.hold_audio(payload.encode("ascii")):
                return []
            # Pre-admission overflow: fatal to the attempt, nothing
            # processed, queue slot freed, retry-with-backoff safe.
            self._queued = False
            self._pre_roll.discard()
            await self._core.close(now)
            return [
                self._error(
                    "buffer_overflow",
                    "pre-admission buffer bound exceeded; the attempt is "
                    "over (retry after backoff, or wait for the admission "
                    "answer before streaming)",
                )
            ]
        return await self._feed_audio(payload, now)

    # @spec ING-ERR-004, ING-LIFE-002
    async def _on_commit(
        self, event: Mapping[str, Any], now: float
    ) -> list[dict[str, Any]]:
        """Commit: final=true finalizes; the first non-final is quiet."""
        if not self._configured:
            return [
                self._error(
                    "model_not_validated",
                    "session.update must precede commit",
                )
            ]
        if bool(event.get("final", False)):
            if self._queued:
                self._pre_roll.hold_finalize()
                return []
            return self._project_all(await self._core.finalize(now))
        if self._core.terminal:
            return self._project_all(
                [
                    SessionError(
                        code=errors.SESSION_TERMINAL,
                        detail="commit after finalize acceptance",
                    )
                ]
            )
        self._nonfinal_commits += 1
        if self._nonfinal_commits == 1:
            # Pin mechanics: the first non-final commit starts
            # generation; the β decode loop is already streaming.
            return []
        # The pin silently ignores this; the deviation surfaces the raw
        # catalog code — the pin's `model_not_validated` name belongs
        # to config-precedence violations only, and a condition the pin
        # never names rides the catalog string itself (ING-ERR-004).
        return [
            self._error(
                errors.PROTOCOL_ORDER,
                "a second non-final commit while generation runs "
                "(surfaced, never silently dropped)",
            )
        ]

    async def _feed_audio(
        self, payload: str, now: float
    ) -> list[dict[str, Any]]:
        """Base64 -> PCM16 -> float32 into the core (byte-safe)."""
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return [self._error("invalid_audio", "audio payload is not base64")]
        data = self._byte_tail + decoded
        usable = len(data) - (len(data) % 2)
        self._byte_tail = data[usable:]
        samples: FloatAudio = (
            np.frombuffer(data[:usable], dtype="<i2").astype(np.float32)
            / 32768.0
        )
        return self._project_all(await self._core.receive_audio(samples, now))

    async def _drain_pre_roll(self, now: float) -> list[dict[str, Any]]:
        """Replay everything held, in arrival order, post-admission."""
        wire: list[dict[str, Any]] = []
        for kind, payload in self._pre_roll.release():
            if kind == "audio" and isinstance(payload, bytes):
                wire.extend(await self._feed_audio(payload.decode("ascii"), now))
            elif kind == "update" and isinstance(payload, Mapping):
                wire.extend(
                    self._project_all(await self._core.update(payload, now))
                )
            elif kind == "finalize":
                wire.extend(self._project_all(await self._core.finalize(now)))
        return wire

    def _extension_fields(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """The additive request fields present on a ``session.update``."""
        return {key: event[key] for key in _EXTENSION_FIELDS if key in event}

    def _project_all(self, events: list[Event]) -> list[dict[str, Any]]:
        """Project a core event list in order."""
        return [wire for event in events for wire in self.project(event)]

    def _delta_from(self, cumulative: str) -> str:
        """Incremental delta between consecutive cumulative hypotheses."""
        if cumulative.startswith(self._last_cumulative):
            delta = cumulative[len(self._last_cumulative) :]
        else:
            delta = cumulative
        self._last_cumulative = cumulative
        return delta

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        """The pin's error event shape."""
        return {"type": "error", "error": message, "code": code}
