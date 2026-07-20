"""The shared vLLM-dialect WebSocket session client (ING-CLI-001).

One client, three roles: the remote provider's transport, the
endpoint-tier parity client (EVAL-PAR endpoint tier), and the
upstream-eligible ``benchmarks/asr`` driver. This module is the
client's sans-IO core — folding server wire events into an outcome —
so the GPU-free tier pins the dialect handling; the WebSocket shell
stays pod-side.

The client tolerates-and-ignores every ``response.audio.*`` event:
omni's ``RealtimeConnection`` always sends a terminal
``response.audio.done {has_audio: false}`` even for text-only ASR
models (verified @ ``d4a869fe``).
"""

import asyncio
import base64
import importlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from nemotron_asr_ingress.core import SAMPLE_RATE

_BYTES_PER_SAMPLE = 2


@dataclass
class RealtimeOutcome:
    """Everything one realtime session produced (client view)."""

    session_id: str | None = None
    provenance: Mapping[str, Any] | None = None
    deltas: list[str] = field(default_factory=list)
    final: str | None = None
    errors: list[Mapping[str, Any]] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        """Whether the terminal transcript has arrived."""
        return self.final is not None


# @spec ING-CLI-001
def fold_server_event(
    outcome: RealtimeOutcome, event: Mapping[str, Any]
) -> None:
    """Fold one server wire event into the outcome.

    ``session.created`` carries the id; ``session.admitted`` (β's
    additive event; stock upstream never sends it) carries provenance;
    ``transcription.delta`` accumulates; ``transcription.done`` sets
    the final transcript; ``error`` is recorded; ``response.audio.*``
    is tolerated-and-ignored; any other unknown type is recorded in
    ``ignored`` — noted, never fatal, never silent in the outcome.
    """
    event_type = str(event.get("type", ""))
    if event_type == "session.created":
        session_id = event.get("id")
        outcome.session_id = str(session_id) if session_id is not None else None
    elif event_type == "session.admitted":
        provenance = event.get("provenance")
        if isinstance(provenance, Mapping):
            outcome.provenance = provenance
    elif event_type == "transcription.delta":
        outcome.deltas.append(str(event.get("delta", "")))
    elif event_type == "transcription.done":
        outcome.final = str(event.get("text", ""))
    elif event_type == "error":
        outcome.errors.append(event)
    elif event_type.startswith("response.audio"):
        pass
    else:
        outcome.ignored.append(event_type)


class LatencyRecorder:
    """Chunk-to-partial latency at the client edge.

    The gap between a ``transcription.delta``'s arrival and the wall
    time the most recent audio frame was sent — transport + queue +
    batch + compute, the number the NIM baseline is measured by
    (EVAL-HARN-005: the load/latency driver is this client). Wraps
    :func:`fold_server_event`; time is an explicit ``now`` so the
    GPU-free tier pins the bookkeeping.
    """

    def __init__(self, outcome: RealtimeOutcome) -> None:
        """Record into (and fold onto) one session outcome."""
        self._outcome = outcome
        self._last_send: float | None = None
        self._gaps: list[float] = []

    def note_audio_sent(self, now: float) -> None:
        """Mark the wall time an audio frame went out."""
        self._last_send = now

    def fold(self, event: Mapping[str, Any], now: float) -> None:
        """Fold one server event; a delta records its arrival gap."""
        fold_server_event(self._outcome, event)
        if (
            event.get("type") == "transcription.delta"
            and self._last_send is not None
        ):
            self._gaps.append(now - self._last_send)

    @property
    def gaps(self) -> list[float]:
        """Chunk-to-partial gaps, in delta-arrival order."""
        return list(self._gaps)


@dataclass
class RealtimeRun:
    """One driven realtime session: outcome plus sweep measurements."""

    outcome: RealtimeOutcome
    latencies_ms: list[float] = field(default_factory=list)
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    gpu_mem_mb: int = 0

    @property
    def final(self) -> str | None:
        """The terminal transcript (client view)."""
        return self.outcome.final

    @property
    def deltas(self) -> list[str]:
        """Incremental deltas, in arrival order."""
        return self.outcome.deltas

    @property
    def errors(self) -> list[Mapping[str, Any]]:
        """Error events, in arrival order."""
        return self.outcome.errors

    @property
    def provenance(self) -> Mapping[str, Any] | None:
        """`session.admitted` provenance (β; stock upstream sends none)."""
        return self.outcome.provenance


async def run_realtime_session(
    server: str,
    pcm16: bytes,
    *,
    model: str,
    chunk_ms: int = 560,
    target_lang: str = "en-US",
    pace_realtime: bool = True,
    recv_timeout_s: float = 120.0,
) -> RealtimeRun:
    """Stream one clip through one `/v1/realtime` session.

    The one WebSocket driver (ING-CLI-001): the sweep/load driver, the
    endpoint-tier parity client, and the remote provider's transport —
    identical against β and stock upstream/omni. ``websockets`` is
    imported lazily; this module stays importable GPU-free.

    Args:
        server: ``ws://host:port`` endpoint (the route is appended).
        pcm16: Raw little-endian PCM16 mono audio at 16 kHz.
        model: The served model name (`session.update` requires it).
        chunk_ms: Admitted chunk size; also the send-frame size.
        target_lang: Admitted locale (a β `session.update` extension;
            stock upstream ignores it and runs `auto`).
        pace_realtime: Sleep each frame's audio duration after sending
            (the like-for-like pacing the NIM sweep uses).
        recv_timeout_s: Guard on the terminal event after commit.

    Returns:
        The run; a rejected admission returns with the error recorded
        and no final.
    """
    websockets = importlib.import_module("websockets")
    outcome = RealtimeOutcome()
    recorder = LatencyRecorder(outcome)
    run = RealtimeRun(outcome=outcome)
    frame_bytes = chunk_ms * (SAMPLE_RATE // 1000) * _BYTES_PER_SAMPLE
    run.audio_seconds = len(pcm16) / _BYTES_PER_SAMPLE / SAMPLE_RATE
    url = server.rstrip("/") + "/v1/realtime"
    async with websockets.connect(url, max_size=None) as ws:
        start = time.monotonic()

        async def _receive() -> None:
            async for raw in ws:
                event = json.loads(raw)
                recorder.fold(event, time.monotonic())
                if event.get("type") == "transcription.done":
                    run.gpu_mem_mb = int(event.get("gpu_mem_mb", 0))
                    return

        receiver = asyncio.create_task(_receive())
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "model": model,
                    "chunk_ms": chunk_ms,
                    "target_lang": target_lang,
                }
            )
        )
        try:
            for offset in range(0, len(pcm16), frame_bytes):
                if receiver.done():
                    break  # a fatal answer (busy, overflow) ended us
                frame = pcm16[offset : offset + frame_bytes]
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(frame).decode("ascii"),
                        }
                    )
                )
                recorder.note_audio_sent(time.monotonic())
                if pace_realtime:
                    await asyncio.sleep(
                        len(frame) / _BYTES_PER_SAMPLE / SAMPLE_RATE
                    )
            if not receiver.done():
                await ws.send(
                    json.dumps(
                        {"type": "input_audio_buffer.commit", "final": True}
                    )
                )
        except websockets.ConnectionClosed:
            pass  # the receiver saw (or will see) the server's last word
        await asyncio.wait_for(receiver, timeout=recv_timeout_s)
        run.wall_seconds = time.monotonic() - start
    run.latencies_ms = [gap * 1000.0 for gap in recorder.gaps]
    return run
