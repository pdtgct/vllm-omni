"""Shared fakes for the ING-1 session-core suite (GPU-free tier).

The fake transcriber records exactly what the session core hands the
PORT seam — chunk sizes, the raw residual, locale forwards, aborts —
independent of any model code, so the contract tests never validate
the code with the code.
"""

import struct
from typing import Any

import numpy as np
import numpy.typing as npt

from nemotron_asr_ingress.core import (
    IngressValues,
    InProcessGate,
    SessionCore,
)

CHUNK_MS = 80
CHUNK_SAMPLES = CHUNK_MS * 16  # 16 kHz

WORDS = ("hey", "there", "friend", "how", "are", "you")

PROVENANCE: dict[str, Any] = {
    "precision_policy_id": "pp-d479361445b4",
    "fingerprint": {"device_identity": "NVIDIA L4"},
}


class FakeTranscriber:
    """Records the PORT-seam calls; returns scripted hypotheses."""

    def __init__(self) -> None:
        self.steps: list[npt.NDArray[np.float32]] = []
        self.flushed: npt.NDArray[np.float32] | None = None
        self.flush_called = False
        self.locales: list[str] = []
        self.aborted = False

    def step(self, chunk: npt.NDArray[np.float32]) -> str:
        self.steps.append(chunk)
        return " ".join(WORDS[: min(len(self.steps), len(WORDS))])

    def flush(self, residual: npt.NDArray[np.float32]) -> str:
        self.flush_called = True
        self.flushed = residual
        return self.cumulative() + " [flushed]"

    def update_locale(self, target_lang: str) -> None:
        self.locales.append(target_lang)

    def abort(self) -> None:
        self.aborted = True

    def cumulative(self) -> str:
        return " ".join(WORDS[: min(len(self.steps), len(WORDS))])


def make_values(**overrides: Any) -> IngressValues:
    """ING values with GPU-free-test defaults."""
    values: dict[str, Any] = {
        "watermark": 2,
        "admission_queue": 0,
        "admission_wait_s": 5.0,
        "chunk_buffer_s": 10.0,
        "pre_roll_bytes": 65536,
        "idle_ttl_s": 60.0,
        "locales": ("en-US", "es-US", "auto"),
    }
    values.update(overrides)
    return IngressValues(**values)


def make_gate(values: IngressValues) -> InProcessGate:
    """The authoritative gate for one server, from the shared values."""
    return InProcessGate(
        watermark=values.watermark,
        queue_len=values.admission_queue,
        wait_s=values.admission_wait_s,
    )


def make_core(
    gate: InProcessGate | None = None,
    values: IngressValues | None = None,
) -> tuple[SessionCore, FakeTranscriber, InProcessGate]:
    """One session core bound to a fake transcriber."""
    vals = values if values is not None else make_values()
    the_gate = gate if gate is not None else make_gate(vals)
    fake = FakeTranscriber()
    core = SessionCore(
        gate=the_gate,
        transcriber=fake,
        values=vals,
        provenance=PROVENANCE,
    )
    return core, fake, the_gate


def audio(n_samples: int, start: int = 0) -> npt.NDArray[np.float32]:
    """Distinct-valued float32 audio so passthrough is assertable."""
    ramp = np.arange(start, start + n_samples, dtype=np.float32)
    return np.asarray(ramp / 32768.0, dtype=np.float32)


def riff_wav(
    format_code: int,
    payload: bytes,
    rate: int,
    bits: int,
    channels: int = 1,
    with_fact: bool = False,
) -> bytes:
    """Hand-built RIFF/WAVE bytes (the wave module writes PCM only).

    Format codes per the public WAVE registry: 1 = PCM, 6 = A-law,
    7 = mu-law. ``with_fact`` inserts the fact chunk non-PCM WAVs
    customarily carry, so chunk-walking is exercised.
    """
    block = max(1, channels * bits // 8)
    fmt = struct.pack(
        "<HHIIHH", format_code, channels, rate, rate * block, block, bits
    )
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if with_fact:
        body += b"fact" + struct.pack("<I", 4) + struct.pack("<I", len(payload))
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def pcm16_wav(n_samples: int, rate: int = 16000) -> bytes:
    """A mono PCM16 WAV with a distinct-valued ramp payload."""
    return riff_wav(
        1, np.arange(n_samples, dtype="<i2").tobytes(), rate=rate, bits=16
    )
