"""The downstream audio front-end: one conversion point, every adapter.

Every downstream adapter runs client audio through this module before
the session core; the core and everything behind it sees only 16 kHz
mono float32 (ingress-design.md §Audio front-end). The accepted input
set is the explicit four-cell **accept-matrix** — (LINEAR_PCM, 16 kHz),
(LINEAR_PCM, 8 kHz), (MULAW, 8 kHz), (ALAW, 8 kHz), mono — anything
else is ``unsupported_format`` naming both fields (ING-FE-001); G.711
is intrinsically 8 kHz, so a declared 16 kHz G.711 stream is a client
error, never resampled into silent garbage.

Decoding follows the public ITU-T G.711 tables (the CCITT reference
expansion to 16-bit linear, as shipped in every public codec since the
Sun reference code): 256-entry lookups, pure numpy. LINEAR_PCM is
little-endian int16, matching the upstream realtime decode this package
interoperates with. Message framing never aligns with sample
boundaries, so the front-end carries a byte-level residual across
messages and decodes complete samples only (ING-FE-003).

8 kHz input is resampled to 16 kHz by the pinned resampler — the named
value ``scipy-poly-v1``: output identical to
``scipy.signal.resample_poly(whole_signal, up=2, down=1)`` (polyphase
at the exact integer ratio, scipy's default kaiser window) regardless
of how the stream was split into messages. Resampling shifts
waveforms, so the resampler identity is part of any result's
provenance (ING-FE-004; EVAL-ART-001 carries the conditional
``resampler_identifier`` field).
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, lfilter

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.core import FloatAudio
from nemotron_asr_ingress.events import SessionError

#: The pinned resampler's provenance identifier (ING-FE-004).
RESAMPLER_ID = "scipy-poly-v1"

#: The four accepted (encoding, sample_rate_hz) cells, mono only
#: (ING-FE-001). Encodings use the session-core vocabulary — the LLD
#: accept-matrix names; dialect adapters map their own enums onto
#: these (gRPC ``AudioEncoding``, NIM ``g711_ulaw``/``g711_alaw``).
ACCEPT_MATRIX: frozenset[tuple[str, int]] = frozenset(
    {
        ("LINEAR_PCM", 16000),
        ("LINEAR_PCM", 8000),
        ("MULAW", 8000),
        ("ALAW", 8000),
    }
)


# @spec ING-FE-001
def validate_format(
    encoding: str, sample_rate_hz: int, channels: int = 1
) -> SessionError | None:
    """Check one declared format against the accept-matrix.

    Returns ``None`` for an accepted cell, else the
    ``unsupported_format`` catalog error naming both the encoding and
    sample-rate fields (a channel count other than 1 is named too —
    the matrix is mono by definition).
    """
    fields: list[str] = []
    if (encoding, sample_rate_hz) not in ACCEPT_MATRIX:
        fields += ["encoding", "sample_rate_hz"]
    if channels != 1:
        fields.append("channels")
    if not fields:
        return None
    return SessionError(
        code=errors.UNSUPPORTED_FORMAT,
        fields=tuple(fields),
        detail=(
            f"({encoding}, {sample_rate_hz} Hz, {channels} ch) is outside "
            "the accept-matrix {(LINEAR_PCM, 16000), (LINEAR_PCM, 8000), "
            "(MULAW, 8000), (ALAW, 8000)}, mono"
        ),
    )


@dataclass(frozen=True)
class RiffFormat:
    """Format facts a RIFF/WAVE header resolves (ING-GRPC-007).

    ``data_offset`` is where the audio payload starts — header bytes
    are stripped, never decoded as samples. Validation against the
    accept-matrix stays :func:`validate_format`'s job; this is only
    the header's testimony.
    """

    encoding: str
    sample_rate_hz: int
    channels: int
    data_offset: int


#: RIFF/WAVE ``fmt`` format codes -> accept-matrix encodings (the
#: public WAVE format registry: 1 = PCM, 6 = A-law, 7 = μ-law).
_WAV_FORMAT_CODES: dict[int, str] = {1: "LINEAR_PCM", 6: "ALAW", 7: "MULAW"}

#: Bits-per-sample each decodable format code must declare.
_WAV_FORMAT_BITS: dict[str, int] = {"LINEAR_PCM": 16, "MULAW": 8, "ALAW": 8}


def _no_header_error() -> SessionError:
    """The ING-GRPC-007 rejection for an unresolvable stream head."""
    return SessionError(
        code=errors.UNSUPPORTED_FORMAT,
        fields=("encoding", "sample_rate_hz"),
        detail=(
            "no declared format and no parseable RIFF/WAVE header at "
            "the head of the audio stream"
        ),
    )


# @spec ING-GRPC-007
def sniff_riff(data: bytes) -> RiffFormat | SessionError | None:
    """Resolve a format from a RIFF/WAVE header at the stream head.

    The canonical ``python-clients`` tools declare no format and
    stream raw audio-file bytes, header included; an ``UNSPECIFIED``
    encoding therefore defers to this sniff (ING-GRPC-007). Returns
    the resolved :class:`RiffFormat`, ``None`` when more bytes are
    needed to decide, or the ``unsupported_format`` catalog error when
    the head is not a parseable WAVE header of a decodable format.
    """
    if data[: min(4, len(data))] != b"RIFF"[: min(4, len(data))]:
        return _no_header_error()
    if len(data) < 12:
        return None
    if data[8:12] != b"WAVE":
        return _no_header_error()
    pos = 12
    fmt: tuple[int, int, int] | None = None
    bits = 0
    while True:
        if len(data) < pos + 8:
            return None
        chunk_id = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        if chunk_id == b"fmt ":
            if size < 16:
                return _no_header_error()
            if len(data) < pos + 24:
                return None
            body = data[pos + 8 : pos + 24]
            code = int.from_bytes(body[0:2], "little")
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            fmt = (code, channels, rate)
            pos += 8 + size + (size & 1)
        elif chunk_id == b"data":
            if fmt is None:
                return _no_header_error()
            code, channels, rate = fmt
            encoding = _WAV_FORMAT_CODES.get(code)
            if encoding is None or bits != _WAV_FORMAT_BITS[encoding]:
                return SessionError(
                    code=errors.UNSUPPORTED_FORMAT,
                    fields=("encoding", "sample_rate_hz"),
                    detail=(
                        f"WAV format code {code} at {bits}-bit is not a "
                        "decodable accept-matrix cell"
                    ),
                )
            return RiffFormat(
                encoding=encoding,
                sample_rate_hz=rate,
                channels=channels,
                data_offset=pos + 8,
            )
        else:
            pos += 8 + size + (size & 1)


# @spec ING-FE-002
def mulaw_table() -> FloatAudio:
    """The 256-entry ITU G.711 μ-law -> float32 lookup table.

    Entries are the CCITT 16-bit linear expansions scaled by 1/32768
    (μ-law full scale is ±32124).
    """
    u = np.arange(256, dtype=np.int32) ^ 0xFF  # transmitted complemented
    t = ((u & 0x0F) << 3) + 0x84
    t = t << ((u & 0x70) >> 4)
    linear = np.where(u & 0x80, 0x84 - t, t - 0x84)
    return np.asarray(linear / 32768.0, dtype=np.float32)


# @spec ING-FE-002
def alaw_table() -> FloatAudio:
    """The 256-entry ITU G.711 A-law -> float32 lookup table.

    Entries are the CCITT 16-bit linear expansions scaled by 1/32768
    (A-law full scale is ±32256).
    """
    a = np.arange(256, dtype=np.int32) ^ 0x55  # even bits inverted
    seg = (a & 0x70) >> 4
    t = (a & 0x0F) << 4
    t = np.where(seg == 0, t + 8, (t + 0x108) << np.maximum(seg - 1, 0))
    linear = np.where(a & 0x80, t, -t)  # A-law sign bit set is positive
    return np.asarray(linear / 32768.0, dtype=np.float32)


class _StreamingResampler:
    """Stateful 8 kHz -> 16 kHz polyphase FIR, ``scipy-poly-v1`` exact.

    ``resample_poly(x, 2, 1)`` is the convolution of the zero-stuffed
    input with scipy's kaiser-windowed filter, trimmed by the filter's
    half-length; carrying the convolution state in ``lfilter`` makes
    the streamed output identical to the whole-signal call no matter
    how the stream was split. ``flush`` drains the half-length tail
    the lookahead still holds.
    """

    #: scipy's resample_poly design for up=2, down=1: half_len = 20,
    #: h = firwin(2*20 + 1, 1/2, window=("kaiser", 5.0)) * 2.
    _HALF_LEN = 20

    def __init__(self) -> None:
        self._h = (
            firwin(2 * self._HALF_LEN + 1, 0.5, window=("kaiser", 5.0)) * 2.0
        )
        self._zi = np.zeros(len(self._h) - 1)
        self._to_skip = self._HALF_LEN

    def push(self, samples: FloatAudio) -> FloatAudio:
        """Advance the filter over one decoded 8 kHz piece."""
        if not len(samples):
            return np.zeros(0, dtype=np.float32)
        stuffed = np.zeros(2 * len(samples))
        stuffed[0::2] = samples
        out, self._zi = lfilter(self._h, 1.0, stuffed, zi=self._zi)
        return self._emit(out)

    def flush(self) -> FloatAudio:
        """Drain the lookahead tail (exactly the filter half-length)."""
        out, self._zi = lfilter(
            self._h, 1.0, np.zeros(self._HALF_LEN), zi=self._zi
        )
        return self._emit(out)

    def _emit(self, out: FloatAudio) -> FloatAudio:
        """Drop the initial half-length delay, once, then pass through."""
        if self._to_skip:
            skip = min(self._to_skip, len(out))
            self._to_skip -= skip
            out = out[skip:]
        return np.asarray(out, dtype=np.float32)


# @spec ING-FE-002, ING-FE-003, ING-FE-004
class AudioFrontEnd:
    """One session's decode path: raw client bytes -> 16 kHz float32.

    Stateful on purpose: a byte-level residual carries across ``feed``
    calls so a sample straddling two client messages is never decoded
    as two fragments (ING-FE-003), and the 8 kHz path carries
    resampler state so the streamed output equals the whole-signal
    ``scipy-poly-v1`` result. ``flush`` drains whatever the resampler
    still holds at finalize; a trailing incomplete sample's bytes are
    dropped (they can never become a sample).
    """

    def __init__(self, encoding: str, sample_rate_hz: int) -> None:
        """Bind one accepted matrix cell.

        Raises:
            ValueError: For a cell outside the accept-matrix — adapters
                validate first via :func:`validate_format`; reaching
                here unvalidated is a programming error, not a client
                error.
        """
        if (encoding, sample_rate_hz) not in ACCEPT_MATRIX:
            raise ValueError(
                f"({encoding}, {sample_rate_hz}) is outside the "
                "accept-matrix; adapters must validate_format() first"
            )
        self._bytes_per_sample = 2 if encoding == "LINEAR_PCM" else 1
        self._table: FloatAudio | None = None
        if encoding == "MULAW":
            self._table = mulaw_table()
        elif encoding == "ALAW":
            self._table = alaw_table()
        self._resampler = (
            _StreamingResampler() if sample_rate_hz == 8000 else None
        )
        self._byte_tail = b""

    @property
    def resampler(self) -> str | None:
        """The provenance identifier the resampled path stamps.

        ``scipy-poly-v1`` at 8 kHz, ``None`` at 16 kHz (ING-FE-004 —
        recorded in run provenance for any resampled result).
        """
        return RESAMPLER_ID if self._resampler is not None else None

    def feed(self, raw: bytes) -> FloatAudio:
        """Decode one client message's bytes; return 16 kHz float32.

        Only complete samples decode; a trailing partial sample's
        bytes wait for the next message (ING-FE-003). At 8 kHz the
        output lags input by the resampler's lookahead — ``flush``
        emits the tail.
        """
        data = self._byte_tail + raw
        usable = len(data) - (len(data) % self._bytes_per_sample)
        self._byte_tail = data[usable:]
        if self._table is not None:
            samples = self._table[np.frombuffer(data[:usable], dtype=np.uint8)]
        else:
            samples = np.asarray(
                np.frombuffer(data[:usable], dtype="<i2").astype(np.float32)
                / 32768.0,
                dtype=np.float32,
            )
        if self._resampler is None:
            return samples
        return self._resampler.push(samples)

    def flush(self) -> FloatAudio:
        """Drain the resampler tail at finalize (empty at 16 kHz)."""
        if self._resampler is None:
            return np.zeros(0, dtype=np.float32)
        return self._resampler.flush()
