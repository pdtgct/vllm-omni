"""Audio front-end: accept-matrix, G.711, residual, resampler (ING-FE-001..004).

G.711 anchor values are the ITU-T G.711 16-bit linear expansions from
the public CCITT reference tables (μ-law full scale ±32124, A-law full
scale ±32256) — decoded floats are those integers / 32768.
"""

import itertools

import numpy as np
import pytest
from ingress_helpers import audio, pcm16_wav, riff_wav
from scipy.signal import resample_poly

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.events import SessionError
from nemotron_asr_ingress.frontend import (
    ACCEPT_MATRIX,
    RESAMPLER_ID,
    AudioFrontEnd,
    RiffFormat,
    alaw_table,
    mulaw_table,
    sniff_riff,
    validate_format,
)


def pcm16_bytes(n_samples: int, start: int = 0) -> bytes:
    """Little-endian int16 ramp; distinct values so order is assertable."""
    return (
        np.arange(start, start + n_samples, dtype=np.int16)
        .astype("<i2")
        .tobytes()
    )


# ---- ING-FE-001: the accept-matrix ----------------------------------------


# @spec ING-FE-001
def test_accept_matrix_is_exactly_the_four_cells() -> None:
    assert (
        frozenset(
            {
                ("LINEAR_PCM", 16000),
                ("LINEAR_PCM", 8000),
                ("MULAW", 8000),
                ("ALAW", 8000),
            }
        )
        == ACCEPT_MATRIX
    )


# @spec ING-FE-001
@pytest.mark.parametrize(("encoding", "rate"), sorted(ACCEPT_MATRIX))
def test_matrix_cells_validate_clean(encoding: str, rate: int) -> None:
    assert validate_format(encoding, rate) is None


# @spec ING-FE-001
@pytest.mark.parametrize(
    ("encoding", "rate"),
    [
        ("MULAW", 16000),  # G.711 is intrinsically 8 kHz: a client error
        ("ALAW", 16000),
        ("LINEAR_PCM", 44100),
        ("LINEAR_PCM", 48000),
        ("FLAC", 16000),  # codec streams rejected explicitly (LLD)
        ("OGGOPUS", 16000),
        ("ENCODING_UNSPECIFIED", 16000),
    ],
)
def test_off_matrix_cells_name_both_fields(encoding: str, rate: int) -> None:
    error = validate_format(encoding, rate)
    assert error is not None
    assert error.code == errors.UNSUPPORTED_FORMAT
    assert "encoding" in error.fields
    assert any("sample_rate" in field for field in error.fields)


# @spec ING-FE-001
def test_multichannel_is_off_matrix() -> None:
    error = validate_format("LINEAR_PCM", 16000, channels=2)
    assert error is not None
    assert error.code == errors.UNSUPPORTED_FORMAT
    assert any("channel" in field for field in error.fields)


def test_front_end_refuses_an_unvalidated_cell() -> None:
    # Adapters validate first; construction outside the matrix is a
    # programming error, not a client error.
    with pytest.raises(ValueError):
        AudioFrontEnd("MULAW", 16000)


# ---- ING-FE-002: ITU tables + little-endian PCM ----------------------------


# @spec ING-FE-002
def test_mulaw_table_matches_the_itu_anchors() -> None:
    table = mulaw_table()
    assert table.shape == (256,)
    assert table.dtype == np.float32
    assert table[0x00] == pytest.approx(-32124 / 32768)
    assert table[0x80] == pytest.approx(+32124 / 32768)
    assert table[0xFF] == 0.0
    assert table[0x7F] == 0.0


# @spec ING-FE-002
def test_alaw_table_matches_the_itu_anchors() -> None:
    table = alaw_table()
    assert table.shape == (256,)
    assert table.dtype == np.float32
    assert table[0x55] == pytest.approx(-8 / 32768)
    assert table[0xD5] == pytest.approx(+8 / 32768)
    assert table[0x2A] == pytest.approx(-32256 / 32768)
    assert table[0xAA] == pytest.approx(+32256 / 32768)


# @spec ING-FE-002
@pytest.mark.parametrize("table_fn", [mulaw_table, alaw_table])
def test_g711_sign_bit_negates(table_fn: object) -> None:
    # In both laws the sign bit flips polarity at equal magnitude.
    table = table_fn()  # type: ignore[operator]
    flipped = table[np.arange(256) ^ 0x80]
    np.testing.assert_allclose(flipped, -table, atol=1e-9)


# @spec ING-FE-002
def test_linear_pcm_is_little_endian_int16() -> None:
    front = AudioFrontEnd("LINEAR_PCM", 16000)
    out = front.feed(b"\x01\x02")  # LE: 0x0201 == 513
    np.testing.assert_allclose(out, [513 / 32768], atol=1e-9)
    assert out.dtype == np.float32


# @spec ING-FE-002
def test_pcm16_decode_matches_the_source_samples() -> None:
    front = AudioFrontEnd("LINEAR_PCM", 16000)
    out = front.feed(pcm16_bytes(200, start=-100))
    expected = np.arange(-100, 100, dtype=np.float32) / 32768.0
    np.testing.assert_allclose(out, expected, atol=1e-9)


# ---- ING-FE-003: byte-level residual across messages -----------------------


# @spec ING-FE-003
def test_a_sample_straddling_two_messages_decodes_whole() -> None:
    whole = pcm16_bytes(6)
    front = AudioFrontEnd("LINEAR_PCM", 16000)
    split = 5  # mid-sample: an odd byte boundary
    first = front.feed(whole[:split])
    second = front.feed(whole[split:])
    reference = AudioFrontEnd("LINEAR_PCM", 16000).feed(whole)
    np.testing.assert_allclose(
        np.concatenate([first, second]), reference, atol=1e-9
    )


# @spec ING-FE-003
def test_a_lone_byte_waits_for_its_other_half() -> None:
    front = AudioFrontEnd("LINEAR_PCM", 16000)
    assert len(front.feed(b"\x01")) == 0
    out = front.feed(b"\x02")
    np.testing.assert_allclose(out, [513 / 32768], atol=1e-9)


# @spec ING-FE-003
def test_discard_partial_sample_drops_only_the_byte_residual() -> None:
    # The dialect buffer-clear seam (ING-NIMWS-006): the carried
    # partial-sample bytes drop; decode state realigns on the next
    # message's first byte, and nothing already decoded is touched.
    front = AudioFrontEnd("LINEAR_PCM", 16000)
    assert len(front.feed(pcm16_bytes(4) + b"\x01")) == 4
    front.discard_partial_sample()
    out = front.feed(pcm16_bytes(4, start=100))
    expected = np.arange(100, 104, dtype=np.float32) / 32768.0
    np.testing.assert_allclose(out, expected, atol=1e-9)


# @spec ING-FE-003
def test_many_arbitrary_splits_decode_identically() -> None:
    whole = pcm16_bytes(500)
    reference = AudioFrontEnd("LINEAR_PCM", 16000).feed(whole)
    front = AudioFrontEnd("LINEAR_PCM", 16000)
    pieces = []
    cuts = [0, 3, 17, 18, 255, 256, 499, 731, len(whole)]
    for lo, hi in itertools.pairwise(cuts):
        pieces.append(front.feed(whole[lo:hi]))
    np.testing.assert_allclose(np.concatenate(pieces), reference, atol=1e-9)


# ---- ING-FE-004: the pinned resampler --------------------------------------


# @spec ING-FE-004
def test_resampler_identity_is_the_named_value() -> None:
    assert RESAMPLER_ID == "scipy-poly-v1"
    assert AudioFrontEnd("LINEAR_PCM", 8000).resampler == RESAMPLER_ID
    assert AudioFrontEnd("MULAW", 8000).resampler == RESAMPLER_ID
    assert AudioFrontEnd("LINEAR_PCM", 16000).resampler is None


# @spec ING-FE-004
def test_8k_pcm_stream_equals_the_whole_signal_resample() -> None:
    rng = np.random.default_rng(20260712)
    ints = rng.integers(-20000, 20000, size=800, dtype=np.int16)
    whole = ints.astype("<i2").tobytes()
    expected = resample_poly(
        ints.astype(np.float32) / 32768.0, up=2, down=1
    ).astype(np.float32)

    front = AudioFrontEnd("LINEAR_PCM", 8000)
    pieces = []
    cuts = [0, 7, 160, 161, 900, 1599, len(whole)]
    for lo, hi in itertools.pairwise(cuts):
        pieces.append(front.feed(whole[lo:hi]))
    pieces.append(front.flush())
    streamed = np.concatenate(pieces)

    assert len(streamed) == 2 * len(ints)
    np.testing.assert_allclose(streamed, expected, atol=1e-6)


# @spec ING-FE-002, ING-FE-004
def test_mulaw_8k_decodes_through_table_then_resamples() -> None:
    raw = bytes(range(256)) * 3
    table = mulaw_table()
    decoded = table[np.frombuffer(raw, dtype=np.uint8)]
    expected = resample_poly(decoded, up=2, down=1).astype(np.float32)

    front = AudioFrontEnd("MULAW", 8000)
    streamed = np.concatenate([front.feed(raw), front.flush()])
    assert len(streamed) == 2 * len(raw)
    np.testing.assert_allclose(streamed, expected, atol=1e-6)


# @spec ING-FE-002, ING-FE-004
def test_alaw_8k_decodes_through_table_then_resamples() -> None:
    raw = bytes(range(256))
    table = alaw_table()
    decoded = table[np.frombuffer(raw, dtype=np.uint8)]
    expected = resample_poly(decoded, up=2, down=1).astype(np.float32)

    front = AudioFrontEnd("ALAW", 8000)
    streamed = np.concatenate([front.feed(raw), front.flush()])
    assert len(streamed) == 2 * len(raw)
    np.testing.assert_allclose(streamed, expected, atol=1e-6)


# ---- ING-GRPC-007: RIFF/WAVE header resolution ------------------------------


# @spec ING-GRPC-007
def test_sniff_resolves_a_pcm16_wav_header() -> None:
    wav = pcm16_wav(100, rate=16000)
    resolved = sniff_riff(wav)
    assert isinstance(resolved, RiffFormat)
    assert resolved == RiffFormat(
        encoding="LINEAR_PCM", sample_rate_hz=16000, channels=1, data_offset=44
    )
    assert wav[resolved.data_offset :] == pcm16_bytes(100)


# @spec ING-GRPC-007
@pytest.mark.parametrize(
    ("format_code", "encoding"), [(7, "MULAW"), (6, "ALAW")]
)
def test_sniff_resolves_g711_wav_headers(
    format_code: int, encoding: str
) -> None:
    payload = bytes(range(256))
    wav = riff_wav(format_code, payload, rate=8000, bits=8, with_fact=True)
    resolved = sniff_riff(wav)
    assert isinstance(resolved, RiffFormat)
    assert resolved.encoding == encoding
    assert resolved.sample_rate_hz == 8000
    assert resolved.channels == 1
    assert wav[resolved.data_offset :] == payload


# @spec ING-GRPC-007
def test_sniff_asks_for_more_bytes_on_a_truncated_header() -> None:
    wav = pcm16_wav(10)
    for cut in (0, 3, 11, 20, 43):
        assert sniff_riff(wav[:cut]) is None


# @spec ING-GRPC-007
def test_sniff_rejects_a_non_riff_head() -> None:
    rejection = sniff_riff(b"OggS anything else entirely")
    assert isinstance(rejection, SessionError)
    assert rejection.code == errors.UNSUPPORTED_FORMAT
    assert "encoding" in rejection.fields
    # An early mismatch decides without waiting for more bytes.
    assert isinstance(sniff_riff(b"Ogg"), SessionError)


# @spec ING-GRPC-007
def test_sniff_rejects_undecodable_wav_formats() -> None:
    ieee_float = riff_wav(3, b"\x00" * 32, rate=16000, bits=32)
    rejection = sniff_riff(ieee_float)
    assert isinstance(rejection, SessionError)
    assert rejection.code == errors.UNSUPPORTED_FORMAT
    pcm8 = riff_wav(1, b"\x00" * 32, rate=16000, bits=8)
    assert isinstance(sniff_riff(pcm8), SessionError)


# @spec ING-GRPC-007
def test_sniff_reports_stereo_for_the_validator_to_reject() -> None:
    stereo = riff_wav(1, b"\x00" * 64, rate=16000, bits=16, channels=2)
    resolved = sniff_riff(stereo)
    assert isinstance(resolved, RiffFormat)
    assert resolved.channels == 2
    rejection = validate_format(
        resolved.encoding, resolved.sample_rate_hz, resolved.channels
    )
    assert rejection is not None
    assert rejection.code == errors.UNSUPPORTED_FORMAT


# @spec ING-FE-004
def test_16k_path_never_resamples_and_flush_is_empty() -> None:
    source = audio(320)
    raw = (
        (np.asarray(source) * 32768.0).astype(np.int16).astype("<i2").tobytes()
    )
    front = AudioFrontEnd("LINEAR_PCM", 16000)
    out = front.feed(raw)
    assert len(out) == 320  # passthrough length: no rate change
    assert len(front.flush()) == 0
