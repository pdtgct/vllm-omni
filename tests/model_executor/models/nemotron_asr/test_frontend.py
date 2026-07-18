# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential proof for the exact bounded frontend (PORT-FEAT-002).

``frontend.py`` and ``featurizer.py`` import only ``torch`` — loaded
by file path, these tests run locally on macOS. The comparison unit is
``(mel_window, valid_mel_length, next_frontend_state)`` against the
whole-prefix ``MelFeaturizer`` (design §Exact Bounded Frontend), over
arbitrary packet splits, every cadence, exact and partial boundaries,
pre-emphasis discontinuities, silence, and final tails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_PKG = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/nemotron_asr"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"nemotron_asr_{name}_under_test", _PKG / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


frontend = _load("frontend")
featurizer_mod = _load("featurizer")
manifests = _load("manifests")

N_MELS = 16
RAW_TAIL_CAP = 1953


def _featurizer() -> Any:
    torch.manual_seed(3)
    return featurizer_mod.MelFeaturizer(
        filterbank=torch.rand(N_MELS, 257) * 0.01,
        window=torch.hann_window(400),
    )


def _fresh_state(batch: int = 1) -> dict[str, torch.Tensor]:
    return {
        "raw_tail": torch.zeros(batch, RAW_TAIL_CAP),
        "mel_tail": torch.zeros(batch, N_MELS, frontend.MEL_TAIL_FRAMES),
        "counters": torch.zeros(batch, 8, dtype=torch.int64),
    }


def _whole_mel(feat: Any, signal: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mel, mel_len = feat(
            signal.unsqueeze(0), torch.tensor([signal.shape[0]])
        )
    trimmed: torch.Tensor = mel[0, :, : int(mel_len[0])]
    return trimmed


def _stream(
    feat: Any,
    signal: torch.Tensor,
    splits: list[int],
    *,
    final: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Feed ``signal`` through the incremental frontend in ``splits``-
    sized packets (the last one final iff ``final``); concatenate the
    committed frames."""
    state = _fresh_state()
    committed: list[torch.Tensor] = []
    offset = 0
    for i, n in enumerate(splits):
        chunk = signal[offset : offset + n]
        offset += n
        is_final = final and i == len(splits) - 1
        # Arbitrary-packet differentials cap at stability itself (the
        # loosest legal target); cadence tests pass real boundaries.
        target = frontend.stable_frames(offset, n_fft=512, hop=160)
        out, counts = frontend.advance_frontend(
            feat,
            chunk.unsqueeze(0),
            torch.tensor([n], dtype=torch.long),
            torch.tensor([is_final]),
            torch.tensor([target], dtype=torch.long),
            **state,
        )
        if int(counts[0]):
            committed.append(out[0])
    assert offset == signal.shape[0]
    mel = (
        torch.cat(committed, dim=1)
        if committed
        else torch.zeros(N_MELS, 0)
    )
    return mel, state


def _assert_prefix_equal(
    feat: Any, signal: torch.Tensor, streamed: torch.Tensor
) -> None:
    """Streamed frames must equal the same frames of the whole-signal
    featurization — committed frames are FINAL, never recomputed."""
    whole = _whole_mel(feat, signal)
    n = streamed.shape[1]
    assert n <= whole.shape[1]
    torch.testing.assert_close(
        streamed, whole[:, :n], rtol=0, atol=2e-6
    )


CADENCE_SAMPLES = [1280, 2560, 5120, 8960, 17920]


@pytest.mark.parametrize("chunk", CADENCE_SAMPLES)
def test_final_run_matches_whole_signal_exactly(chunk: int) -> None:
    # @spec PORT-FEAT-002
    # Every cadence, four chunks, final tail: the committed stream must
    # equal the whole-signal featurization bit-for-bit, full length.
    feat = _featurizer()
    torch.manual_seed(11)
    signal = torch.randn(4 * chunk) * 0.1
    streamed, state = _stream(feat, signal, [chunk] * 4, final=True)
    whole = _whole_mel(feat, signal)
    assert streamed.shape == whole.shape
    torch.testing.assert_close(streamed, whole, rtol=0, atol=2e-6)
    assert int(state["counters"][0, frontend.CTR_FINALIZED]) == 1


def test_nonfinal_commits_are_a_stable_prefix() -> None:
    # @spec PORT-FEAT-002
    # Without a final tail, commits are exactly the stable frames — a
    # strict prefix of the whole-signal result, never recomputed later.
    feat = _featurizer()
    torch.manual_seed(12)
    signal = torch.randn(3 * 17920) * 0.1
    streamed, state = _stream(feat, signal, [17920] * 3, final=False)
    _assert_prefix_equal(feat, signal, streamed)
    n = signal.shape[0]
    expected = frontend.stable_frames(n, n_fft=512, hop=160)
    assert streamed.shape[1] == expected
    ctr = state["counters"][0]
    assert int(ctr[frontend.CTR_COMMITTED_MEL_FRAMES]) == expected
    assert int(ctr[frontend.CTR_TOTAL_VALID_SAMPLES]) == n
    assert int(ctr[frontend.CTR_FINALIZED]) == 0


def test_arbitrary_packet_splits_match_cadence_splits() -> None:
    # @spec PORT-FEAT-002
    # Transport packet boundaries must not affect the committed frames:
    # ragged splits vs one whole-final call, identical output.
    feat = _featurizer()
    torch.manual_seed(13)
    signal = torch.randn(20_000) * 0.1
    ragged = [1, 159, 160, 4096, 2, 7000, 8582]
    assert sum(ragged) == 20_000
    streamed_r, _ = _stream(feat, signal, ragged, final=True)
    streamed_w, _ = _stream(feat, signal, [20_000], final=True)
    # Frames whose STFT ran in different-shaped calls (a 1-frame FFT
    # vs a batched one) differ by FFT-plan ulps that log() surfaces at
    # quiet bins (~2e-7 quiet, ~1.4e-6 loud bins on cu130 torch); the few-ulp
    # bound is the honest bar for cross-shape comparisons.
    torch.testing.assert_close(streamed_r, streamed_w, rtol=0, atol=2e-6)
    torch.testing.assert_close(
        streamed_r, _whole_mel(feat, signal), rtol=0, atol=2e-6
    )


def test_preemphasis_continuity_across_a_boundary() -> None:
    # @spec PORT-FEAT-002
    # A hard discontinuity exactly at a packet boundary: the
    # cross-boundary x[t-1] term must come from the retained tail.
    feat = _featurizer()
    signal = torch.cat([torch.full((8960,), 0.5), torch.full((8960,), -0.5)])
    streamed, _ = _stream(feat, signal, [8960, 8960], final=True)
    torch.testing.assert_close(
        streamed, _whole_mel(feat, signal), rtol=0, atol=2e-6
    )


def test_silence_matches() -> None:
    # @spec PORT-FEAT-002
    feat = _featurizer()
    signal = torch.zeros(2 * 17920)
    streamed, _ = _stream(feat, signal, [17920, 17920], final=True)
    torch.testing.assert_close(
        streamed, _whole_mel(feat, signal), rtol=0, atol=2e-6
    )


def test_partial_final_tail_off_the_hop_grid() -> None:
    # @spec PORT-FEAT-002
    # A final residual that is not a multiple of the hop: committed
    # length must equal output_lengths (floor), matching whole-signal.
    feat = _featurizer()
    torch.manual_seed(14)
    n = 17920 + 4321
    signal = torch.randn(n) * 0.1
    streamed, _ = _stream(feat, signal, [17920, 4321], final=True)
    whole = _whole_mel(feat, signal)
    assert streamed.shape == whole.shape == (N_MELS, n // 160)
    torch.testing.assert_close(streamed, whole, rtol=0, atol=2e-6)


def test_zero_sample_final_tail_finalizes() -> None:
    # @spec PORT-FEAT-002
    # A final tail carrying zero samples still atomically finalizes.
    feat = _featurizer()
    torch.manual_seed(15)
    signal = torch.randn(17920) * 0.1
    streamed, state = _stream(feat, signal, [17920, 0], final=True)
    # The lone catch-up frame runs in a 1-frame STFT call — cross-shape
    # FFT ulps only (see the ragged-splits test).
    torch.testing.assert_close(
        streamed, _whole_mel(feat, signal), rtol=0, atol=2e-6
    )
    assert int(state["counters"][0, frontend.CTR_FINALIZED]) == 1


def test_chunk_after_finalization_is_a_protocol_error() -> None:
    # @spec PORT-FEAT-002
    feat = _featurizer()
    state = _fresh_state()
    args = (
        torch.zeros(1, 160),
        torch.tensor([160], dtype=torch.long),
        torch.tensor([True]),
        torch.zeros(1, dtype=torch.long),  # target ignored on final
    )
    frontend.advance_frontend(feat, *args, **state)
    with pytest.raises(ValueError, match="finaliz"):
        frontend.advance_frontend(feat, *args, **state)


def test_mel_tail_holds_the_frames_before_the_boundary() -> None:
    # @spec PORT-FEAT-002
    # The mel tail must hold the MEL_TAIL_FRAMES committed frames
    # preceding the boundary — the encoder's pre-encode cache source.
    feat = _featurizer()
    torch.manual_seed(16)
    signal = torch.randn(2 * 17920) * 0.1
    streamed, state = _stream(feat, signal, [17920, 17920], final=False)
    k = frontend.MEL_TAIL_FRAMES
    torch.testing.assert_close(
        state["mel_tail"][0], streamed[:, -k:], rtol=0, atol=2e-6
    )
    assert int(
        state["counters"][0, frontend.CTR_MEL_TAIL_LENGTH]
    ) == k


LOOKAHEADS = {1280: 0, 2560: 1, 5120: 3, 8960: 6, 17920: 13}


@pytest.mark.parametrize(("chunk", "la"), sorted(LOOKAHEADS.items()))
def test_cadence_boundary_margin_is_six(chunk: int, la: int) -> None:
    # @spec PORT-FEAT-002
    # The option-(d) premise (ledger 2026-07-17): after k regular
    # cadence units, stable frames exceed the approved encoder
    # boundary B_k = (8L+1)+(k-1)C = kC-7 by EXACTLY six, uniformly.
    for k in (1, 2, 3, 7):
        stable = frontend.stable_frames(k * chunk, n_fft=512, hop=160)
        boundary = frontend.cadence_boundary(k, lookahead=la)
        assert boundary == k * 8 * (la + 1) - 7
        assert stable - boundary == 6


def test_cadence_capped_run_commits_exactly_the_boundaries() -> None:
    # @spec PORT-FEAT-002
    # Regular 1120 ms units under real cadence targets: commits land
    # exactly on B_k each update, remain a prefix of the whole-signal
    # featurization, and the final tail flushes the residual so the
    # full stream equals the whole signal.
    feat = _featurizer()
    torch.manual_seed(18)
    chunk, la = 17920, 13
    signal = torch.randn(3 * chunk) * 0.1
    state = _fresh_state()
    committed: list[torch.Tensor] = []
    for k in (1, 2, 3):
        out, counts = frontend.advance_frontend(
            feat,
            signal[(k - 1) * chunk : k * chunk].unsqueeze(0),
            torch.tensor([chunk], dtype=torch.long),
            torch.tensor([False]),
            torch.tensor(
                [frontend.cadence_boundary(k, lookahead=la)],
                dtype=torch.long,
            ),
            **state,
        )
        committed.append(out[0])
        assert int(
            state["counters"][0, frontend.CTR_COMMITTED_MEL_FRAMES]
        ) == frontend.cadence_boundary(k, lookahead=la)
    streamed = torch.cat(committed, dim=1)
    _assert_prefix_equal(feat, signal, streamed)
    # The mel tail sits relative to the CAPPED boundary.
    k9 = frontend.MEL_TAIL_FRAMES
    torch.testing.assert_close(
        state["mel_tail"][0], streamed[:, -k9:], rtol=0, atol=2e-6
    )
    out, _ = frontend.advance_frontend(
        feat,
        torch.zeros(1, 0),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([True]),
        torch.zeros(1, dtype=torch.long),
        **state,
    )
    full = torch.cat([streamed, out[0]], dim=1)
    whole = _whole_mel(feat, signal)
    assert full.shape == whole.shape
    torch.testing.assert_close(full, whole, rtol=0, atol=2e-6)


def test_target_past_stability_is_a_margin_violation() -> None:
    # @spec PORT-FEAT-002
    feat = _featurizer()
    state = _fresh_state()
    with pytest.raises(ValueError, match="margin"):
        frontend.advance_frontend(
            feat,
            torch.zeros(1, 1600),
            torch.tensor([1600], dtype=torch.long),
            torch.tensor([False]),
            torch.tensor([99], dtype=torch.long),  # stable is only 9
            **state,
        )


def test_target_below_committed_is_rejected() -> None:
    # @spec PORT-FEAT-002
    feat = _featurizer()
    torch.manual_seed(19)
    signal = torch.randn(17920) * 0.1
    _, state = _stream(feat, signal, [17920], final=False)
    with pytest.raises(ValueError, match="below the committed"):
        frontend.advance_frontend(
            feat,
            torch.zeros(1, 160),
            torch.tensor([160], dtype=torch.long),
            torch.tensor([False]),
            torch.tensor([1], dtype=torch.long),
            **state,
        )


def test_counter_slots_mirror_the_manifest_order() -> None:
    # @spec PORT-STATE-001
    order = [
        ("CTR_TOTAL_VALID_SAMPLES", "total_valid_samples"),
        ("CTR_COMMITTED_MEL_FRAMES", "committed_mel_frames"),
        ("CTR_ENCODED_MEL_FRAMES", "encoded_mel_frames"),
        ("CTR_RAW_TAIL_ORIGIN", "raw_tail_origin"),
        ("CTR_RAW_TAIL_LENGTH", "raw_tail_length"),
        ("CTR_MEL_TAIL_LENGTH", "mel_tail_length"),
        ("CTR_EXPECTED_CHUNK_SEQUENCE", "expected_chunk_sequence"),
        ("CTR_FINALIZED", "finalized"),
    ]
    for const, name in order:
        assert (
            manifests.FRONTEND_COUNTER_FIELDS[getattr(frontend, const)]
            == name
        )


def test_batched_rows_with_shared_geometry_match_single_rows() -> None:
    # @spec PORT-FEAT-002
    # Two rows advanced together must equal each row advanced alone
    # (the grouped-STFT batching is behavior-invariant).
    feat = _featurizer()
    torch.manual_seed(17)
    a = torch.randn(17920) * 0.1
    b = torch.randn(17920) * 0.2
    state2 = _fresh_state(batch=2)
    out2, counts2 = frontend.advance_frontend(
        feat,
        torch.stack([a, b]),
        torch.tensor([17920, 17920], dtype=torch.long),
        torch.tensor([False, False]),
        torch.tensor(
            [frontend.stable_frames(17920, n_fft=512, hop=160)] * 2,
            dtype=torch.long,
        ),
        **state2,
    )
    for row, signal in ((0, a), (1, b)):
        streamed, state1 = _stream(feat, signal, [17920], final=False)
        assert int(counts2[row]) == streamed.shape[1]
        torch.testing.assert_close(out2[row], streamed, rtol=0, atol=2e-6)
        for key in ("raw_tail", "mel_tail", "counters"):
            torch.testing.assert_close(
                state2[key][row], state1[key][0], rtol=0, atol=2e-6
            )
