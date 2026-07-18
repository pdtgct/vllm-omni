# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact-active decode vs the greedy oracle (PORT-DEC-008 /
PORT-PERF-001).

``rnnt.py`` needs only torch + ``rnnt_cell`` — loaded by file path
with a stubbed package chain, these tests run locally on macOS. The
math-tier ``greedy_decode_batch`` is the differential oracle the
compact-active production loop must match exactly: labels, order,
per-row lengths, and the advanced predictor state, including rows
masked by ``enc_lengths`` (padded frames never decode).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch

_PKG = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/nemotron_asr"
)


def _load_rnnt() -> Any:
    """Load rnnt.py by path, pre-seeding its rnnt_cell package import."""
    base = "vllm_omni.model_executor.models.nemotron_asr"
    for name in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        base,
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    cell_spec = importlib.util.spec_from_file_location(
        f"{base}.rnnt_cell", _PKG / "rnnt_cell.py"
    )
    assert cell_spec is not None and cell_spec.loader is not None
    cell = importlib.util.module_from_spec(cell_spec)
    sys.modules[f"{base}.rnnt_cell"] = cell
    cell_spec.loader.exec_module(cell)
    spec = importlib.util.spec_from_file_location(
        f"{base}.rnnt", _PKG / "rnnt.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{base}.rnnt"] = module
    spec.loader.exec_module(module)
    return module


rnnt = _load_rnnt()

VOCAB = 12
ENC = 32
PRED = 16


def _nets(seed: int = 7) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    predictor = rnnt.Predictor(
        vocab_size=VOCAB, pred_hidden=PRED, pred_rnn_layers=2
    )
    joint = rnnt.Joint(
        enc_hidden=ENC, pred_hidden=PRED, joint_hidden=16,
        vocab_size=VOCAB,
    )
    # Damp the random recurrence: untrained weights explode to NaN
    # over cap-saturated 60-step runs, which would poison the exact
    # state comparison.
    with torch.no_grad():
        for module in (predictor, joint):
            for param in module.parameters():
                param.mul_(0.3)
    return predictor, joint


def _state(batch: int) -> Any:
    return rnnt.DecodeState(
        h=torch.zeros(2, batch, PRED),
        c=torch.zeros(2, batch, PRED),
        last_label=torch.full((batch,), VOCAB, dtype=torch.long),
    )


def _oracle_rows(
    predictor: Any,
    joint: Any,
    enc: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[list[list[int]], Any]:
    """Length-aware oracle: run greedy_decode_batch PER ROW on the
    row's valid frames only (the reference loop has no length mask)."""
    bursts: list[list[int]] = []
    hs, cs, lasts = [], [], []
    for b in range(enc.shape[0]):
        state = _state(1)
        labels, out = rnnt.greedy_decode_batch(
            enc[b : b + 1, : int(lengths[b])], predictor, joint, state
        )
        bursts.append(labels[0])
        hs.append(out.h)
        cs.append(out.c)
        lasts.append(out.last_label)
    return bursts, rnnt.DecodeState(
        h=torch.cat(hs, dim=1),
        c=torch.cat(cs, dim=1),
        last_label=torch.cat(lasts),
    )


def _assert_matches_oracle(
    enc: torch.Tensor, lengths: torch.Tensor, *, seed: int = 7
) -> None:
    predictor, joint = _nets(seed)
    with torch.no_grad():
        expected, exp_state = _oracle_rows(predictor, joint, enc, lengths)
        ids, lens, state = rnnt.decode_compact_active(
            enc, lengths, predictor, joint, _state(enc.shape[0])
        )
    for b, burst in enumerate(expected):
        assert int(lens[b]) == len(burst), f"row {b} length"
        assert ids[b, : len(burst)].tolist() == burst, f"row {b} labels"
        assert not bool((ids[b, len(burst):] != 0).any())  # padding
    assert not bool(state.h.isnan().any())  # fixture sanity
    # Labels/lengths/padding are EXACT above; the LSTM state compares
    # at a few-ulp bound — compacted-batch matmul reductions differ
    # from the oracle's per-row batch-1 runs by ~2e-9 (the same
    # cross-shape kernel effect as the frontend's FFT-plan bound).
    torch.testing.assert_close(state.h, exp_state.h, rtol=0, atol=1e-6)
    torch.testing.assert_close(state.c, exp_state.c, rtol=0, atol=1e-6)
    assert state.last_label.tolist() == exp_state.last_label.tolist()


def test_uniform_lengths_match_the_oracle() -> None:
    # @spec PORT-DEC-008
    torch.manual_seed(11)
    enc = torch.randn(3, 6, ENC)
    _assert_matches_oracle(enc, torch.tensor([6, 6, 6]))


def test_padded_rows_never_decode_past_their_length() -> None:
    # @spec PORT-DEC-008 / PORT-PERF-001
    # Mixed lengths incl. a zero-length row: padded frames carry loud
    # garbage that MUST NOT reach the joint.
    torch.manual_seed(12)
    enc = torch.randn(4, 8, ENC)
    lengths = torch.tensor([8, 3, 0, 5])
    for b, n in enumerate(lengths.tolist()):
        enc[b, n:] = 99.0  # poison the padding
    _assert_matches_oracle(enc, lengths)


def test_zero_length_batch_is_a_no_op() -> None:
    # @spec PORT-PERF-001
    predictor, joint = _nets()
    enc = torch.randn(2, 4, ENC)
    with torch.no_grad():
        ids, lens, state = rnnt.decode_compact_active(
            enc, torch.tensor([0, 0]), predictor, joint, _state(2)
        )
    assert lens.tolist() == [0, 0]
    assert not bool((ids != 0).any())
    assert state.last_label.tolist() == [VOCAB, VOCAB]
    assert not bool(state.h.any()) and not bool(state.c.any())


def test_carried_state_continues_across_calls() -> None:
    # @spec PORT-DEC-001
    # Two chunks through compact-active == one concatenated chunk
    # through the oracle (state carries, no SOS re-injection).
    predictor, joint = _nets(9)
    torch.manual_seed(13)
    a = torch.randn(1, 4, ENC)
    b = torch.randn(1, 3, ENC)
    with torch.no_grad():
        whole, _ = rnnt.greedy_decode_batch(
            torch.cat([a, b], dim=1), predictor, joint, _state(1)
        )
        ids1, lens1, mid = rnnt.decode_compact_active(
            a, torch.tensor([4]), predictor, joint, _state(1)
        )
        ids2, lens2, _ = rnnt.decode_compact_active(
            b, torch.tensor([3]), predictor, joint, mid
        )
    streamed = (
        ids1[0, : int(lens1[0])].tolist()
        + ids2[0, : int(lens2[0])].tolist()
    )
    assert streamed == whole[0]


def test_cap_saturated_burst_matches() -> None:
    # @spec PORT-DEC-008
    # A quiet joint bias can drive long per-frame emissions; the
    # ten-symbol cap path must match the oracle exactly.
    predictor, joint = _nets(5)
    with torch.no_grad():
        joint.joint_net[1].bias[:VOCAB] += 3.0  # favor non-blank
    torch.manual_seed(14)
    enc = torch.randn(2, 3, ENC)
    lengths = torch.tensor([3, 2])
    predictor2, joint2 = _nets(5)
    with torch.no_grad():
        joint2.joint_net[1].bias[:VOCAB] += 3.0
        expected, _ = _oracle_rows(predictor2, joint2, enc, lengths)
        ids, lens, _ = rnnt.decode_compact_active(
            enc, lengths, predictor, joint, _state(2)
        )
    for b, burst in enumerate(expected):
        assert len(burst) == int(lengths[b]) * rnnt.MAX_SYMBOLS_PER_STEP
        assert ids[b, : int(lens[b])].tolist() == burst
