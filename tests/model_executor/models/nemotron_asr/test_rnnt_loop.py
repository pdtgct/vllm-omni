# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy label-looping decode semantics (PORT-DEC-001/002).

Invariant tests over tiny random networks; end-to-end numerical parity
with NeMo's batched computer is gated by the golden transcripts at P2
integration (the goldens' partial sequences pin the same loop).
"""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    DecodeState,
    Joint,
    Predictor,
    greedy_decode_batch,
    greedy_decode_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 12
_HID = 16


def _nets(seed: int = 5) -> tuple[Predictor, Joint]:
    torch.manual_seed(seed)
    predictor = Predictor(vocab_size=_VOCAB, pred_hidden=_HID, pred_rnn_layers=2)
    joint = Joint(
        enc_hidden=_HID,
        pred_hidden=_HID,
        joint_hidden=_HID,
        vocab_size=_VOCAB,
    )
    for net in (predictor, joint):
        for param in net.parameters():
            torch.nn.init.uniform_(param, -0.5, 0.5)
    return predictor, joint


def _fresh_state(predictor: Predictor) -> DecodeState:
    return DecodeState(
        h=torch.zeros(2, 1, _HID),
        c=torch.zeros(2, 1, _HID),
        last_label=torch.tensor([predictor.blank_id]),
    )


def test_blank_biased_joint_emits_nothing():
    predictor, joint = _nets()
    with torch.no_grad():
        joint.joint_net[1].bias[predictor.blank_id] = 100.0
    state = _fresh_state(predictor)
    emitted, out = greedy_decode_chunk(torch.randn(7, _HID), predictor, joint, state)
    assert emitted == []
    # Blank never advances predictor state (PORT-DEC-001).
    torch.testing.assert_close(out.h, state.h)
    assert int(out.last_label) == predictor.blank_id


def test_max_symbols_cap_forces_frame_advance():
    predictor, joint = _nets()
    with torch.no_grad():
        joint.joint_net[1].bias[predictor.blank_id] = -100.0
    state = _fresh_state(predictor)
    frames = 3
    emitted, _ = greedy_decode_chunk(torch.randn(frames, _HID), predictor, joint, state)
    # Never-blank joint emits exactly the cap per frame, then advances.
    assert len(emitted) == frames * 10


def test_state_carries_across_chunks_without_sos_reinjection():
    predictor, joint = _nets()
    torch.manual_seed(11)
    audio = torch.randn(8, _HID)
    # One pass over all frames...
    whole, whole_state = greedy_decode_chunk(audio, predictor, joint, _fresh_state(predictor))
    # ...equals two chunked passes threading the state (PORT-DEC-001:
    # chunk boundaries are invisible to the decode).
    first, mid = greedy_decode_chunk(audio[:4], predictor, joint, _fresh_state(predictor))
    second, end_state = greedy_decode_chunk(audio[4:], predictor, joint, mid)
    assert whole == first + second
    torch.testing.assert_close(whole_state.h, end_state.h)
    torch.testing.assert_close(whole_state.c, end_state.c)


def test_batched_decode_matches_the_reference_loop_per_stream():
    predictor, joint = _nets()
    torch.manual_seed(23)
    frames = 6
    batch = 4
    audio = torch.randn(batch, frames, _HID)
    audio[2] = 0.0  # one silent-ish stream among speech-like ones

    # Diversify per-stream state with a warmup chunk before comparing.
    singles = []
    for i in range(batch):
        _, warm = greedy_decode_chunk(
            audio[i, : frames // 2],
            predictor,
            joint,
            _fresh_state(predictor),
        )
        singles.append(warm)
    stacked = DecodeState(
        h=torch.cat([s.h for s in singles], dim=1),
        c=torch.cat([s.c for s in singles], dim=1),
        last_label=torch.cat([s.last_label for s in singles]),
    )

    batched_emitted, batched_state = greedy_decode_batch(audio, predictor, joint, stacked)
    for i in range(batch):
        ref_emitted, ref_state = greedy_decode_chunk(audio[i], predictor, joint, singles[i])
        assert batched_emitted[i] == ref_emitted
        torch.testing.assert_close(batched_state.h[:, i : i + 1], ref_state.h)
        torch.testing.assert_close(batched_state.c[:, i : i + 1], ref_state.c)
        assert int(batched_state.last_label[i]) == int(ref_state.last_label[0])


def test_batched_decode_respects_the_per_frame_cap():
    predictor, joint = _nets()
    with torch.no_grad():
        joint.joint_net[1].bias[predictor.blank_id] = -100.0
    frames, batch = 3, 2
    state = DecodeState(
        h=torch.zeros(2, batch, _HID),
        c=torch.zeros(2, batch, _HID),
        last_label=torch.full((batch,), predictor.blank_id),
    )
    emitted, _ = greedy_decode_batch(torch.randn(batch, frames, _HID), predictor, joint, state)
    assert [len(labels) for labels in emitted] == [frames * 10] * batch


def test_sos_is_blank_and_embeds_to_zeros():
    # Fresh construction: padding_idx zeroes the blank row at init (the
    # _nets fixture's blanket uniform re-init would overwrite it; real
    # weights come from the checkpoint, whose blank row trained as pad).
    predictor = Predictor(vocab_size=_VOCAB, pred_hidden=_HID, pred_rnn_layers=2)
    sos = torch.tensor([predictor.blank_id])
    assert torch.all(predictor.embed(sos) == 0.0)
