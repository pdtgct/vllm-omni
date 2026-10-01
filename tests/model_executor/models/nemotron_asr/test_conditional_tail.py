# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact CPU control-flow checks; CUDA conditional capture is a separate gate."""

import gc
import weakref

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.conditional_tail import (
    ConditionalCapture,
    ConditionalTailDecoder,
    _copy_tail_merges,
    _decode_frames_with_if,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    DecodeState,
    Joint,
    Predictor,
    decode_dense_masked_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CPUBranch:
    """Execute the mathematical branch on CPU; makes no CUDA claim."""

    def __init__(self):
        self.decisions = []
        self.tensors = []

    def keep(self, *tensors):
        self.tensors.extend(tensor for tensor in tensors if tensor is not None)

    def run_if(self, active, body):
        assert active.device.type == "cpu"
        taken = bool(active.any())
        self.decisions.append(taken)
        if taken:
            body()


class _CountingPredictor:
    blank_id = 1

    def __init__(self):
        self.calls = 0

    def step(self, labels, state):
        self.calls += 1
        h, c = state
        next_h, next_c = h + 1, c + 2
        return next_h[-1], (next_h, next_c)


class _ThresholdJoint:
    def logits(self, frame, pred):
        emit = pred[:, 0] <= frame[:, 0]
        return torch.stack((emit.float(), (~emit).float()), dim=1)


def _state(batch, *, hidden=1, blank=1):
    return DecodeState(
        torch.zeros(2, batch, hidden), torch.zeros(2, batch, hidden), torch.full((batch,), blank, dtype=torch.long)
    )


def _outputs(result):
    return (
        result.token_ids,
        result.token_lengths,
        result.state.h,
        result.state.c,
        result.state.last_label,
        result.frame_emission_counts,
        result.frame_final_labels,
    )


def _equal(left, right):
    for actual, expected in zip(_outputs(left), _outputs(right), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("emissions", [0, 3, 4, 5, 10])
def test_threshold_and_committed_state_are_exact(emissions):
    predictor, joint, branch = _CountingPredictor(), _ThresholdJoint(), _CPUBranch()
    frames, lengths = torch.tensor([[[float(emissions)]]]), torch.tensor([1])
    state = _state(1)
    expected = decode_dense_masked_frames(frames, lengths, predictor, joint, state)
    predictor.calls = 0
    actual = _decode_frames_with_if(frames, lengths, predictor, joint, state, capture=branch)
    _equal(actual, expected)
    assert actual.frame_emission_counts.tolist() == [[emissions]]
    assert actual.state.h.tolist() == [[[float(emissions)]], [[float(emissions)]]]
    assert actual.state.c.tolist() == [[[2.0 * emissions]], [[2.0 * emissions]]]
    assert branch.decisions == [emissions >= 4]  # Exactly four MUST execute attempt five.
    assert predictor.calls == (10 if emissions >= 4 else 5)
    assert torch.count_nonzero(state.h) == torch.count_nonzero(state.c) == 0
    assert actual.state.h.data_ptr() != state.h.data_ptr()


def test_skip_then_emit_and_carried_recurrence_preserve_lookahead():
    predictor, joint, branch = _CountingPredictor(), _ThresholdJoint(), _CPUBranch()
    frames, lengths, state = torch.tensor([[[3.0], [3.0], [8.0]]]), torch.tensor([3]), _state(1)
    actual = _decode_frames_with_if(frames, lengths, predictor, joint, state, capture=branch)
    expected = decode_dense_masked_frames(frames, lengths, predictor, joint, state)
    _equal(actual, expected)
    assert actual.frame_emission_counts.tolist() == [[3, 0, 5]]
    assert branch.decisions == [False, False, True]
    next_frames = torch.tensor([[[8.0], [12.0]]])
    next_lengths = torch.tensor([2])
    continued = _decode_frames_with_if(next_frames, next_lengths, predictor, joint, actual.state, capture=_CPUBranch())
    independent = decode_dense_masked_frames(next_frames, next_lengths, predictor, joint, expected.state)
    _equal(continued, independent)
    assert continued.frame_emission_counts.tolist() == [[0, 4]]


def test_mixed_lanes_clamping_zero_lengths_and_cap_exhaustion():
    predictor, joint = _CountingPredictor(), _ThresholdJoint()
    frames = torch.tensor([[[3.0], [7.0], [7.0]], [[4.0], [4.0], [14.0]], [[10.0], [13.0], [13.0]], [[99.0]] * 3])
    lengths, state, branch = torch.tensor([99, 3, 2, -1]), _state(4), _CPUBranch()
    before = tuple(value.clone() for value in (frames, lengths, state.h, state.c, state.last_label))
    expected = decode_dense_masked_frames(frames, lengths, predictor, joint, state)
    actual = _decode_frames_with_if(frames, lengths, predictor, joint, state, capture=branch)
    _equal(actual, expected)
    assert actual.frame_emission_counts.tolist() == [[3, 4, 0], [4, 0, 10], [10, 3, 0], [0, 0, 0]]
    for value, saved in zip((frames, lengths, state.h, state.c, state.last_label), before, strict=True):
        torch.testing.assert_close(value, saved, atol=0, rtol=0)


@pytest.mark.parametrize("frames,max_symbols", [(0, 10), (2, 0), (1, 1), (2, 4), (2, 5), (3, 10)])
def test_degenerate_shapes_and_custom_caps_keep_the_dense_contract(frames, max_symbols):
    predictor, joint, state, branch = _CountingPredictor(), _ThresholdJoint(), _state(2), _CPUBranch()
    encoded = torch.full((2, frames, 1), 100.0)
    lengths = torch.tensor([frames, 0])
    expected = decode_dense_masked_frames(encoded, lengths, predictor, joint, state, max_symbols=max_symbols)
    actual = _decode_frames_with_if(encoded, lengths, predictor, joint, state, capture=branch, max_symbols=max_symbols)
    _equal(actual, expected)
    assert len(branch.decisions) == (frames if max_symbols > 4 else 0)


@pytest.mark.parametrize("mode", ["blank", "cap", "random"])
def test_real_predictor_joint_keeps_exact_arithmetic_and_one_projection_per_frame(mode):
    torch.manual_seed(1729)
    predictor = Predictor(vocab_size=12, pred_hidden=8, pred_rnn_layers=2).eval()
    joint = Joint(enc_hidden=8, pred_hidden=8, joint_hidden=8, vocab_size=12).eval()
    if mode != "random":
        with torch.no_grad():
            joint.joint_net[-1].bias[12] = 100 if mode == "blank" else -100
    frames, lengths, state = torch.randn(3, 3, 8), torch.tensor([3, 2, 0]), _state(3, hidden=8, blank=12)
    with torch.inference_mode():
        expected = decode_dense_masked_frames(frames, lengths, predictor, joint, state)
        projections = []
        handle = joint.enc.register_forward_hook(lambda *args: projections.append(1))
        try:
            actual = _decode_frames_with_if(frames, lengths, predictor, joint, state, capture=_CPUBranch())
        finally:
            handle.remove()
    _equal(actual, expected)
    assert len(projections) == 3


def test_eager_oracle_does_not_compile_or_require_cuda():
    decoder = ConditionalTailDecoder()
    assert decoder.prefix_attempts == 4
    with pytest.raises(AttributeError):
        decoder.prefix_attempts = 3
    state = _state(1)
    args = (torch.ones(1, 1, 1), torch.ones(1, dtype=torch.long), _CountingPredictor(), _ThresholdJoint(), state)
    _equal(decoder(*args), decode_dense_masked_frames(*args))
    assert decoder._device is None
    with pytest.raises(ValueError, match="requires CUDA"):
        decoder.prepare(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="not prepared"):
        with decoder.capture_scope():
            pass


def test_capture_owners_release_per_graph_buffers_independently():
    decoder = ConditionalTailDecoder()
    decoder._device = torch.device("cuda", 0)
    with decoder.capture_scope() as first:
        tensor = torch.zeros(1)
        first.keep(tensor)
        tensor_ref = weakref.ref(tensor)
        with pytest.raises(RuntimeError, match="cannot overlap"):
            with decoder.capture_scope():
                pass
    with decoder.capture_scope() as second:
        assert second is not first
    assert decoder._capture is None
    first_ref = weakref.ref(first)
    del decoder, tensor, first
    gc.collect()
    assert first_ref() is None and tensor_ref() is None
    assert second._graph is None
    second_ref = weakref.ref(second)
    del second
    gc.collect()
    assert second_ref() is None


def _merge_fields(offset=0):
    fields = []
    for index in range(7):
        shape = (3,) if index in (2, 6) else ((3, 4) if index == 3 else (2, 3, 4))
        dtype = torch.int64 if index in (2, 6) else torch.float32
        tensor = torch.full(shape, offset + index, dtype=dtype)
        if dtype == torch.float32 and offset:
            # Include signed zero, infinity and a NaN payload: compare raw bits.
            tensor.view(torch.int32).flatten()[:3] = torch.tensor([-2147483648, 2139095040, 2143289635])
        elif dtype == torch.int64 and offset:
            tensor[0] = 2**60 + index
        fields.append(tensor)
    return tuple(fields)


def _bits(tensor):
    return tensor.view(torch.uint8).clone()


@pytest.mark.parametrize("final_frame", [False, True])
def test_grouped_merges_match_serial_copies_bitwise_and_preserve_sources(monkeypatch, final_frame):
    destinations, sources = _merge_fields(), _merge_fields(100)
    expected = tuple(tensor.clone() for tensor in destinations)
    source_bits = tuple(_bits(tensor) for tensor in sources)
    selected = (0, 1, 2, 6) if final_frame else range(7)
    for index in selected:
        expected[index].copy_(sources[index])
    calls = []
    foreach_copy = torch._foreach_copy_

    def record(targets, values):
        calls.append(
            (targets[0].dtype, tuple(next(i for i, t in enumerate(destinations) if t is dst) for dst in targets))
        )
        return foreach_copy(targets, values)

    monkeypatch.setattr(torch, "_foreach_copy_", record)
    _copy_tail_merges(destinations, sources, final_frame=final_frame)
    assert calls == [(torch.float32, (0, 1) if final_frame else (0, 1, 3, 4, 5)), (torch.int64, (2, 6))]
    assert all(torch.equal(_bits(actual), _bits(reference)) for actual, reference in zip(destinations, expected))
    assert all(torch.equal(_bits(actual), before) for actual, before in zip(sources, source_bits))


def test_two_frame_tail_uses_four_grouped_copy_calls_with_final_lookahead_excluded(monkeypatch):
    calls = []
    foreach_copy = torch._foreach_copy_

    def record(targets, sources):
        calls.append((targets[0].dtype, len(targets)))
        return foreach_copy(targets, sources)

    monkeypatch.setattr(torch, "_foreach_copy_", record)
    frames, lengths, state = torch.tensor([[[10.0], [20.0]]]), torch.tensor([2]), _state(1)
    predictor, joint = _CountingPredictor(), _ThresholdJoint()
    actual = _decode_frames_with_if(frames, lengths, predictor, joint, state, capture=_CPUBranch())
    expected = decode_dense_masked_frames(frames, lengths, predictor, joint, state)
    _equal(actual, expected)
    assert actual.frame_emission_counts.tolist() == [[10, 10]]
    assert calls == [(torch.float32, 5), (torch.int64, 2), (torch.float32, 2), (torch.int64, 2)]


@pytest.mark.parametrize(
    "invalid",
    [
        "dtype",
        "device",
        "layout",
        "noncontiguous",
        "shape",
        "stride",
        "empty",
        "negative_view",
        "dest_alias",
        "read_alias",
        "cross_dtype_alias",
    ],
)
def test_merge_metadata_and_write_aliases_fail_before_either_group_writes(monkeypatch, invalid):
    destinations, sources = list(_merge_fields()), list(_merge_fields(100))
    if invalid == "dtype":
        sources[6] = sources[6].float()  # Bad second group must prevent first-group writes.
    elif invalid == "device":
        sources[6] = torch.empty(3, dtype=torch.int64, device="meta")
    elif invalid == "layout":
        sources[6] = sources[6].to_sparse()
    elif invalid == "noncontiguous":
        sources[0] = sources[0].transpose(0, 1)
    elif invalid == "shape":
        sources[6] = torch.zeros(4, dtype=torch.int64)
    elif invalid == "stride":
        destinations[2] = torch.zeros(1, 3, dtype=torch.int64)
        sources[2] = torch.ones(3, dtype=torch.int64).as_strided((1, 3), (9, 1))
        assert sources[2].is_contiguous()
    elif invalid == "empty":
        destinations[6] = sources[6] = torch.empty(0, dtype=torch.int64)
    elif invalid == "negative_view":
        sources[0] = torch._neg_view(sources[0])
        assert sources[0].is_neg()
    elif invalid == "cross_dtype_alias":
        sources[6] = destinations[0].view(torch.int64).flatten()[:3]
    else:
        storage = torch.arange(48, dtype=torch.float32)
        destinations[0] = storage[:24].view(2, 3, 4)
        if invalid == "dest_alias":
            destinations[1] = storage[12:36].view(2, 3, 4)
        else:
            sources[1] = storage[12:36].view(2, 3, 4)
    before = tuple(_bits(tensor) for tensor in destinations)
    monkeypatch.setattr(
        torch, "_foreach_copy_", lambda *args: pytest.fail("copy ran before all metadata was validated")
    )
    with pytest.raises(ValueError, match="conditional-tail merge"):
        _copy_tail_merges(tuple(destinations), tuple(sources), final_frame=False)
    assert all(torch.equal(_bits(tensor), saved) for tensor, saved in zip(destinations, before))


def test_merge_accepts_disjoint_storage_views_and_shared_read_only_source():
    destinations, sources = list(_merge_fields()), list(_merge_fields(100))
    storage = torch.arange(48, dtype=torch.float32)
    destinations[0] = storage[:24].view(2, 3, 4)
    sources[0] = sources[1] = storage[24:].view(2, 3, 4)
    expected = tuple(_bits(tensor) for tensor in sources)
    _copy_tail_merges(tuple(destinations), tuple(sources), final_frame=False)
    assert all(torch.equal(_bits(tensor), saved) for tensor, saved in zip(destinations, expected))


def test_grouped_merge_buffers_remain_independent_between_graph_owners():
    first, second = ConditionalCapture(), ConditionalCapture()
    first_buffers, second_buffers, sources = _merge_fields(), _merge_fields(200), _merge_fields(100)
    first.keep(*first_buffers)
    second.keep(*second_buffers)
    second_before = tuple(_bits(tensor) for tensor in second_buffers)
    _copy_tail_merges(first_buffers, sources, final_frame=False)
    assert all(torch.equal(_bits(tensor), saved) for tensor, saved in zip(second_buffers, second_before))
    first_ref, first_tensor_ref = weakref.ref(first), weakref.ref(first_buffers[0])
    del first, first_buffers
    gc.collect()
    assert first_ref() is None and first_tensor_ref() is None
    _copy_tail_merges(second_buffers, sources, final_frame=True)
    assert torch.equal(_bits(second_buffers[0]), _bits(sources[0]))
    assert torch.equal(_bits(second_buffers[3]), second_before[3])
