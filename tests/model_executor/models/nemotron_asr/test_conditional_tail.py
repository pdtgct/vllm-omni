# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact CPU control-flow checks; CUDA conditional capture is a separate gate."""

import gc
import weakref

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.conditional_tail import (
    ConditionalTailDecoder,
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
    assert decoder._compiled is None
    with pytest.raises(ValueError, match="requires CUDA"):
        decoder.prepare(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="not prepared"):
        with decoder.capture_scope():
            pass


def test_capture_owners_release_buffers_and_keep_shared_module_until_last_owner():
    class Compiled:
        pass

    decoder = ConditionalTailDecoder()
    decoder._compiled = Compiled()
    compiled_ref = weakref.ref(decoder._compiled)
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
    assert compiled_ref() is not None
    del second
    gc.collect()
    assert compiled_ref() is None


def test_raw_if_body_failure_restores_parent_stream_and_propagates(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.nemotron_asr.conditional_tail import ConditionalCapture

    calls: list[tuple] = []

    def completed(name, payload, result=(0,)):
        calls.append((name, payload))
        return result

    parent, child = SimpleNamespace(cuda_stream=1), SimpleNamespace(cuda_stream=2)
    runtime = SimpleNamespace(
        cudaStreamCaptureStatus=SimpleNamespace(cudaStreamCaptureStatusActive="active"),
        cudaStreamUpdateCaptureDependenciesFlags=SimpleNamespace(cudaStreamSetCaptureDependencies="set"),
        cudaStreamCaptureMode=SimpleNamespace(cudaStreamCaptureModeThreadLocal="thread"),
        cudaStreamGetCaptureInfo=lambda stream: (0, "active", 7, "parent-graph", ["prefix"]),
        cudaGraphConditionalHandleCreate=lambda graph, value, flags: (0, 9),
        cudaStreamUpdateCaptureDependencies=lambda *args: completed("dependencies", args),
        cudaStreamBeginCaptureToGraph=lambda *args: completed("begin", args[0]),
        cudaStreamEndCapture=lambda stream: completed("end", stream, (0, "body")),
    )

    def add_node(graph, dependencies, edge_data, count, params):
        assert graph == "parent-graph" and dependencies == ["prefix"] and count == 1
        assert params.conditional.type == "if" and params.conditional.size == 1
        calls.append(("if",))
        return 0, "if-node"

    driver = SimpleNamespace(
        CUgraphNodeParams=lambda: SimpleNamespace(conditional=SimpleNamespace(phGraph_out=["body"])),
        CUgraphNodeType=SimpleNamespace(CU_GRAPH_NODE_TYPE_CONDITIONAL="conditional"),
        CUgraphConditionalNodeType=SimpleNamespace(CU_GRAPH_COND_TYPE_IF="if"),
        cuGraphAddNode=add_node,
    )
    compiled = SimpleNamespace(
        driver=driver,
        runtime=runtime,
        context="context",
        launch=lambda arguments, stream: calls.append(("predicate", stream.cuda_stream)),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: parent)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: child)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: calls.append(("stream", stream.cuda_stream)))
    capture = ConditionalCapture(compiled)

    def fail():
        raise RuntimeError("body capture failed")

    with pytest.raises(RuntimeError, match="body capture failed"):
        capture.run_if(torch.ones(2, dtype=torch.bool), fail)
    assert [call[0] for call in calls] == ["predicate", "if", "dependencies", "begin", "stream", "end", "stream"]
    assert calls[-1] == ("stream", parent.cuda_stream)
