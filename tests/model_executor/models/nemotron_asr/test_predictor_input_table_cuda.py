# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact synthetic rejection tests before trained predictor-table admission.

Run with pytest -x before loading checkpoint weights. Passing these tests does
not replace protocol 2 or trained qualification. Every comparison is exact;
the maximum absolute error in a failure is diagnostic, never a tolerance.
"""

import copy

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    DecodeState,
    Joint,
    Predictor,
    decode_dense_masked_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

_TIERS = (1, 2, 4, 8, 16, 32, 64, 128)
_VOCAB = 13087  # 13088 embedding rows including blank.
_HIDDEN = 640


@pytest.fixture
def fp32_no_tf32():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(precision)


def _predictors(batch_sizes):
    torch.manual_seed(71)
    candidate = Predictor(vocab_size=_VOCAB, pred_hidden=_HIDDEN, pred_rnn_layers=2).to("cuda").eval()
    candidate.embed.weight[candidate.blank_id].uniform_(-0.3, 0.3)
    candidate.prepare_input_projection_table(batch_sizes=batch_sizes)
    baseline = copy.deepcopy(candidate)
    assert candidate.input_projection_table_enabled
    assert not baseline.input_projection_table_enabled
    return baseline, candidate


def _labels(batch, step):
    patterns = ((_VOCAB,), (_VOCAB - 1,), (7, 0, _VOCAB, _VOCAB - 1, 640, 7))
    pattern = torch.tensor(patterns[step], device="cuda", dtype=torch.long)
    return pattern[torch.arange(batch, device="cuda") % len(pattern)]


def _exact(name, actual, expected):
    if not torch.equal(actual, expected):
        count = int(torch.count_nonzero(actual != expected))
        maximum = float((actual.to(torch.float64) - expected.to(torch.float64)).abs().max())
        pytest.fail(f"{name}: unequal={count}/{actual.numel()}, max_abs={maximum:.9g}")


def _predictor_fields(output):
    value, (h, c) = output
    return {"output": value, "h": h, "c": c}


def _decode_fields(output):
    return {
        "token_ids": output.token_ids,
        "token_lengths": output.token_lengths,
        "h": output.state.h,
        "c": output.state.c,
        "last_label": output.state.last_label,
        "frame_emission_counts": output.frame_emission_counts,
        "frame_final_labels": output.frame_final_labels,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", _TIERS)
@torch.no_grad()
def test_actual_predictor_geometry_exact_eager_and_replay(batch, fp32_no_tf32):
    baseline, candidate = _predictors((batch,))
    table = getattr(candidate, f"_input_projection_table_{batch}")
    baseline_state = (torch.randn(2, batch, _HIDDEN, device="cuda"), torch.randn(2, batch, _HIDDEN, device="cuda"))
    eager_state = tuple(value.clone() for value in baseline_state)
    replay_state = tuple(value.clone() for value in baseline_state)
    labels = _labels(batch, 0)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            candidate.step(labels, replay_state)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = candidate.step(labels, replay_state)
    for step in range(3):
        labels.copy_(_labels(batch, step))
        expected = baseline.step(labels, baseline_state)
        eager = candidate.step(labels, eager_state)
        graph.replay()
        for name, reference in _predictor_fields(expected).items():
            _exact(f"tier={batch} step={step} eager {name}", _predictor_fields(eager)[name], reference)
            _exact(f"tier={batch} step={step} replay {name}", _predictor_fields(replay)[name], reference)
        # Carry each arm's own result, never copy baseline results into candidate.
        baseline_state = expected[1]
        eager_state = eager[1]
        for carried, result in zip(replay_state, replay[1]):
            carried.copy_(result)
        assert getattr(candidate, f"_input_projection_table_{batch}") is table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", _TIERS)
@torch.no_grad()
def test_forced_cap_decoder_exact_seven_fields(batch, fp32_no_tf32):
    baseline, candidate = _predictors((batch,))
    # Only the joint is constant. Predictor parameters and carried states stay
    # nonzero, so the cap fixture cannot conceal predictor GEMM rounding.
    joint = Joint(enc_hidden=2, pred_hidden=_HIDDEN, joint_hidden=2, vocab_size=_VOCAB).to("cuda").eval()
    for parameter in joint.parameters():
        parameter.zero_()
    joint.joint_net[1].bias[_VOCAB - 1] = 1.0
    frames = torch.randn(batch, 2, 2, device="cuda")
    lengths = torch.arange(batch, device="cuda") % 3
    lengths[0] = 2
    baseline_state = DecodeState(
        h=torch.randn(2, batch, _HIDDEN, device="cuda"),
        c=torch.randn(2, batch, _HIDDEN, device="cuda"),
        last_label=_labels(batch, 2),
    )
    eager_state = copy.deepcopy(baseline_state)
    replay_state = copy.deepcopy(baseline_state)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            decode_dense_masked_frames(frames, lengths, candidate, joint, replay_state)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = decode_dense_masked_frames(frames, lengths, candidate, joint, replay_state)
    for step in range(2):
        expected = decode_dense_masked_frames(frames, lengths, baseline, joint, baseline_state)
        eager = decode_dense_masked_frames(frames, lengths, candidate, joint, eager_state)
        graph.replay()
        for name, reference in _decode_fields(expected).items():
            _exact(f"tier={batch} chunk={step} eager {name}", _decode_fields(eager)[name], reference)
            _exact(f"tier={batch} chunk={step} replay {name}", _decode_fields(replay)[name], reference)
        _exact("forced cap token lengths", expected.token_lengths, (lengths * 10).to(torch.int32))
        baseline_state = expected.state
        eager_state = eager.state
        replay_state.h.copy_(replay.state.h)
        replay_state.c.copy_(replay.state.c)
        replay_state.last_label.copy_(replay.state.last_label)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_all_tables_and_graphs_retain_independent_state_after_all_captures(fp32_no_tf32):
    baseline, candidate = _predictors(_TIERS)
    assert candidate.input_projection_table_batch_sizes == _TIERS
    tables = {batch: getattr(candidate, f"_input_projection_table_{batch}") for batch in _TIERS}
    captures = []
    for batch in _TIERS:
        labels = _labels(batch, 0)
        baseline_state = (torch.randn(2, batch, _HIDDEN, device="cuda"), torch.randn(2, batch, _HIDDEN, device="cuda"))
        replay_state = tuple(value.clone() for value in baseline_state)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                candidate.step(labels, replay_state)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = candidate.step(labels, replay_state)
        captures.append((batch, labels, baseline_state, replay_state, graph, replay))
    # All tables, captured graphs, inputs and outputs remain alive together.
    # Replaying early keys now detects storage reused by later preparations or
    # captures. Each key and arm advances only its own independently held state.
    for step in (1, 2):
        for batch, labels, baseline_state, replay_state, graph, replay in captures:
            labels.copy_(_labels(batch, step))
            expected = baseline.step(labels, baseline_state)
            graph.replay()
            for name, reference in _predictor_fields(expected).items():
                _exact(
                    f"all captures tier={batch} step={step} replay {name}", _predictor_fields(replay)[name], reference
                )
            for carried, result in zip(baseline_state, expected[1]):
                carried.copy_(result)
            for carried, result in zip(replay_state, replay[1]):
                carried.copy_(result)
            assert getattr(candidate, f"_input_projection_table_{batch}") is tables[batch]
