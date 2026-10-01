# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA replay regression for the explicitly prepared predictor table."""

import copy

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.rnnt import Predictor

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_prepared_table_replays_with_changed_labels_and_carried_state(fp32_no_tf32):
    torch.manual_seed(71)
    candidate = Predictor(vocab_size=19, pred_hidden=16, pred_rnn_layers=2).to("cuda").eval()
    candidate.embed.weight[candidate.blank_id].fill_(0.3)
    candidate.prepare_input_projection_table()
    table = candidate._input_projection_table
    baseline = copy.deepcopy(candidate)
    assert not baseline.input_projection_table_enabled
    labels = torch.tensor([19, 0, 7, 19], device="cuda")
    state = (torch.randn(2, 4, 16, device="cuda"), torch.randn(2, 4, 16, device="cuda"))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            candidate.step(labels, state)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_state = candidate.step(labels, state)
    for ids in ([19, 2, 8, 19], [3, 19, 3, 8], [0, 1, 18, 19]):
        labels.copy_(torch.tensor(ids, device="cuda"))
        expected, expected_state = baseline.step(labels, state)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for result, reference in zip(actual_state, expected_state):
            torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
        for carried, result in zip(state, actual_state):
            carried.copy_(result)
        assert candidate._input_projection_table is table
