# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental predictor table numerics and derived-state lifetime."""

import copy

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.rnnt import Predictor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _predictor():
    torch.manual_seed(71)
    predictor = Predictor(vocab_size=19, pred_hidden=16, pred_rnn_layers=2).eval()
    # Exercise checkpoint-provided blank embeddings, not the zero initializer.
    with torch.no_grad():
        predictor.embed.weight[predictor.blank_id].fill_(0.3)
    return predictor


@torch.no_grad()
def test_table_matches_carried_state_and_nonzero_blank():
    candidate = _predictor()
    baseline = copy.deepcopy(candidate)
    candidate.prepare_input_projection_table()
    h, c = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
    original_h, original_c = h.clone(), c.clone()
    base_state, table_state = (h, c), (h.clone(), c.clone())
    for labels in ([19, 0, 7, 19], [3, 19, 3, 8], [0, 1, 18, 19]) * 4:
        labels = torch.tensor(labels)
        expected, base_state = baseline.step(labels, base_state)
        actual, table_state = candidate.step(labels, table_state)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for actual_state, expected_state in zip(table_state, base_state):
            torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-6)
    assert torch.equal(h, original_h)
    assert torch.equal(c, original_c)


@torch.no_grad()
def test_deepcopy_and_checkpoint_never_carry_admission():
    candidate = _predictor()
    unprepared = copy.deepcopy(candidate)
    keys = set(candidate.state_dict())
    candidate.prepare_input_projection_table()
    prepared_copy = copy.deepcopy(candidate)
    assert not unprepared.input_projection_table_enabled
    assert not prepared_copy.input_projection_table_enabled
    assert candidate.input_projection_table_enabled
    assert set(candidate.state_dict()) == keys
    labels = torch.tensor([19, 2])
    state = (torch.randn(2, 2, 16), torch.randn(2, 2, 16))
    # A copy of the candidate must execute the exact original GEMMs.
    actual, _ = prepared_copy.step(labels, state)
    expected, _ = unprepared.step(labels, state)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mutation", ["embedding", "input_weight", "replace", "child_load", "child_dtype"])
@torch.no_grad()
def test_mutated_projection_sources_fail_closed(mutation):
    predictor = _predictor()
    predictor.prepare_input_projection_table()
    if mutation == "embedding":
        predictor.embed.weight.add_(1)
    elif mutation == "input_weight":
        predictor.rnn.weight_ih_l0.add_(1)
    elif mutation == "replace":
        predictor.embed.weight = torch.nn.Parameter(predictor.embed.weight.clone())
    elif mutation == "child_load":
        predictor.rnn.load_state_dict(predictor.rnn.state_dict())
    else:
        predictor.rnn.double()
    with pytest.raises(RuntimeError, match="changed"):
        predictor.step(torch.tensor([0]), (torch.zeros(2, 1, 16), torch.zeros(2, 1, 16)))
    predictor.invalidate_input_projection_table()
    predictor.float()
    predictor.prepare_input_projection_table()
    assert predictor.input_projection_table_enabled


@pytest.mark.parametrize("mutation", ["load", "dtype", "device", "train", "disable"])
@torch.no_grad()
def test_lifecycle_transitions_remove_derived_table(mutation):
    predictor = _predictor()
    predictor.prepare_input_projection_table()
    if mutation == "load":
        predictor.load_state_dict(predictor.state_dict())
    elif mutation == "dtype":
        predictor.to(torch.float64)
    elif mutation == "device":
        predictor.to("cpu")
    elif mutation == "train":
        predictor.train()
    else:
        predictor.invalidate_input_projection_table()
    assert not predictor.input_projection_table_enabled


def test_unsupported_admission_is_explicit():
    predictor = _predictor()
    with pytest.raises(ValueError, match="inference"):
        predictor.prepare_input_projection_table()
    with torch.no_grad():
        predictor.double()
        with pytest.raises(ValueError, match="float32"):
            predictor.prepare_input_projection_table()
        assert not predictor.input_projection_table_enabled


@torch.no_grad()
def test_table_rejects_autocast_and_non_fp32_state():
    predictor = _predictor()
    predictor.prepare_input_projection_table()
    labels = torch.tensor([19])
    state = (torch.zeros(2, 1, 16), torch.zeros(2, 1, 16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            predictor.step(labels, state)
    with pytest.raises(ValueError, match="float32"):
        predictor.step(labels, tuple(x.double() for x in state))


@torch.no_grad()
def test_input_table_removes_one_gemm_and_reuses_preparation():
    predictor = _predictor()
    labels = torch.tensor([19, 0, 2])
    state = (torch.zeros(2, 3, 16), torch.zeros(2, 3, 16))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as baseline:
        predictor.step(labels, state)
    predictor.prepare_input_projection_table()
    table = predictor._input_projection_table
    predictor.prepare_input_projection_table()
    assert predictor._input_projection_table is table
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as candidate:
        predictor.step(labels, state)
    assert sum(event.count for event in baseline.key_averages() if event.key == "aten::mm") == 4
    assert sum(event.count for event in candidate.key_averages() if event.key == "aten::mm") == 3


def test_unversioned_parameters_rejected():
    with torch.inference_mode():
        predictor = _predictor()
        with pytest.raises(ValueError, match="versioned"):
            predictor.prepare_input_projection_table()
        assert not predictor.input_projection_table_enabled


@torch.no_grad()
def test_preparation_failure_does_not_publish_table(monkeypatch):
    predictor = _predictor()

    def fail(*args, **kwargs):
        raise RuntimeError("injected allocation failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__matmul__", fail)
        with pytest.raises(RuntimeError, match="injected"):
            predictor.prepare_input_projection_table()
    assert not predictor.input_projection_table_enabled
    predictor.prepare_input_projection_table()
    assert predictor.input_projection_table_enabled
