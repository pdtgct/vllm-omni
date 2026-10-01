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
    candidate.prepare_input_projection_table(batch_sizes=(1, 4))
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
    candidate.prepare_input_projection_table(batch_sizes=(1, 4))
    prepared_copy = copy.deepcopy(candidate)
    assert not unprepared.input_projection_table_enabled
    assert not prepared_copy.input_projection_table_enabled
    assert prepared_copy.input_projection_table_batch_sizes == ()
    assert not list(prepared_copy.named_buffers())
    assert candidate.input_projection_table_enabled
    assert candidate.input_projection_table_batch_sizes == (1, 4)
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
    predictor.prepare_input_projection_table(batch_sizes=(1, 4))
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
    predictor.prepare_input_projection_table(batch_sizes=(1, 4))
    assert predictor.input_projection_table_enabled


@pytest.mark.parametrize("mutation", ["load", "dtype", "device", "train", "disable"])
@torch.no_grad()
def test_lifecycle_transitions_remove_derived_table(mutation):
    predictor = _predictor()
    predictor.prepare_input_projection_table(batch_sizes=(1, 4))
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
    assert predictor.input_projection_table_batch_sizes == ()
    assert not list(predictor.named_buffers())


def test_unsupported_admission_is_explicit():
    predictor = _predictor()
    with pytest.raises(ValueError, match="inference"):
        predictor.prepare_input_projection_table(batch_sizes=(1, 4))
    with torch.no_grad():
        predictor.double()
        with pytest.raises(ValueError, match="float32"):
            predictor.prepare_input_projection_table(batch_sizes=(1, 4))
        assert not predictor.input_projection_table_enabled


@torch.no_grad()
def test_table_rejects_autocast_and_non_fp32_state():
    predictor = _predictor()
    predictor.prepare_input_projection_table(batch_sizes=(1, 4))
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
    predictor.prepare_input_projection_table(batch_sizes=(3,))
    table = predictor._input_projection_table_3
    predictor.prepare_input_projection_table(batch_sizes=(3,))
    assert predictor._input_projection_table_3 is table
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as candidate:
        predictor.step(labels, state)
    assert sum(event.count for event in baseline.key_averages() if event.key == "aten::mm") == 4
    assert sum(event.count for event in candidate.key_averages() if event.key == "aten::mm") == 3


def test_unversioned_parameters_rejected():
    with torch.inference_mode():
        predictor = _predictor()
        with pytest.raises(ValueError, match="versioned"):
            predictor.prepare_input_projection_table(batch_sizes=(1, 4))
        assert not predictor.input_projection_table_enabled


@torch.no_grad()
def test_preparation_failure_does_not_publish_table(monkeypatch):
    predictor = _predictor()

    def fail(*args, **kwargs):
        raise RuntimeError("injected allocation failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__matmul__", fail)
        with pytest.raises(RuntimeError, match="injected"):
            predictor.prepare_input_projection_table(batch_sizes=(1, 4))
    assert not predictor.input_projection_table_enabled
    predictor.prepare_input_projection_table(batch_sizes=(1, 4))
    assert predictor.input_projection_table_enabled


@torch.no_grad()
def test_missing_batch_rejected_until_explicitly_prepared():
    predictor = _predictor()
    original = copy.deepcopy(predictor)
    predictor.prepare_input_projection_table(batch_sizes=(1, 4))
    original_table = predictor._input_projection_table_1
    labels = torch.tensor([19, 3])
    state = (torch.randn(2, 2, 16), torch.randn(2, 2, 16))
    with pytest.raises(RuntimeError, match="batch size 2 was not prepared"):
        predictor.step(labels, state)
    predictor.prepare_input_projection_table(batch_sizes=(2, 4))
    assert predictor.input_projection_table_batch_sizes == (1, 2, 4)
    assert predictor._input_projection_table_1 is original_table
    expected, expected_state = original.step(labels, state)
    actual, actual_state = predictor.step(labels, state)
    assert torch.equal(actual, expected)
    for value, reference in zip(actual_state, expected_state):
        assert torch.equal(value, reference)


@pytest.mark.parametrize("batch", (1, 2, 4, 8))
@torch.no_grad()
def test_per_batch_tables_preserve_exact_projection_and_independent_states(batch):
    candidate = _predictor()
    baseline = copy.deepcopy(candidate)
    candidate.prepare_input_projection_table(batch_sizes=(1, 2, 4, 8))
    table = getattr(candidate, f"_input_projection_table_{batch}")
    baseline_state = (torch.randn(2, batch, 16), torch.randn(2, batch, 16))
    candidate_state = tuple(value.clone() for value in baseline_state)
    for pattern in ((19, 18, 0, 7), (7, 19, 7, 0), (18, 18, 19, 19)):
        labels = torch.tensor([pattern[row % len(pattern)] for row in range(batch)])
        original_projection = baseline.embed(labels) @ baseline.rnn.weight_ih_l0.t()
        assert torch.equal(torch.nn.functional.embedding(labels, table), original_projection)
        expected, baseline_state = baseline.step(labels, baseline_state)
        actual, candidate_state = candidate.step(labels, candidate_state)
        assert torch.equal(actual, expected)
        for value, reference in zip(candidate_state, baseline_state):
            assert torch.equal(value, reference)


@torch.no_grad()
def test_tail_is_projected_at_admitted_batch_shape(monkeypatch):
    predictor = _predictor()
    original_mm = torch.Tensor.__matmul__
    inputs = []

    def record(left, right):
        inputs.append(left.clone())
        return original_mm(left, right)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__matmul__", record)
        predictor.prepare_input_projection_table(batch_sizes=(8,))
    assert [tuple(value.shape) for value in inputs] == [(8, 16)] * 3
    # Twenty vocabulary rows leave a four-row tail, including the real blank.
    assert torch.equal(inputs[-1][:4], predictor.embed.weight[-4:])
    assert torch.count_nonzero(inputs[-1][4:]) == 0
    expected = inputs[-1] @ predictor.rnn.weight_ih_l0.t()
    assert torch.equal(predictor._input_projection_table_8[-4:], expected[:4])


@pytest.mark.parametrize("existing", (False, True))
@torch.no_grad()
def test_multi_shape_preparation_is_atomic_on_failure(monkeypatch, existing):
    predictor = _predictor()
    if existing:
        predictor.prepare_input_projection_table(batch_sizes=(1,))
    original_buffers = dict(predictor.named_buffers())
    original_mm = torch.Tensor.__matmul__

    def fail_second_shape(left, right):
        if left.shape[0] == 8:
            raise RuntimeError("injected allocation failure after first table")
        return original_mm(left, right)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__matmul__", fail_second_shape)
        with pytest.raises(RuntimeError, match="injected"):
            predictor.prepare_input_projection_table(batch_sizes=(4, 8))
    assert predictor.input_projection_table_batch_sizes == ((1,) if existing else ())
    assert dict(predictor.named_buffers()).keys() == original_buffers.keys()
    assert all(dict(predictor.named_buffers())[name] is value for name, value in original_buffers.items())
    predictor.prepare_input_projection_table(batch_sizes=(4, 8))
    assert predictor.input_projection_table_batch_sizes == ((1, 4, 8) if existing else (4, 8))


@torch.no_grad()
def test_source_change_during_preparation_rejects_all_new_tables(monkeypatch):
    predictor = _predictor()
    original_mm = torch.Tensor.__matmul__
    changed = False

    def mutate(left, right):
        nonlocal changed
        result = original_mm(left, right)
        if not changed:
            predictor.embed.weight.add_(0.1)
            changed = True
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__matmul__", mutate)
        with pytest.raises(RuntimeError, match="changed"):
            predictor.prepare_input_projection_table(batch_sizes=(1, 4))
    assert predictor.input_projection_table_batch_sizes == ()
    assert not list(predictor.named_buffers())


@pytest.mark.parametrize("batch_sizes", ((), (0,), (-1,), (True,), (1.5,)))
@torch.no_grad()
def test_batch_admission_requires_explicit_positive_integers(batch_sizes):
    predictor = _predictor()
    with pytest.raises(ValueError, match="batch sizes"):
        predictor.prepare_input_projection_table(batch_sizes=batch_sizes)
    assert not predictor.input_projection_table_enabled
