# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Predictor LSTM: manual cell (default) vs fused torch.nn.LSTM variant.

Specs: PORT-PREC-004 (manual cell from primitives is the default and
keeps ``(h, c)`` under the PrecisionPolicy; a weight-shared fused
``torch.nn.LSTM`` is the optional fp32-CUDA variant; an L1 test holds
the two to tensor-tolerance equivalence), PORT-PREC-005 (state dtype is
policy-controlled, defaults fp32).
"""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr import rnnt_cell
from vllm_omni.model_executor.models.nemotron_asr.rnnt_cell import (
    ManualLSTM,
)

pytestmark = [pytest.mark.core_model]

_LAYERS = 2
_HIDDEN = 640
_INPUT = 640


def _fused_reference(manual: ManualLSTM) -> torch.nn.LSTM:
    """A torch.nn.LSTM sharing the manual cell's weights."""
    fused = torch.nn.LSTM(
        input_size=_INPUT,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        batch_first=True,
    )
    with torch.no_grad():
        for name, param in fused.named_parameters():
            param.copy_(manual.get_parameter(name))
    return fused


@pytest.fixture
def manual() -> ManualLSTM:
    torch.manual_seed(1234)
    cell = ManualLSTM(input_size=_INPUT, hidden_size=_HIDDEN, num_layers=_LAYERS)
    for param in cell.parameters():
        torch.nn.init.uniform_(param, -0.1, 0.1)
    return cell


@pytest.mark.cpu
def test_weight_names_match_torch_lstm(manual):
    # Weight layout mirrors torch.nn.LSTM so NeMo/HF predictor weights
    # load into either implementation unchanged.
    expected = {
        f"{kind}_{gate}_l{layer}" for kind in ("weight", "bias") for gate in ("ih", "hh") for layer in range(_LAYERS)
    }
    assert {n for n, _ in manual.named_parameters()} == expected


@pytest.mark.cpu
def test_manual_equals_fused_over_sequence(manual):
    # PORT-PREC-004's equivalence bar: stepwise manual == fused within
    # tensor tolerance over a multi-step sequence, states included.
    fused = _fused_reference(manual)
    torch.manual_seed(7)
    seq = torch.randn(3, 17, _INPUT)

    h = torch.zeros(_LAYERS, 3, _HIDDEN)
    c = torch.zeros(_LAYERS, 3, _HIDDEN)
    outs = []
    for t in range(seq.shape[1]):
        y, (h, c) = manual.step(seq[:, t], (h, c))
        outs.append(y)
    manual_out = torch.stack(outs, dim=1)

    fused_out, (fh, fc) = fused(seq)
    torch.testing.assert_close(manual_out, fused_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h, fh, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c, fc, atol=1e-5, rtol=1e-5)


@pytest.mark.cpu
def test_state_dtype_is_policy_controlled(manual):
    # (h, c) may be held fp32 while inputs arrive in a lower compute
    # dtype — the recurrent-state axis is independent (PORT-PREC-005).
    x = torch.randn(2, _INPUT, dtype=torch.bfloat16)
    h = torch.zeros(_LAYERS, 2, _HIDDEN, dtype=torch.float32)
    c = torch.zeros(_LAYERS, 2, _HIDDEN, dtype=torch.float32)
    y, (h2, c2) = manual.step(x, (h, c))
    assert h2.dtype == torch.float32
    assert c2.dtype == torch.float32


def _pointwise_reference(input_mm, hidden_mm, bias_ih, bias_hh, cell, *, gate_values=None):
    # Keep the original eager association, including every materialized product.
    gates = input_mm + bias_ih + hidden_mm + bias_hh
    i, f, g, o = gates.chunk(4, dim=1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c_next = f * cell + i * g
    h_next = o * c_next.tanh()
    if gate_values is not None:
        gate_values[0].copy_(gates)
        gate_values[1].copy_(torch.cat((i, f, g, o), dim=1))
    return h_next, c_next


def _step_reference(model, x, state):
    h, c = state
    layer_input = x.to(h.dtype)
    next_h, next_c = [], []
    for layer in range(model.num_layers):
        w_ih = model.get_parameter(f"weight_ih_l{layer}").to(h.dtype)
        w_hh = model.get_parameter(f"weight_hh_l{layer}").to(h.dtype)
        b_ih = model.get_parameter(f"bias_ih_l{layer}").to(h.dtype)
        b_hh = model.get_parameter(f"bias_hh_l{layer}").to(h.dtype)
        # The two GEMMs have the same full batch/shape as the original step.
        gates = layer_input @ w_ih.t() + b_ih + h[layer] @ w_hh.t() + b_hh
        i, f, g, o = gates.chunk(4, dim=1)
        c_next = f.sigmoid() * c[layer] + i.sigmoid() * g.tanh()
        layer_input = o.sigmoid() * c_next.tanh()
        next_h.append(layer_input)
        next_c.append(c_next)
    return layer_input, (torch.stack(next_h), torch.stack(next_c))


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_pointwise_dispatch_preserves_cpu_and_grad_path(dtype):
    h = torch.zeros(2, 3, 7, dtype=dtype)
    with torch.no_grad():
        assert not rnnt_cell._use_fused_pointwise(h, h)
    assert not rnnt_cell._use_fused_pointwise(h, h)


@pytest.mark.cpu
def test_pointwise_integration_preserves_full_batch_gemms(monkeypatch):
    torch.manual_seed(246)
    model = ManualLSTM(input_size=5, hidden_size=7, num_layers=2)
    state = (torch.randn(2, 3, 7), torch.randn(2, 3, 7))
    x = torch.randn(3, 5)
    calls = []

    def checked_pointwise(input_mm, hidden_mm, bias_ih, bias_hh, cell):
        calls.append((input_mm.shape, hidden_mm.shape, cell.shape))
        return _pointwise_reference(input_mm, hidden_mm, bias_ih, bias_hh, cell)

    monkeypatch.setattr(rnnt_cell, "_use_fused_pointwise", lambda h, c: True)
    monkeypatch.setattr(rnnt_cell, "lstm_pointwise", checked_pointwise)
    with torch.no_grad():
        expected = _step_reference(model, x, state)
        actual = model.step(x, state)
    assert calls == [((3, 28), (3, 28), (3, 7))] * 2
    for got, want in zip((actual[0], *actual[1]), (expected[0], *expected[1])):
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@pytest.fixture
def cuda_device():
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_cuda():
        pytest.skip("Requires NVIDIA CUDA; CPU is not an exactness substitute")
    from vllm.triton_utils import HAS_TRITON

    assert HAS_TRITON, "CUDA pointwise validation requires the runtime Triton compiler"
    return torch.device("cuda")


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [7, 640])
@pytest.mark.parametrize(
    "case", ["mixed", "cancellation", "cell_cancellation", "saturation", "near_zero", "denormal", "signed_zero"]
)
def test_pointwise_cuda_bitwise(cuda_device, hidden, case):
    """First falsifier: identical precomputed GEMMs, no matmul error mixing."""
    from vllm_omni.model_executor.models.nemotron_asr.rnnt_cell_kernel import lstm_pointwise

    torch.manual_seed(91)
    batch = 3
    input_mm = torch.randn(batch, 4 * hidden, device=cuda_device)
    hidden_mm = torch.randn_like(input_mm)
    bias_ih = torch.randn(4 * hidden, device=cuda_device)
    bias_hh = torch.randn_like(bias_ih)
    cell = torch.randn(batch, hidden, device=cuda_device)
    if case == "cancellation":
        input_mm.fill_(2**24)
        hidden_mm.fill_(-(2**24))
        bias_ih.fill_(1.0)
        bias_hh.fill_(-0.25)
    elif case == "cell_cancellation":
        input_mm.copy_(torch.tensor([-0.375, 0.25, 0.625, 0.75], device=cuda_device).repeat_interleave(hidden))
        hidden_mm.zero_()
        bias_ih.zero_()
        bias_hh.zero_()
        i, f, g, _ = input_mm.chunk(4, dim=1)
        cell.copy_(-(i.sigmoid() * g.tanh()) / f.sigmoid())
    elif case == "saturation":
        values = torch.tensor([-1000.0, -100.0, -89.0, -20.0, -10.0, 10.0, 20.0, 89.0, 100.0, 1000.0])
        input_mm.copy_(values.repeat((input_mm.numel() + 9) // 10)[: input_mm.numel()].reshape_as(input_mm))
        hidden_mm.zero_()
        bias_ih.zero_()
        bias_hh.zero_()
    elif case == "near_zero":
        values = torch.tensor([-(2.0**-12), -(2.0**-24), -(2.0**-50), 2.0**-50, 2.0**-24, 2.0**-12])
        input_mm.copy_(values.repeat((input_mm.numel() + 5) // 6)[: input_mm.numel()].reshape_as(input_mm))
        hidden_mm.zero_()
        bias_ih.zero_()
        bias_hh.zero_()
    elif case == "denormal":
        # Integer construction retains subnormal and signed-zero bit patterns.
        bits = torch.tensor([1, -2147483647, 0x007FFFFF, -2139095041, 0x00800000, -2139095040])
        values = bits.to(torch.int32).view(torch.float32)
        input_mm.copy_(values.repeat((input_mm.numel() + 5) // 6)[: input_mm.numel()].reshape_as(input_mm))
        hidden_mm.zero_()
        bias_ih.zero_()
        bias_hh.zero_()
        cell.copy_(input_mm[:, :hidden])
    elif case == "signed_zero":
        input_mm.fill_(-0.0)
        hidden_mm.fill_(-0.0)
        bias_ih.fill_(-0.0)
        bias_hh.fill_(-0.0)
        cell.fill_(-0.0)
    # Include noncontiguous c; the inference API does not require packed state.
    cell = cell.t().contiguous().t()
    inputs = (input_mm, hidden_mm, bias_ih, bias_hh, cell)
    saved_inputs = [value.clone() for value in inputs]
    expected_gates = torch.empty(2, batch, 4 * hidden, device=cuda_device)
    actual_gates = torch.empty_like(expected_gates)
    expected = _pointwise_reference(*inputs, gate_values=expected_gates)
    actual = lstm_pointwise(*inputs, gate_values=actual_gates)
    normal = lstm_pointwise(*inputs)
    for name, got, want in zip(("h", "c", "gates"), (*actual, actual_gates), (*expected, expected_gates)):
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), f"{case}: {name} bit mismatch"
    for got, want in zip(normal, expected):
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))
    for got, want in zip(inputs, saved_inputs):
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@pytest.mark.cpu
def test_pointwise_oracle_detects_association_and_zero_sign():
    input_mm = torch.full((1, 4), 2.0**24)
    hidden_mm = -input_mm
    bias_ih = torch.ones(4)
    bias_hh = torch.full((4,), -0.25)
    assert not torch.equal(input_mm + bias_ih + hidden_mm + bias_hh, input_mm + hidden_mm + bias_ih + bias_hh)
    plus, minus = torch.tensor([0.0]), torch.tensor([-0.0])
    assert torch.equal(plus, minus)
    assert not torch.equal(plus.view(torch.int32), minus.view(torch.int32))


@pytest.mark.cuda
def test_pointwise_cuda_fallback_guards(cuda_device):
    h = torch.zeros(2, 3, 7, device=cuda_device)
    assert torch.is_grad_enabled()
    assert not rnnt_cell._use_fused_pointwise(h, h)
    with torch.no_grad():
        assert rnnt_cell._use_fused_pointwise(h, h)
        for dtype in (torch.float64, torch.bfloat16, torch.float16):
            other = h.to(dtype)
            assert not rnnt_cell._use_fused_pointwise(other, other)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert not rnnt_cell._use_fused_pointwise(h, h)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_pointwise_fallback_preserves_gradients(dtype):
    torch.manual_seed(134)
    model = ManualLSTM(input_size=5, hidden_size=7, num_layers=2).to(dtype)
    x = torch.randn(3, 5, dtype=dtype, requires_grad=True)
    h = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
    c = torch.randn_like(h, requires_grad=True)
    y, (nh, nc) = model.step(x, (h, c))
    expected_y, (expected_h, expected_c) = _step_reference(model, x, (h, c))
    args = (x, h, c, *model.parameters())
    got = torch.autograd.grad(y.sum() + nh.sum() + nc.sum(), args)
    want = torch.autograd.grad(expected_y.sum() + expected_h.sum() + expected_c.sum(), args)
    for actual, expected in zip(got, want):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "device", [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_pointwise_recurrent_bits_and_retained_outputs(batch, compute_dtype, device, request):
    if device == "cuda":
        request.getfixturevalue("cuda_device")
    torch.manual_seed(339)
    # Actual predictor width; two layers preserve the inter-layer dependency.
    model = ManualLSTM(input_size=640, hidden_size=640, num_layers=2).to(device=device, dtype=compute_dtype)
    state = (torch.randn(2, batch, 640, device=device), torch.randn(2, batch, 640, device=device))
    ref_state = tuple(t.clone() for t in state)
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        if device == "cuda":
            assert rnnt_cell._use_fused_pointwise(*state), "Must exercise the candidate, not fallback"
        for _ in range(4):
            x = torch.randn(batch, 640, device=device, dtype=compute_dtype)
            inputs = (x, *state)
            snapshots = [t.clone() for t in inputs]
            y, next_state = model.step(x, state)
            ref_y, ref_state = _step_reference(model, x, ref_state)
            outputs = (y, *next_state)
            for got, want in zip(outputs, (ref_y, *ref_state)):
                assert torch.equal(got.view(torch.int32), want.view(torch.int32))
            for got, want in zip(inputs, snapshots):
                assert torch.equal(got.view(torch.int32), want.view(torch.int32))
            assert len({t.untyped_storage().data_ptr() for t in outputs}) == 3
            assert not {t.untyped_storage().data_ptr() for t in outputs}.intersection(
                t.untyped_storage().data_ptr() for t in inputs
            )
            for old, snapshot in saved:
                assert torch.equal(old.view(torch.int32), snapshot.view(torch.int32))
                assert old.untyped_storage().data_ptr() not in {t.untyped_storage().data_ptr() for t in outputs}
            saved.extend((t, t.clone()) for t in outputs)
            state = next_state
