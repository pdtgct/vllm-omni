# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only CUDA FP32 pointwise half of the manual predictor cell.

The two GEMMs remain PyTorch operations. This kernel retains the eager
association ((input_mm + bias_ih) + hidden_mm) + bias_hh, and rounds both
cell-update products before adding. Libdevice activation identity must be
validated against the target Torch/CUDA build, not inferred from their names.
"""

import torch
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _gate(
    input_mm, hidden_mm, bias_ih, bias_hh, row, col, mask, input_stride: tl.constexpr, hidden_stride: tl.constexpr
):
    a = tl.load(input_mm + row * input_stride + col, mask, other=0)
    b = tl.load(bias_ih + col)
    c = tl.load(hidden_mm + row * hidden_stride + col, mask, other=0)
    d = tl.load(bias_hh + col)
    return tldevice.add_rn(tldevice.add_rn(tldevice.add_rn(a, b), c), d)


@triton.jit
def _sigmoid(x):
    # Unlike tl.sigmoid, neither exp2 approximation nor approximate division.
    return tldevice.div_rn(1.0, tldevice.add_rn(1.0, tldevice.exp(-x)))


@triton.jit
def _lstm_pointwise_kernel(
    input_mm,
    hidden_mm,
    bias_ih,
    bias_hh,
    cell,
    out_h,
    out_c,
    gates,
    batch: tl.constexpr,
    hidden: tl.constexpr,
    input_stride: tl.constexpr,
    hidden_stride: tl.constexpr,
    cell_stride0: tl.constexpr,
    cell_stride1: tl.constexpr,
    save_gates: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    row, col = index // hidden, index % hidden
    mask = index < batch * hidden
    raw_i = _gate(input_mm, hidden_mm, bias_ih, bias_hh, row, col, mask, input_stride, hidden_stride)
    raw_f = _gate(input_mm, hidden_mm, bias_ih, bias_hh, row, col + hidden, mask, input_stride, hidden_stride)
    raw_g = _gate(input_mm, hidden_mm, bias_ih, bias_hh, row, col + 2 * hidden, mask, input_stride, hidden_stride)
    raw_o = _gate(input_mm, hidden_mm, bias_ih, bias_hh, row, col + 3 * hidden, mask, input_stride, hidden_stride)
    i, f, g, o = _sigmoid(raw_i), _sigmoid(raw_f), tldevice.tanh(raw_g), _sigmoid(raw_o)
    old_c = tl.load(cell + row * cell_stride0 + col * cell_stride1, mask, other=0)
    new_c = tldevice.add_rn(tldevice.mul_rn(f, old_c), tldevice.mul_rn(i, g))
    new_h = tldevice.mul_rn(o, tldevice.tanh(new_c))
    tl.store(out_h + index, new_h, mask)
    tl.store(out_c + index, new_c, mask)
    if save_gates:
        offset = row * (4 * hidden) + col
        tl.store(gates + offset, raw_i, mask)
        tl.store(gates + offset + hidden, raw_f, mask)
        tl.store(gates + offset + 2 * hidden, raw_g, mask)
        tl.store(gates + offset + 3 * hidden, raw_o, mask)
        offset += batch * 4 * hidden
        tl.store(gates + offset, i, mask)
        tl.store(gates + offset + hidden, f, mask)
        tl.store(gates + offset + 2 * hidden, g, mask)
        tl.store(gates + offset + 3 * hidden, o, mask)


def lstm_pointwise(
    input_mm: torch.Tensor,
    hidden_mm: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    cell: torch.Tensor,
    *,
    gate_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fresh h/c; optional gate output is solely an exactness diagnostic.

    Inputs are CUDA FP32, dense GEMM rows and contiguous bias vectors; cell
    may be strided. gate_values, when supplied, is contiguous [2, B, 4H]
    (preactivation and activated gates). No gate buffer exists in serving.
    """
    batch, hidden = cell.shape
    h_next = torch.empty((batch, hidden), dtype=cell.dtype, device=cell.device)
    c_next = torch.empty_like(h_next)
    if batch * hidden:
        _lstm_pointwise_kernel[(triton.cdiv(batch * hidden, 256),)](
            input_mm,
            hidden_mm,
            bias_ih,
            bias_hh,
            cell,
            h_next,
            c_next,
            gate_values,
            batch,
            hidden,
            input_mm.stride(0),
            hidden_mm.stride(0),
            cell.stride(0),
            cell.stride(1),
            gate_values is not None,
            256,
            num_warps=4,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return h_next, c_next
