# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pure dense-decoder selections; compilation is owned by the graph binding."""

from collections.abc import Callable
from typing import NamedTuple

import torch


def select_committed(
    gate: torch.Tensor,
    pred_h: torch.Tensor,
    h: torch.Tensor,
    pred_c: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.where(gate, pred_h, h), torch.where(gate, pred_c, c)


def select_predicted(
    mask: torch.Tensor,
    gate: torch.Tensor,
    new_out: torch.Tensor,
    pred_out: torch.Tensor,
    new_h: torch.Tensor,
    pred_h: torch.Tensor,
    new_c: torch.Tensor,
    pred_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.where(mask, new_out, pred_out),
        torch.where(gate, new_h, pred_h),
        torch.where(gate, new_c, pred_c),
    )


class DenseSelections(NamedTuple):
    committed: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    predicted: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


EAGER_SELECTIONS = DenseSelections(select_committed, select_predicted)


def compile_dense_selections() -> DenseSelections:
    """Create one static helper pair for the lifetime of one graph binding.

    Each batch tier warms in the binding's inference context before its
    capture. Torch 2.13's isolated regions give each binding its own default
    recompile budget even though the helper Python code is shared. The global
    accumulated safety limit remains in force; compilation errors propagate.
    Inductor must not create nested CUDA graphs or borrow its output buffers.
    """
    return DenseSelections(
        torch.compile(
            select_committed,
            fullgraph=True,
            dynamic=False,
            isolate_recompiles=True,
            options={"triton.cudagraphs": False},
        ),
        torch.compile(
            select_predicted,
            fullgraph=True,
            dynamic=False,
            isolate_recompiles=True,
            options={"triton.cudagraphs": False},
        ),
    )
