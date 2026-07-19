# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA qualification bar for the predicated resident-page commit."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.advance import (
    warmup_advance_model_rows_scatter,
)
from vllm_omni.model_executor.models.nemotron_asr.state_scatter import (
    masked_page_scatter_,
    validate_masked_page_scatter,
    warmup_masked_page_scatter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.cuda]


# @spec PORT-ADV-004, PORT-STATE-008
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_cuda_masked_scatter_matches_cpu_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(17)
    cpu_pool = torch.arange(9 * 3 * 5).reshape(9, 3, 5).to(dtype)
    cpu_scratch = torch.randint(-40, 40, (6, 3, 5)).to(dtype)
    blocks = torch.tensor([8, 2, 6, 1, 7, 4], dtype=torch.long)
    status = torch.tensor([0, 1, 0, 8, 0, 16], dtype=torch.int32)
    expected = cpu_pool.clone()
    for row in range(6):
        if int(status[row]) == 0:
            expected[int(blocks[row])].copy_(cpu_scratch[row])

    pool = cpu_pool.cuda()
    scratch = cpu_scratch.cuda()
    warmup_masked_page_scatter(pool, scratch)
    masked_page_scatter_(
        pool,
        scratch,
        blocks.cuda(),
        status.cuda(),
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(pool.cpu(), expected, rtol=0, atol=0)


# @spec PORT-STATE-008, PORT-PERF-001
def test_cuda_scatter_handles_padded_pool_stride_and_dirty_null_rows() -> None:
    backing = torch.arange(10 * 37, device="cuda", dtype=torch.float32)
    pool = torch.as_strided(backing, (10, 4, 7), (37, 7, 1))
    before = pool.clone()
    scratch = torch.full((5, 4, 7), 23.0, device="cuda")
    blocks = torch.tensor([9, 0, 5, 0, 0], device="cuda")
    status = torch.tensor([0, 1, 0, 2, 4], device="cuda", dtype=torch.int32)

    warmup_masked_page_scatter(pool, scratch)
    masked_page_scatter_(pool, scratch, blocks, status)
    torch.accelerator.synchronize()

    torch.testing.assert_close(pool[9], scratch[0], rtol=0, atol=0)
    torch.testing.assert_close(pool[5], scratch[2], rtol=0, atol=0)
    torch.testing.assert_close(pool[0], before[0], rtol=0, atol=0)


# @spec PORT-ADV-004, PORT-PERF-001
def test_cuda_scatter_captures_and_replays_with_changed_values() -> None:
    pool = torch.zeros(8, 16, device="cuda")
    scratch = torch.ones(4, 16, device="cuda")
    blocks = torch.tensor([7, 5, 3, 1], device="cuda")
    status = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    warmup_masked_page_scatter(pool, scratch)
    masked_page_scatter_(pool, scratch, blocks, status)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        masked_page_scatter_(pool, scratch, blocks, status)

    pool.zero_()
    scratch.fill_(9)
    blocks.copy_(torch.tensor([6, 4, 2, 0], device="cuda"))
    status.copy_(torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.accelerator.synchronize()

    torch.testing.assert_close(pool[4], scratch[1], rtol=0, atol=0)
    torch.testing.assert_close(pool[0], scratch[3], rtol=0, atol=0)
    torch.testing.assert_close(pool[6], torch.zeros_like(pool[6]), rtol=0, atol=0)
    torch.testing.assert_close(pool[2], torch.zeros_like(pool[2]), rtol=0, atol=0)


# @spec PORT-PERF-001, PORT-STATE-008
def test_cuda_scatter_rejects_unwarmed_or_late_specialization() -> None:
    pool = torch.zeros(5, 19, device="cuda")
    scratch = torch.ones(2, 19, device="cuda")
    blocks = torch.tensor([1, 2], device="cuda")
    status = torch.zeros(2, device="cuda", dtype=torch.int32)
    before = pool.clone()
    with pytest.raises(ValueError, match="not warmed"):
        masked_page_scatter_(pool, scratch, blocks, status)
    torch.testing.assert_close(pool, before, rtol=0, atol=0)

    warmup_masked_page_scatter(pool, scratch)
    masked_page_scatter_(pool, scratch, blocks, status)
    torch.accelerator.synchronize()
    torch.testing.assert_close(pool[1], scratch[0], rtol=0, atol=0)

    late_pool = torch.zeros(5, 23, device="cuda")
    late_scratch = torch.ones(2, 23, device="cuda")
    late_before = late_pool.clone()
    with pytest.raises(ValueError, match="not warmed"):
        masked_page_scatter_(late_pool, late_scratch, blocks, status)
    torch.testing.assert_close(late_pool, late_before, rtol=0, atol=0)


# @spec PORT-PERF-001, PORT-STATE-008
@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((56, 1024), torch.float32),
        ((1024, 8), torch.float32),
        ((2, 640), torch.float32),
        ((128, 9), torch.float32),
        ((1,), torch.int32),
        ((7,), torch.int32),
        ((8,), torch.int64),
    ],
)
def test_cuda_scatter_production_page_shapes(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    rows = 17
    pool = torch.zeros((32, *shape), device="cuda", dtype=dtype)
    scratch = (
        torch.arange(
            rows * math.prod(shape),
            device="cuda",
            dtype=torch.int64,
        )
        .reshape(rows, *shape)
        .to(dtype)
    )
    blocks = torch.arange(1, rows + 1, device="cuda")
    status = torch.zeros(rows, device="cuda", dtype=torch.int32)
    status[1::3] = 1
    before = pool.clone()
    warmup_masked_page_scatter(pool, scratch)
    masked_page_scatter_(pool, scratch, blocks, status)
    torch.accelerator.synchronize()
    for row in range(rows):
        expected = scratch[row] if int(status[row]) == 0 else before[row + 1]
        torch.testing.assert_close(pool[row + 1], expected, rtol=0, atol=0)


# @spec PORT-PERF-001, PORT-STATE-008
def test_transaction_startup_warms_complete_resident_inventory() -> None:
    pools = {
        "channel_pools": [torch.zeros(4, 8, 32, device="cuda")],
        "time_pools": [torch.zeros(4, 32, 4, device="cuda")],
        "len_pools": [torch.zeros(4, 1, dtype=torch.int32, device="cuda")],
        "h_pool": torch.zeros(4, 2, 16, device="cuda"),
        "c_pool": torch.zeros(4, 2, 16, device="cuda"),
        "queue_pool": torch.zeros(4, 48, dtype=torch.int32, device="cuda"),
        "book_pool": torch.zeros(4, 7, dtype=torch.int32, device="cuda"),
        "frontend_raw_pool": torch.zeros(4, 1_953, device="cuda"),
        "frontend_mel_pool": torch.zeros(4, 128, 9, device="cuda"),
        "frontend_counter_pool": torch.zeros(4, 8, dtype=torch.int64, device="cuda"),
    }
    warmup_advance_model_rows_scatter(**pools)
    inventory = [
        *pools["channel_pools"],
        *pools["time_pools"],
        *pools["len_pools"],
        pools["h_pool"],
        pools["c_pool"],
        pools["queue_pool"],
        pools["book_pool"],
        pools["frontend_raw_pool"],
        pools["frontend_mel_pool"],
        pools["frontend_counter_pool"],
    ]
    for pool in inventory:
        scratch = torch.empty((1, *pool.shape[1:]), dtype=pool.dtype, device=pool.device)
        validate_masked_page_scatter(
            pool,
            scratch,
            torch.ones(1, dtype=torch.int64, device=pool.device),
            torch.ones(1, dtype=torch.int32, device=pool.device),
        )


# @spec PORT-PERF-001
def test_cuda_scatter_custom_op_is_fullgraph_value_stable() -> None:
    pool = torch.zeros(8, 31, device="cuda")
    scratch = torch.ones(4, 31, device="cuda")
    blocks = torch.tensor([7, 5, 3, 1], device="cuda")
    status = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    warmup_masked_page_scatter(pool, scratch)
    compile_count = 0

    def backend(graph: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> Any:
        del inputs
        nonlocal compile_count
        compile_count += 1
        return graph.forward

    def commit(
        resident: torch.Tensor,
        proposed: torch.Tensor,
        indices: torch.Tensor,
        failures: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.vllm.nemotron_asr_masked_page_scatter_(resident, proposed, indices, failures)
        return resident

    compiled = torch.compile(commit, backend=backend, fullgraph=True)
    compiled(pool, scratch, blocks, status)
    blocks.copy_(torch.tensor([6, 4, 2, 0], device="cuda"))
    status.copy_(torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32))
    compiled(pool, scratch, blocks, status)
    torch.accelerator.synchronize()
    assert compile_count == 1


# @spec PORT-PERF-001
def test_cuda_scatter_custom_op_compiles_with_inductor() -> None:
    pool = torch.zeros(8, 31, device="cuda")
    scratch = torch.ones(4, 31, device="cuda")
    blocks = torch.tensor([7, 5, 3, 1], device="cuda")
    status = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    warmup_masked_page_scatter(pool, scratch)

    def commit(
        resident: torch.Tensor,
        proposed: torch.Tensor,
        indices: torch.Tensor,
        failures: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.vllm.nemotron_asr_masked_page_scatter_(
            resident,
            proposed,
            indices,
            failures,
        )
        return resident

    compiled = torch.compile(commit, fullgraph=True)
    compiled(pool, scratch, blocks, status)
    blocks.copy_(torch.tensor([6, 4, 2, 0], device="cuda"))
    status.copy_(torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32))
    compiled(pool, scratch, blocks, status)
    torch.accelerator.synchronize()
    torch.testing.assert_close(pool[4], scratch[1], rtol=0, atol=0)
    torch.testing.assert_close(pool[0], scratch[3], rtol=0, atol=0)
