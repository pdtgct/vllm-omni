# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: N803
"""Experimental public-core persistent matmul with IEEE FP32 dot.

Kernel copied from vllm/model_executor/determinism/batch_invariant.py at
98dff2a81d747d1dba01a47f939f48c3526d4206 (Apache-2.0). It removes
optional launch metadata and requests input_precision="ieee" explicitly.
The experiment fixes the public core's FP32 128x128x32 tiling. Device state
is resolved at startup, before graph capture; no global override is installed.
"""

import torch
from vllm.triton_utils import tl, triton

CORE_COMMIT = "98dff2a81d747d1dba01a47f939f48c3526d4206"
CORE_SOURCE_SHA256 = "347cdf3deb9825ec6f9123674f98f3010d842d83791db323857da8ba9b1f3fd7"
CONFIG = dict(
    BLOCK_SIZE_M=128,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=32,
    GROUP_SIZE_M=8,
    num_stages=3,
    num_warps=8,
)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


class CoreIEEEProjection:
    """Device-bound fixed-tile callable, constructed outside torch.compile."""

    def __init__(self, device: torch.device) -> None:
        self.programs_per_sm = 1
        self.config = dict(CONFIG)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("core-ieee projection requires CUDA")
        self.device_index = torch.accelerator.current_device_index() if device.index is None else device.index
        self.num_sms = torch.cuda.get_device_properties(self.device_index).multi_processor_count
        self.program_budget = self.num_sms * self.programs_per_sm

    def _matmul(self, x, weight, *, capture=False):
        if x.dtype != torch.float32 or weight.dtype != torch.float32:
            raise ValueError("core-ieee projection requires FP32")
        if x.device.type != "cuda" or x.device.index != self.device_index or weight.device != x.device:
            raise ValueError("core-ieee projection device mismatch")
        if x.shape[-1] != weight.shape[1]:
            raise ValueError("core-ieee projection width mismatch")
        a = x.reshape(-1, x.shape[-1])
        b = weight.t()
        m, k = a.shape
        n = b.shape[1]
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        grid = (
            min(
                self.program_budget,
                triton.cdiv(m, self.config["BLOCK_SIZE_M"]) * triton.cdiv(n, self.config["BLOCK_SIZE_N"]),
            ),
        )
        compiled = matmul_kernel_persistent[grid](
            a,
            b,
            out,
            None,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            NUM_SMS=self.program_budget,
            A_LARGE=a.numel() > 2**31,
            B_LARGE=b.numel() > 2**31,
            C_LARGE=out.numel() > 2**31,
            HAS_BIAS=False,
            **self.config,
        )
        result = out.reshape(x.shape[:-1] + (n,))
        return (result, compiled) if capture else result

    def __call__(self, x, weight, bias=None):
        output = self._matmul(x, weight)
        # Match core linear_batch_invariant: bias is added after matmul.
        return output if bias is None else output + bias
