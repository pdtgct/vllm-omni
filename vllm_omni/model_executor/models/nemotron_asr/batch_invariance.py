# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned vLLM installation adapter and Nemotron accumulation contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class BatchInvariantExecution:
    """@spec PORT-PREC-019, PORT-PREC-025, PORT-PREC-028: frozen identity axis."""

    enabled: bool

    def fingerprint(self, baseline: Mapping[str, Any]) -> dict[str, Any]:
        return {**baseline, **({"batch_invariant_mode": "on"} if self.enabled else {})}

    readiness_receipt = fingerprint
    service_receipt = fingerprint

    def graph_key(self, kind: str, key: tuple[Any, ...]) -> tuple[Any, ...]:
        return ("batch_invariant_mode=on", kind, key) if self.enabled else key


MODE_OFF = BatchInvariantExecution(False)
_install_lock = RLock()
_installed: BatchInvariantExecution | None = None
_install_failure: Exception | None = None


def installed_batch_invariant_mode(
    *, module: Any = None, initialize: Callable[[], None] | None = None
) -> BatchInvariantExecution:
    """@spec PORT-PREC-018: require successful initialization, not just a library.

    The production witness is process-local and sticky, including failure:
    vLLM sets its flag before registering operators, so retrying a failed
    install could otherwise mistake its early return for successful completion.
    Explicit dependencies give probes the same validation without global state.
    """
    global _installed, _install_failure
    if module is not None or initialize is not None:
        if module is None or initialize is None:
            raise ValueError("module and initialize must be supplied together")
        return _snapshot_install(module, initialize)
    with _install_lock:
        if _install_failure is not None:
            raise _install_failure
        if _installed is None:
            from vllm.model_executor.determinism import batch_invariant

            try:
                _installed = _snapshot_install(batch_invariant, batch_invariant.init_batch_invariance)
            except Exception as error:
                _install_failure = error
                raise
        return _installed


def _snapshot_install(module: Any, initialize: Callable[[], None]) -> BatchInvariantExecution:
    initialize()
    flag = getattr(module, "_batch_invariant_MODE", None)
    if type(flag) is not bool or not hasattr(module, "_batch_invariant_LIB"):
        raise RuntimeError("vLLM batch-invariant installed state is absent")
    if flag and module._batch_invariant_LIB is None:
        raise RuntimeError("vLLM batch-invariant installation is incomplete")
    return BatchInvariantExecution(flag)


def execution_mode(owner: Any) -> BatchInvariantExecution:
    return getattr(owner, "batch_invariant_mode", MODE_OFF)


def bind_batch_invariant_mode(core: Any, record: BatchInvariantExecution) -> None:
    """@spec PORT-PREC-018, PORT-PREC-028: bind once, before any model work."""
    owners = [core]
    encoder = getattr(core, "encoder", None)
    if encoder is not None:
        owners.extend(encoder.modules())
    for owner in owners:
        previous = getattr(owner, "batch_invariant_mode", None)
        if previous is not None and previous != record:
            raise RuntimeError("batch-invariant mode cannot change on a loaded model")
    for owner in owners:
        if isinstance(getattr(type(owner), "batch_invariant_mode", None), property):
            owner._batch_invariant_mode = record
        else:
            owner.batch_invariant_mode = record


def resolve_execution_mode(
    record: BatchInvariantExecution, *, hf_overrides: Any = None, session_controls: Any = None
) -> BatchInvariantExecution:
    """@spec PORT-PREC-028: configuration is not an installed-mode selector."""
    return record


def resolve_optimizations(record: BatchInvariantExecution, selected: Mapping[str, Any]) -> dict[str, Any]:
    """@spec PORT-PREC-027: mode selection does not change optimizations."""
    return dict(selected)


def artifact_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """@spec PORT-PREC-025: match the entire execution/kernel identity."""
    return dict(expected) == dict(actual)


def operator_accumulation_contract(*, device_capability: tuple[int, int]) -> dict[str, Any]:
    """@spec PORT-PREC-021: obligations, separate from actual-backend evidence."""
    vllm_rows = {"gemm", "bmm", "mean"}
    names = (
        *sorted(vllm_rows),
        "softmax",
        "log_softmax",
        "layer_norm",
        "depthwise_conv1d",
        "strided_conv2d",
        "pointwise_conv",
        "sum",
        "sdpa",
    )
    rows = {
        name: {
            "owner": "vllm" if name in vllm_rows else "port",
            "required_accumulator": "float32",
            "fp16_accumulation": "unspecified",
            "accumulator_established": name in {"bmm", "mean"} or (name == "gemm" and device_capability[0] == 8),
        }
        for name in names
    }
    rows["pointwise_conv"]["bias_order"] = "before_result_cast"
    rows["softmax"]["reduction"] = "summation"
    return {"version": "operator-accumulation-v1", "rows": rows}


def accumulation_evidence_complete(operator: str, *, backend: str, execution: str, evidence: Mapping[str, Any]) -> bool:
    """@spec PORT-PREC-021: a value comparison is not accumulator evidence."""
    digest = evidence.get("source_sha256", "")
    return (
        evidence.get("operator") == operator
        and evidence.get("backend") == backend
        and evidence.get("execution") == execution
        and evidence.get("accumulator") == "float32"
        and evidence.get("kind")
        in ({"generated_kernel"} if execution == "compiled" else {"dispatch", "pinned_backend"})
        and isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        and bool(evidence.get("symbol"))
        and (operator != "pointwise_conv" or evidence.get("bias_before_cast") is True)
    )


def pointwise_conv_as_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """@spec PORT-PREC-021: k=1 GEMM with bias inside the accumulator."""
    if weight.ndim != x.ndim or any(size != 1 for size in weight.shape[2:]):
        raise ValueError("pointwise lowering requires a k=1 Conv1d or Conv2d weight")
    channels_last = x.movedim(1, -1)
    operand = channels_last.reshape(-1, x.shape[1])
    matrix = weight.reshape(weight.shape[0], weight.shape[1]).t()
    # CUDA uses vLLM's installed addmm accumulator. The CPU oracle has no
    # installed kernel, so explicitly widen before the biased reduction.
    if x.device.type == "cpu":
        operand, matrix = operand.float(), matrix.float()
        bias = None if bias is None else bias.float()
    result = torch.mm(operand, matrix) if bias is None else torch.addmm(bias, operand, matrix)
    return result.to(x.dtype).reshape(*channels_last.shape[:-1], weight.shape[0]).movedim(-1, 1)


def sum_fp32(x: torch.Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> torch.Tensor:
    """@spec PORT-PREC-021: widen before summation, restore the activation dtype."""
    return x.float().sum(dim=dim, keepdim=keepdim).to(x.dtype)


def softmax_fp32_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """@spec PORT-PREC-021: exp rounding is distinct from summation width."""
    if x.shape[dim] == 0:
        return x.clone()
    exp = (x - x.amax(dim=dim, keepdim=True)).exp()
    return (exp.float() / exp.float().sum(dim=dim, keepdim=True)).to(x.dtype)


def log_softmax_fp32_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """@spec PORT-PREC-021: accumulate exp in FP32 before log normalization."""
    if x.shape[dim] == 0:
        return x.clone()
    shifted = x - x.amax(dim=dim, keepdim=True)
    return (shifted.float() - shifted.exp().float().sum(dim=dim, keepdim=True).log()).to(x.dtype)


def layer_norm_fp32(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """@spec PORT-PREC-021: normalization and affine precede the result cast."""
    return F.layer_norm(
        x.float(),
        tuple(normalized_shape),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    ).to(x.dtype)


def depthwise_conv1d_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """@spec PORT-PREC-021: depthwise taps reduce in FP32, independently per row."""
    taps = weight.shape[-1]
    windows = F.pad(x, (padding, padding)).unfold(-1, dilation * (taps - 1) + 1, stride)[..., ::dilation]
    result = (windows.float() * weight[:, 0, :].float()[None, :, None, :]).sum(-1)
    if bias is not None:
        result = result + bias.float()[None, :, None]
    return result.to(x.dtype)


def strided_conv2d_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """@spec PORT-PREC-020, PORT-PREC-021: fixed spatial contraction on CUDA.

    cuDNN can choose a different convolution algorithm as population changes.
    Lower CUDA subsampling to the installed GEMM (or a depthwise tap sum),
    with FP32 operands and one result cast. CPU convolution is the oracle.
    """
    if x.device.type != "cuda":
        return F.conv2d(
            x.float(),
            weight.float(),
            None if bias is None else bias.float(),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        ).to(x.dtype)
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    kh, kw = weight.shape[-2:]
    height = (x.shape[-2] + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    width = (x.shape[-1] + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    patches = F.unfold(x.float(), (kh, kw), dilation=dilation, padding=padding, stride=stride)
    batch, _, positions = patches.shape
    if groups == x.shape[1] == weight.shape[0]:
        taps = patches.reshape(batch, groups, kh * kw, positions).transpose(-1, -2)
        result = (taps * weight.float().reshape(groups, 1, kh * kw)).sum(-1)
        if bias is not None:
            result = result + bias.float()[None, :, None]
    elif groups == 1:
        operand = patches.transpose(1, 2).reshape(batch * positions, -1)
        matrix = weight.float().flatten(1).t()
        result = torch.mm(operand, matrix) if bias is None else torch.addmm(bias.float(), operand, matrix)
        result = result.reshape(batch, positions, weight.shape[0]).transpose(1, 2)
    else:
        raise ValueError("subsampling requires dense or depthwise convolution")
    return result.reshape(batch, weight.shape[0], height, width).to(x.dtype)


@dataclass(frozen=True)
class QualificationVerdict:
    ready: bool
    qualified: bool
    default: bool = False
    supported: bool = False


def qualification_verdict(
    fingerprint: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    served_geometries: Sequence[int],
    deployment_cap: int,
) -> QualificationVerdict:
    """@spec PORT-PREC-020, PORT-PREC-022, PORT-PREC-023, PORT-PREC-024,
    PORT-PREC-026, PORT-PREC-027: accept only a complete, identity-bound envelope.

    Gates are the lab's verified evidence attestations, not inferred from mode
    selection. The deployment cap never reduces the B128 qualification envelope.
    """
    ready = evidence.get("executable") is True and not evidence.get("execution_failure")
    gates = evidence.get("gates", {})
    qualified = (
        ready
        and artifact_matches(fingerprint, evidence.get("fingerprint", {}))
        and fingerprint.get("batch_invariant_mode") == "on"
        and str(fingerprint.get("gpu", "")).split("-")[0] in {"NVIDIA A100", "NVIDIA A40"}
        and {1, 2, 31, 63, 64, 128} <= set(evidence.get("populations", ()))
        and bool(served_geometries)
        and set(served_geometries) <= set(evidence.get("geometries", ()))
        and {"repeated", "mixed"} <= set(evidence.get("compositions", ()))
        and evidence.get("independent_b128_runs", 0) >= 2
        and all(gates.get(f"PORT-PREC-{number:03}") is True for number in (14, 16, 20, 21, 22, 24, 26, 27))
    )
    return QualificationVerdict(ready=ready, qualified=qualified)
