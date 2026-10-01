# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental, opt-in IF after four static RNN-T attempts per frame.

The raw conditional-node insertion follows the tested project prototype and
public NVIDIA-NeMo/Speech ``nemo/core/utils/cuda_python_utils.py`` at
``de242add77945a110568c7c44bdae4891451851e`` (Apache-2.0). The predicate and
control flow here use one IF per frame, not the prototype's per-symbol WHILE.
Tensor arithmetic follows this package's ``decode_dense_masked_frames``.

Eager calls remain the dense preparation/oracle path. A capturing call requires
an explicit graph owner; capture failures propagate. Streams, arguments and
merge buffers belong to that graph owner.

The immutable predicate module has one cache slot per visible CUDA device for
its PyTorch-managed primary context's lifetime. There is no GC unload callback:
CUDA context/process destruction reclaims successful modules. Slots never hold
graphs or their buffers. Context replacement, module reload and a failed first
initialization require a worker restart; resetting a live PyTorch context is
unsupported. A unique CUDA context ID detects replacement even if its address
is reused. This trades fixed module/PTX/scalar-anchor retention per device for
safe graph-independent code lifetime, not an accumulated capture history.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    MAX_SYMBOLS_PER_STEP,
    DecodeState,
    FrameAlignedDecode,
    Joint,
    decode_dense_masked_frames,
)

PREFIX_ATTEMPTS = 4

_CONDITION_SOURCE = r"""
typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned int);
extern "C" __global__ void condition(cudaGraphConditionalHandle handle, const bool *active, int batch) {
    bool any = false;
    for (int i = 0; i < batch; ++i) any |= active[i];
    cudaGraphSetConditional(handle, any);
}
"""


def _checked(result: tuple[Any, ...]) -> tuple[Any, ...]:
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"conditional-tail CUDA operation failed: {error}")
    return tuple(values)


logger = logging.getLogger(__name__)


class _CompiledCondition:
    def __init__(self, *, context: Any, context_id: int, anchor: torch.Tensor) -> None:
        # Lazy imports keep default-off and CPU/eager callers CUDA-independent.
        from cuda.bindings import __version__ as bindings_version
        from cuda.bindings import driver, nvrtc, runtime

        if int(bindings_version.split(".")[0]) < 13:
            raise RuntimeError("conditional-tail capture requires CUDA Python bindings 13 or newer")
        self.driver, self.runtime = driver, runtime
        self.context, self.context_id, self._anchor = context, context_id, anchor
        (program,) = _checked(nvrtc.nvrtcCreateProgram(_CONDITION_SOURCE.encode(), b"tail_if.cu", 0, [], []))
        module = None
        try:
            try:
                result = nvrtc.nvrtcCompileProgram(program, 0, [])
                if int(result[0]):
                    (size,) = _checked(nvrtc.nvrtcGetProgramLogSize(program))
                    log = b" " * size
                    _checked(nvrtc.nvrtcGetProgramLog(program, log))
                    raise RuntimeError(f"conditional-tail NVRTC compile failed: {log.decode()}")
                (size,) = _checked(nvrtc.nvrtcGetPTXSize(program))
                ptx = b" " * size
                _checked(nvrtc.nvrtcGetPTX(program, ptx))
                self._ptx = np.frombuffer(ptx, dtype=np.uint8).copy()
                (module,) = _checked(driver.cuModuleLoadData(self._ptx.ctypes.data))
                (self.kernel,) = _checked(driver.cuModuleGetFunction(module, b"condition"))
            except BaseException:
                try:
                    _checked(nvrtc.nvrtcDestroyProgram(program))
                except BaseException:
                    logger.exception("NVRTC cleanup failed; conditional-tail device slot remains poisoned")
                raise
            else:
                _checked(nvrtc.nvrtcDestroyProgram(program))
        except BaseException:
            # Preparation is outside capture. A failed construction is never
            # cached as usable, and cannot abandon another module on a retry.
            if module is not None:
                try:
                    _checked(driver.cuModuleUnload(module))
                except BaseException:
                    logger.exception("Module cleanup failed; conditional-tail device slot remains poisoned")
            raise
        self.module = module

    def launch(self, arguments: np.ndarray, stream: Any) -> None:
        _checked(
            self.driver.cuLaunchKernel(self.kernel, 1, 1, 1, 1, 1, 1, 0, stream.cuda_stream, arguments.ctypes.data, 0)
        )


# One immutable slot per visible device; None poisons an unsuccessful first
# construction. Keep no old-context list and no graph/owner references here.
# CUDA documents cuCtxGetId as unique for the program lifetime and cuCtxDestroy
# as reclaiming CUmodule/CUfunction resources:
# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html
_MODULE_CACHE: dict[int, tuple[int, _CompiledCondition | None]] = {}
_MODULE_CACHE_LOCK = threading.Lock()


_CAPTURE_GC_LOCK = threading.Lock()
_CAPTURE_GC_USERS = 0
_CAPTURE_GC_WAS_ENABLED = False


@contextmanager
def _defer_automatic_gc() -> Iterator[None]:
    """Keep cyclic destruction outside the entire conditional graph capture.

    The matching-runtime failure is sensitive to automatic GC; its exact
    collected object is unknown. Explicit gc.collect remains available to the
    graph wrapper before capture. Count overlapping scopes because GC state is
    process-wide, and restore the caller's state only after the last one exits.
    """
    global _CAPTURE_GC_USERS, _CAPTURE_GC_WAS_ENABLED
    with _CAPTURE_GC_LOCK:
        if _CAPTURE_GC_USERS == 0:
            _CAPTURE_GC_WAS_ENABLED = gc.isenabled()
            gc.disable()
        _CAPTURE_GC_USERS += 1
    try:
        yield
    finally:
        with _CAPTURE_GC_LOCK:
            _CAPTURE_GC_USERS -= 1
            if _CAPTURE_GC_USERS == 0 and _CAPTURE_GC_WAS_ENABLED:
                gc.enable()


def _compiled_condition_for_device(device: torch.device) -> _CompiledCondition:
    from cuda.bindings import driver

    with _MODULE_CACHE_LOCK:
        # Establish the current runtime context before querying/loading driver
        # code. Only the first successful slot retains this scalar allocation.
        anchor = torch.empty((), device=device)
        (visible_devices,) = _checked(driver.cuDeviceGetCount())
        ordinal = device.index
        if ordinal is None or not 0 <= ordinal < int(visible_devices):
            raise ValueError("conditional-tail device is not a visible CUDA ordinal")
        (context_device,) = _checked(driver.cuCtxGetDevice())
        if int(context_device) != ordinal:
            raise ValueError("conditional-tail CUDA context belongs to another device")
        (context,) = _checked(driver.cuCtxGetCurrent())
        (context_id,) = _checked(driver.cuCtxGetId(context))
        identity = int(context_id)
        if ordinal in _MODULE_CACHE:
            saved_identity, compiled = _MODULE_CACHE[ordinal]
            if saved_identity != identity:
                raise RuntimeError("conditional-tail CUDA context changed; restart the worker")
            if compiled is None:
                raise RuntimeError("conditional-tail initialization previously failed; restart the worker")
            return compiled
        _MODULE_CACHE[ordinal] = (identity, None)
        compiled = _CompiledCondition(context=context, context_id=identity, anchor=anchor)
        _MODULE_CACHE[ordinal] = (identity, compiled)
        return compiled


class ConditionalCapture:
    """Resources belonging to one enclosing captured graph, never the process."""

    def __init__(self, compiled: _CompiledCondition) -> None:
        self.compiled = compiled
        self.tensors: list[torch.Tensor] = []
        self.nodes: list[tuple[Any, ...]] = []

    @property
    def if_nodes(self) -> int:
        return len(self.nodes)

    def keep(self, *tensors: torch.Tensor | None) -> None:
        self.tensors.extend(tensor for tensor in tensors if tensor is not None)

    def run_if(self, active: torch.Tensor, body_fn: Callable[[], None]) -> None:
        driver, runtime = self.compiled.driver, self.compiled.runtime
        parent = torch.cuda.current_stream(active.device)
        status, _, graph, *_ = _checked(runtime.cudaStreamGetCaptureInfo(parent.cuda_stream))
        if status != runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
            raise RuntimeError("conditional tail requires active CUDA stream capture")
        # The handle belongs to this graph. Reset it to zero at launch, and set
        # it again from THIS frame's freshly computed activity on every replay.
        (handle,) = _checked(runtime.cudaGraphConditionalHandleCreate(graph, 0, 1))
        scalars = [
            np.array([int(handle)], dtype=np.uint64),
            np.array([active.data_ptr()], dtype=np.uint64),
            np.array([active.numel()], dtype=np.int32),
        ]
        arguments = np.array([scalar.ctypes.data for scalar in scalars], dtype=np.uint64)
        self.compiled.launch(arguments, parent)
        _, _, graph, dependencies, *_ = _checked(runtime.cudaStreamGetCaptureInfo(parent.cuda_stream))
        params = driver.CUgraphNodeParams()
        params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        params.conditional.handle = handle
        params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
        params.conditional.size = 1
        params.conditional.ctx = self.compiled.context
        (node,) = _checked(driver.cuGraphAddNode(graph, dependencies, None, len(dependencies), params))
        body = params.conditional.phGraph_out[0]
        _checked(
            runtime.cudaStreamUpdateCaptureDependencies(
                parent.cuda_stream,
                [node],
                None,
                1,
                runtime.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
            )
        )
        stream = torch.cuda.Stream(device=active.device)
        self.nodes.append((handle, scalars, arguments, params, stream, active))
        begun = False
        try:
            _checked(
                runtime.cudaStreamBeginCaptureToGraph(
                    stream.cuda_stream,
                    body,
                    None,
                    None,
                    0,
                    runtime.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
                )
            )
            begun = True
            torch.cuda.set_stream(stream)
            body_fn()
            result = runtime.cudaStreamEndCapture(stream.cuda_stream)
            begun = False
            _checked(result)
        finally:
            if begun:
                try:
                    _checked(runtime.cudaStreamEndCapture(stream.cuda_stream))
                except Exception:
                    pass  # Preserve the original capture failure; no fallback.
            torch.cuda.set_stream(parent)


def _decode_frames_with_if(
    enc_frames: torch.Tensor,
    enc_lengths: torch.Tensor,
    predictor: Any,
    joint: Any,
    state: DecodeState,
    *,
    capture: ConditionalCapture,
    max_symbols: int = MAX_SYMBOLS_PER_STEP,
) -> FrameAlignedDecode:
    """Static prefix and fixed tail with one conditional merge per frame."""
    batch, t_pad, _ = enc_frames.shape
    device, blank = enc_frames.device, predictor.blank_id
    enc_lengths = enc_lengths.clamp(min=0, max=t_pad)
    h, c, last_label = state.h.clone(), state.c.clone(), state.last_label.clone()
    capacity = max(t_pad * max_symbols, 1)
    token_ids = torch.zeros(batch, t_pad * max_symbols, dtype=torch.int32, device=device)
    token_lengths = torch.zeros(batch, dtype=torch.long, device=device)
    counts = torch.zeros(batch, t_pad, dtype=torch.int32, device=device)
    finals = torch.full((batch, t_pad), blank, dtype=torch.int32, device=device)
    pred_out, (pred_h, pred_c) = predictor.step(last_label, (h, c))
    capture.keep(token_ids, counts, finals)
    for t in range(t_pad):
        frame = enc_frames[:, t]
        projected_frame = (
            joint.enc(frame.to(joint.enc.weight.dtype)) if type(joint) is Joint and max_symbols > 0 else None
        )

        def attempt(values: tuple[torch.Tensor, ...], symbol: int) -> tuple[torch.Tensor, ...]:
            h, c, label, pred, ph, pc, lengths, active = values
            if projected_frame is None:
                logits = joint.logits(frame, pred)
            else:
                logits = joint.joint_net(projected_frame + joint.pred(pred.to(joint.enc.weight.dtype)))
            labels = logits.argmax(dim=-1)
            emit = active & (labels != blank)
            idx = lengths.clamp(max=capacity - 1).unsqueeze(1)
            token_ids.scatter_(
                1,
                idx,
                torch.where(emit.unsqueeze(1), labels.unsqueeze(1).to(torch.int32), token_ids.gather(1, idx)),
            )
            lengths = lengths + emit.long()
            counts[:, t] += emit.to(torch.int32)
            finals[:, t] = torch.where(emit, labels.to(torch.int32), finals[:, t])
            gate = emit.view(1, -1, 1)
            label = torch.where(emit, labels, label)
            h, c = torch.where(gate, ph, h), torch.where(gate, pc, c)
            if t + 1 < t_pad or symbol + 1 < max_symbols:
                new_out, (new_h, new_c) = predictor.step(label, (h, c))
                pred = torch.where(emit.unsqueeze(-1), new_out, pred)
                ph, pc = torch.where(gate, new_h, ph), torch.where(gate, new_c, pc)
            return h, c, label, pred, ph, pc, lengths, emit

        values = (h, c, last_label, pred_out, pred_h, pred_c, token_lengths, t < enc_lengths)
        for symbol in range(min(PREFIX_ATTEMPTS, max_symbols)):
            values = attempt(values, symbol)
        if max_symbols > PREFIX_ATTEMPTS:
            # Prefix results are fresh, independently owned where/add outputs.
            # They initialize every merge on every replay WITHOUT extra prefix
            # copies. The child writes final values once; skipping leaves them
            # intact. No per-symbol copying or predicate launch is introduced.
            prefix = values
            capture.keep(*prefix, projected_frame)

            def tail() -> None:
                result = prefix
                for symbol in range(PREFIX_ATTEMPTS, max_symbols):
                    result = attempt(result, symbol)
                for index, (destination, source) in enumerate(zip(prefix[:7], result[:7], strict=True)):
                    if index in (3, 4, 5) and t + 1 == t_pad:
                        continue  # Final lookahead has no consumer.
                    destination.copy_(source)
                capture.keep(*result)

            capture.run_if(prefix[-1], tail)
        h, c, last_label, pred_out, pred_h, pred_c, token_lengths, _ = values
    return FrameAlignedDecode(
        token_ids=token_ids,
        token_lengths=token_lengths.to(torch.int32),
        state=DecodeState(h=h, c=c, last_label=last_label),
        frame_emission_counts=counts,
        frame_final_labels=finals,
    )


class ConditionalTailDecoder:
    """Inject into DenseGraphBinding; eager preparation remains the dense oracle."""

    @property
    def prefix_attempts(self) -> int:
        return PREFIX_ATTEMPTS

    def __init__(self) -> None:
        self._compiled: _CompiledCondition | None = None
        self._device: torch.device | None = None
        self._capture: ConditionalCapture | None = None

    def prepare(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("conditional-tail graph execution requires CUDA")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare conditional-tail resources before capture")
        current_device = torch.accelerator.current_device_index()
        device = torch.device("cuda", current_device if device.index is None else device.index)
        if device.index != current_device:
            raise ValueError("conditional-tail preparation requires the current CUDA device")
        if self._device is not None and self._device != device:
            raise ValueError("conditional-tail decoder cannot cross CUDA devices")
        # Validate live context identity even for an already prepared decoder.
        self._compiled = _compiled_condition_for_device(device)
        self._device = device

    @contextmanager
    def capture_scope(self) -> Iterator[ConditionalCapture]:
        if self._compiled is None:
            raise RuntimeError("conditional-tail decoder was not prepared before capture")
        if self._capture is not None:
            raise RuntimeError("conditional-tail capture scopes cannot overlap")
        # The caller holds this scope across its graph wrapper call, including
        # the enclosing graph's capture_end, not merely the IF child body.
        with _defer_automatic_gc():
            owner = ConditionalCapture(self._compiled)
            self._capture = owner
            try:
                yield owner
            finally:
                self._capture = None

    def __call__(
        self,
        enc_frames: torch.Tensor,
        enc_lengths: torch.Tensor,
        predictor: Any,
        joint: Any,
        state: DecodeState,
        *,
        max_symbols: int = MAX_SYMBOLS_PER_STEP,
    ) -> FrameAlignedDecode:
        if enc_frames.device.type != "cuda" or not torch.cuda.is_current_stream_capturing():
            return decode_dense_masked_frames(enc_frames, enc_lengths, predictor, joint, state, max_symbols=max_symbols)
        if self._capture is None:
            raise RuntimeError("capturing conditional-tail decoder requires a graph-owned capture scope")
        return _decode_frames_with_if(
            enc_frames,
            enc_lengths,
            predictor,
            joint,
            state,
            capture=self._capture,
            max_symbols=max_symbols,
        )
