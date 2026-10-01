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

The conditional node uses the low-level methods inherited by
``torch.cuda.CUDAGraph`` from ``torch._C._CUDAGraph`` in PyTorch 2.13:
``get_currently_capturing_graph``, ``begin_capture_to_if_node`` and
``end_capture_to_conditional_node``. These exposed, undocumented bindings are
version-sensitive; opt-in preparation requires all three and a CUDA build of
12.4 or newer. Source examined at PyTorch
``cf30153c4c131c8164ee7798e5022d810682e2cb`` (v2.13.0):
``torch/csrc/cuda/Graph.cpp`` and ``aten/src/ATen/cuda/CUDAGraph.cpp``.

PyTorch owns the conditional kernel, child stream and allocator capture routing.
It rejects ``graph_capture_record_stream_reuse=True`` and RNG in a conditional
body; these errors propagate without dense fallback. Each graph owner retains
its predicate scalars and merge buffers. The scalar reduction adds a kernel relative to the previous
custom predicate. Tensor arithmetic follows ``decode_dense_masked_frames``.

Eager calls remain the dense preparation/oracle path. Capturing calls require
an explicit graph owner. This is a distinct experimental candidate; native
allocator routing has not been shown to explain the earlier replay failure.

Tail merge copies use two dtype-homogeneous ``torch._foreach_copy_`` calls per
frame. Metadata guards require a contiguous subset of the CUDA fast route in
the same PyTorch commit's ``aten/src/ATen/native/cuda/ForeachBinaryOpList.cu``
and ``aten/src/ATen/native/ForeachUtils.h``. Unsupported layouts or overlapping
writes raise before either copy; no per-field fallback is provided.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    MAX_SYMBOLS_PER_STEP,
    DecodeState,
    FrameAlignedDecode,
    Joint,
    decode_dense_masked_frames,
)

PREFIX_ATTEMPTS = 4
_FLOAT_MERGES = (0, 1, 3, 4, 5)
_INTEGER_MERGES = (2, 6)

logger = logging.getLogger(__name__)


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


class ConditionalCapture:
    """Resources belonging to one enclosing captured graph, never the process."""

    def __init__(self) -> None:
        self.tensors: list[torch.Tensor] = []
        self._graph: Any = None
        self._if_nodes = 0

    @property
    def if_nodes(self) -> int:
        return self._if_nodes

    def keep(self, *tensors: torch.Tensor | None) -> None:
        self.tensors.extend(tensor for tensor in tensors if tensor is not None)

    def run_if(self, active: torch.Tensor, body_fn: Callable[[], None]) -> None:
        graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
        if self._graph is None:
            self._graph = graph
        elif self._graph is not graph:
            raise RuntimeError("conditional-tail owner cannot cross parent graphs")
        # Capture this scalar reduction on the parent stream on every replay.
        # Native begin/end own child-stream and allocator-filter restoration.
        predicate = active.any()
        self.keep(active, predicate)
        graph.begin_capture_to_if_node(predicate)
        try:
            body_fn()
        except BaseException:
            try:
                graph.end_capture_to_conditional_node()
            except BaseException:
                logger.exception("Native conditional capture cleanup failed after body failure")
            raise
        else:
            graph.end_capture_to_conditional_node()
            self._if_nodes += 1


def _copy_tail_merges(
    destinations: tuple[torch.Tensor, ...], sources: tuple[torch.Tensor, ...], *, final_frame: bool
) -> None:
    """Copy disjoint merge buffers through the known foreach CUDA fast route.

    CPU mathematical tests also use this operation; they prove copy semantics,
    not CUDA launch counts. Validation reads metadata only and happens during
    capture, before either dtype group can write a destination.
    """
    if len(destinations) != 7 or len(sources) != 7:
        raise ValueError("conditional-tail merge requires seven fields")
    groups = ((torch.float32, (0, 1) if final_frame else _FLOAT_MERGES), (torch.int64, _INTEGER_MERGES))
    device = destinations[0].device
    destination_ranges: list[tuple[int, int]] = []
    source_ranges: list[tuple[int, int]] = []
    for dtype, indices in groups:
        for index in indices:
            destination, source = destinations[index], sources[index]
            for tensor in (destination, source):
                if tensor.dtype != dtype or tensor.device != device or tensor.layout != torch.strided:
                    raise ValueError("conditional-tail merge requires matching FP32/int64 fields on one device")
                if not tensor.is_contiguous() or tensor.numel() == 0 or tensor.is_conj() or tensor.is_neg():
                    raise ValueError("conditional-tail merge requires nonempty contiguous materialized fields")
            if destination.shape != source.shape or destination.stride() != source.stride():
                raise ValueError("conditional-tail merge requires matching pair shapes and strides")
            for tensor, ranges in ((destination, destination_ranges), (source, source_ranges)):
                start = tensor.data_ptr()
                ranges.append((start, start + tensor.numel() * tensor.element_size()))
    destination_ranges.sort()
    if any(left[1] > right[0] for left, right in zip(destination_ranges, destination_ranges[1:])) or any(
        destination[0] < source[1] and source[0] < destination[1]
        for destination in destination_ranges
        for source in source_ranges
    ):
        raise ValueError("conditional-tail merge destinations must be disjoint from each other and all sources")
    for _, indices in groups:
        torch._foreach_copy_([destinations[index] for index in indices], [sources[index] for index in indices])


def _prewarm_tail_merges(device: torch.device) -> None:
    # Both same-dtype paths are exercised once per decoder before graph capture.
    # These temporary tensors do not become graph-owner or decoder state.
    destinations = tuple(
        torch.empty(1, dtype=torch.float32 if index in _FLOAT_MERGES else torch.int64, device=device)
        for index in range(7)
    )
    sources = tuple(torch.zeros_like(tensor) for tensor in destinations)
    _copy_tail_merges(destinations, sources, final_frame=False)
    torch.accelerator.synchronize(device)


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
                # Final-frame lookahead fields 3/4/5 have no consumer.
                _copy_tail_merges(prefix[:7], result[:7], final_frame=t + 1 == t_pad)
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
        required = (
            "get_currently_capturing_graph",
            "begin_capture_to_if_node",
            "end_capture_to_conditional_node",
        )
        if any(not callable(getattr(torch.cuda.CUDAGraph, name, None)) for name in required):
            raise RuntimeError("conditional-tail requires the native PyTorch 2.13 conditional graph API")
        cuda_version = torch.version.cuda
        if cuda_version is None or tuple(int(part) for part in cuda_version.split(".")[:2]) < (12, 4):
            raise RuntimeError("conditional-tail requires a PyTorch CUDA build of 12.4 or newer")
        if not callable(getattr(torch, "_foreach_copy_", None)):
            raise RuntimeError("conditional-tail requires torch._foreach_copy_")
        if self._device is None:
            _prewarm_tail_merges(device)
        self._device = device

    @contextmanager
    def capture_scope(self) -> Iterator[ConditionalCapture]:
        if self._device is None:
            raise RuntimeError("conditional-tail decoder was not prepared before capture")
        if self._capture is not None:
            raise RuntimeError("conditional-tail capture scopes cannot overlap")
        # The caller holds this scope across its graph wrapper call, including
        # the enclosing graph's capture_end, not merely the IF child body.
        with _defer_automatic_gc():
            owner = ConditionalCapture()
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
