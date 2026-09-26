# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded experimental capture of one already-gathered CHUNK bucket.

The caller constructs this binding before measured execution and keeps its model
immutable. Only the declared 160-ms B32/B64 cell is callable. This is a scoped
screen seam, not a serving selector or a full-population readiness claim.
"""

from collections.abc import Callable
from typing import Any

import torch

from vllm_omni.model_executor.models.nemotron_asr.advance import (
    ENVELOPE_HEADER_SLOTS,
    ChunkBatch,
    ChunkBucketResult,
    SessionStateBatch,
    advance_chunk_bucket,
)
from vllm_omni.model_executor.models.nemotron_asr.decode_graph import GraphRuntime, platform_graph_runtime
from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import _tensor_signature
from vllm_omni.model_executor.models.nemotron_asr.manifests import CADENCES
from vllm_omni.model_executor.models.nemotron_asr.rnnt import decode_dense_masked_frames


def _state_tensors(state: SessionStateBatch) -> tuple[torch.Tensor, ...]:
    return (
        state.raw_tail,
        state.mel_tail,
        state.frontend_counters,
        *state.channel,
        *state.window_valid,
        *state.time,
        state.h,
        state.c,
        state.last_label,
    )


def _clone_state(state: SessionStateBatch) -> SessionStateBatch:
    return SessionStateBatch(
        raw_tail=state.raw_tail.clone(),
        mel_tail=state.mel_tail.clone(),
        frontend_counters=state.frontend_counters.clone(),
        channel=[tensor.clone() for tensor in state.channel],
        window_valid=[tensor.clone() for tensor in state.window_valid],
        time=[tensor.clone() for tensor in state.time],
        h=state.h.clone(),
        c=state.c.clone(),
        last_label=state.last_label.clone(),
    )


def _outputs(bucket: ChunkBucketResult) -> tuple[torch.Tensor, ...]:
    result, batch = bucket.result, bucket.batch
    values = (
        result.token_ids,
        result.token_lengths,
        result.row_status,
        result.frame_emission_counts,
        result.frame_final_labels,
        result.frame_valid_lengths,
        batch.valid_samples,
        batch.geometry_id,
        batch.final_tail,
        batch.prompt_index,
        batch.chunk_sequence,
        bucket.counter_invariant_bad,
        bucket.counter_delta_bad,
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in values) or result.captures is not None:
        raise ValueError("bucket graph requires all dense frame outputs and capture-off execution")
    return values  # type: ignore[return-value]


@torch.inference_mode()
def capture_chunk_bucket(
    core: Any,
    env: torch.Tensor,
    state: SessionStateBatch,
    *,
    geometry: int,
    admitted_prompt: torch.Tensor,
    incoming_status: torch.Tensor,
    queue_capacity: int,
    vllm_config: Any,
    runtime: GraphRuntime | None = None,
) -> Callable[..., ChunkBucketResult]:
    """Capture a single explicit cell, preserving caller state and output lifetime.

    B32/B64 are exact decoder tiers, so raw dense execution keeps the split
    decoder's physical arithmetic shape. Unsupported shapes fail before staging.
    Fresh-aware resident gathers and every transaction commit remain outside.
    """
    if geometry != 1 or env.shape[0] not in (32, 64):
        raise ValueError("experimental bucket graph supports only 160-ms B32/B64")
    if getattr(core.encoder, "_stream_relative_position_lengths", ()):
        raise ValueError("bucket graph requires unprepared native positional projections")
    runtime = runtime or platform_graph_runtime()
    population = int(env.shape[0])
    sample_width = 8 * (list(CADENCES.values())[geometry][1] + 1) * int(core.featurizer.hop_length)
    static_env = env.clone()
    static_prompt = admitted_prompt.clone()
    static_status = incoming_status.clone()
    static_state = _clone_state(state)
    sources = (env, admitted_prompt, incoming_status, *_state_tensors(state))
    scratch = (static_env, static_prompt, static_status, *_state_tensors(static_state))
    signature = tuple(_tensor_signature(tensor) for tensor in sources)
    descriptor = runtime.descriptor_factory(population)
    result_type: Any = None
    owned: tuple[torch.Tensor, ...] = ()

    def stage(values: tuple[torch.Tensor, ...]) -> None:
        for destination, source in zip(scratch, values, strict=True):
            destination.copy_(source)

    def body() -> tuple[torch.Tensor, ...]:
        bucket = advance_chunk_bucket(
            core,
            static_env,
            static_state,
            geometry=geometry,
            admitted_prompt=static_prompt,
            incoming_status=static_status,
            queue_capacity=queue_capacity,
            decode_fn=decode_dense_masked_frames,
            capture=False,
        )
        for destination, source in zip(owned, _outputs(bucket), strict=True):
            destination.copy_(source)
        return owned

    # Discover layouts outside capture. Strong output buffers and state scratch
    # outlive the platform wrapper's weak output handles.
    probe = advance_chunk_bucket(
        core,
        static_env,
        static_state,
        geometry=geometry,
        admitted_prompt=static_prompt,
        incoming_status=static_status,
        queue_capacity=queue_capacity,
        decode_fn=decode_dense_masked_frames,
        capture=False,
    )
    result_type = type(probe.result)
    owned = tuple(torch.empty_like(tensor) for tensor in _outputs(probe))
    wrapper = runtime.wrapper_factory(body, vllm_config, runtime_mode=runtime.graph_mode)

    def call(mode: Any) -> tuple[torch.Tensor, ...]:
        with runtime.forward_context(None, vllm_config, cudagraph_runtime_mode=mode, batch_descriptor=descriptor):
            return wrapper()

    for _ in range(3):
        stage(sources)
        call(runtime.eager_mode)
    stage(sources)
    expected_outputs = tuple(tensor.clone() for tensor in call(runtime.eager_mode))
    expected_state = tuple(tensor.clone() for tensor in _state_tensors(static_state))
    runtime.synchronize(env.device)
    runtime.set_capture_enabled(True)
    try:
        with runtime.capture_context(env.device):
            stage(sources)
            call(runtime.graph_mode)
        stage(sources)
        actual = call(runtime.graph_mode)
        for actual_tensor, expected in zip(
            (*actual, *_state_tensors(static_state)), (*expected_outputs, *expected_state), strict=True
        ):
            torch.testing.assert_close(actual_tensor, expected, atol=0, rtol=0)
        runtime.synchronize(env.device)
    finally:
        runtime.set_capture_enabled(False)

    @torch.inference_mode()
    def transition(
        actual_core: Any,
        actual_env: torch.Tensor,
        actual_state: SessionStateBatch,
        *,
        geometry: int,
        admitted_prompt: torch.Tensor,
        incoming_status: torch.Tensor,
        queue_capacity: int,
        decode_fn: Any,
        encoder_transition: Any = None,
        capture: bool = False,
        capture_geometry: Any | None = None,
    ) -> ChunkBucketResult:
        values = (actual_env, admitted_prompt, incoming_status, *_state_tensors(actual_state))
        if (
            actual_core is not core
            or geometry != 1
            or queue_capacity != frozen_capacity
            or decode_fn is not decode_dense_masked_frames
            or encoder_transition is not None
            or capture is not False
            or capture_geometry is not None
            or tuple(_tensor_signature(tensor) for tensor in values) != signature
        ):
            raise ValueError("bucket graph invocation differs from its captured cell")
        scratch_storages = {tensor.untyped_storage().data_ptr() for tensor in scratch}
        if any(tensor.untyped_storage().data_ptr() in scratch_storages for tensor in values):
            raise ValueError("bucket caller aliases graph-owned scratch")
        stage(values)
        output = call(runtime.graph_mode)
        for destination, source in zip(_state_tensors(actual_state), _state_tensors(static_state), strict=True):
            destination.copy_(source)
        escaped = tuple(tensor.clone() for tensor in output)
        result = result_type(
            token_ids=escaped[0],
            token_lengths=escaped[1],
            row_status=escaped[2],
            frame_emission_counts=escaped[3],
            frame_final_labels=escaped[4],
            frame_valid_lengths=escaped[5],
        )
        # Samples retain the caller's input lifetime, as in the eager bucket;
        # every graph-produced result/control tensor is independently owned.
        batch = ChunkBatch(
            samples=actual_env[:, ENVELOPE_HEADER_SLOTS : ENVELOPE_HEADER_SLOTS + sample_width],
            valid_samples=escaped[6],
            geometry_id=escaped[7],
            final_tail=escaped[8],
            prompt_index=escaped[9],
            chunk_sequence=escaped[10],
        )
        return ChunkBucketResult(batch, result, escaped[11], escaped[12])

    frozen_capacity = queue_capacity
    return transition
