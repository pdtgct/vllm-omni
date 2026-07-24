# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def realtime_conn() -> RealtimeConnection:
    return RealtimeConnection.__new__(RealtimeConnection)


class TestRealtimeConnectionTensorAndPcm:
    def test_tensor_to_numpy_none(self) -> None:
        assert RealtimeConnection._tensor_to_numpy(None) is None

    def test_tensor_to_numpy_1d_numpy(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.dtype == np.float32
        assert out.shape == (2,)

    def test_tensor_to_numpy_2d_numpy_flattened(self) -> None:
        arr = np.array([[0.5], [-0.5]], dtype=np.float32)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.shape == (2,)

    def test_tensor_to_numpy_torch(self) -> None:
        t = torch.tensor([[0.25, -0.25]], dtype=torch.float32)
        out = RealtimeConnection._tensor_to_numpy(t)
        assert out is not None
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.25, -0.25], rtol=1e-5)

    def test_pcm16_b64_roundtrip(self) -> None:
        audio = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        b64 = RealtimeConnection._pcm16_b64(audio)
        raw = base64.b64decode(b64)
        assert len(raw) == 6
        pcm = np.frombuffer(raw, dtype=np.int16)
        assert pcm[0] == 0
        assert pcm[1] == 32767
        assert pcm[2] == -32767


class TestAsyncOmniStreamingParamsValidation:
    def test_accepts_streaming_friendly_params(self) -> None:
        p = SamplingParams(
            n=1,
            stop=[],
            output_kind=RequestOutputKind.DELTA,
        )
        AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_non_sampling_params(self) -> None:
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(object())  # type: ignore[arg-type]

    def test_rejects_n_greater_than_one(self) -> None:
        p = SamplingParams(n=2, stop=[], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_final_only(self) -> None:
        p = SamplingParams(n=1, stop=[], output_kind=RequestOutputKind.FINAL_ONLY)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_stop_strings(self) -> None:
        p = SamplingParams(n=1, stop=["\n"], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)


class _RecordingStreamingEngine:
    def __init__(self) -> None:
        self.initial: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    async def add_request_async(self, **kwargs: Any) -> None:
        self.initial.append(kwargs)

    async def add_streaming_update_async(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


def _streaming_omni() -> tuple[AsyncOmni, _RecordingStreamingEngine, Any]:
    omni = object.__new__(AsyncOmni)
    engine = _RecordingStreamingEngine()
    request_state = SimpleNamespace(
        queue=asyncio.Queue(),
        input_stream_task=None,
    )
    omni.engine = engine
    omni.model_config = SimpleNamespace(is_encoder_decoder=False)
    omni.request_states = {"rt-test": request_state}
    return omni, engine, request_state


def _streaming_params() -> SamplingParams:
    return SamplingParams(
        n=1,
        stop=[],
        output_kind=RequestOutputKind.DELTA,
    )


# @spec PORT-REGIME-005
@pytest.mark.asyncio
async def test_input_error_before_first_carrier_submits_no_terminal_row() -> None:
    async def broken_input() -> AsyncGenerator[StreamingInput, None]:
        raise RuntimeError("carrier render failed")
        yield StreamingInput(prompt={"prompt_token_ids": [13089]})  # pragma: no cover

    omni, engine, request_state = _streaming_omni()
    task = await omni._add_streaming_input_request(
        request_id="rt-test",
        input_stream=broken_input(),
        sampling_params_list=[_streaming_params()],
        final_stage_id=0,
        final_output_stage_ids=[0],
        arrival_time=0.0,
    )
    await task

    error = request_state.queue.get_nowait()
    assert error.error == "carrier render failed"
    assert engine.initial == []
    assert engine.updates == []


# @spec PORT-REGIME-005, PORT-RTC-006
@pytest.mark.asyncio
async def test_input_error_after_admission_submits_no_synthetic_finalization() -> None:
    first_prompt = {"prompt_token_ids": [13089]}

    async def broken_input() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(prompt=first_prompt)
        raise RuntimeError("second carrier render failed")

    omni, engine, request_state = _streaming_omni()
    task = await omni._add_streaming_input_request(
        request_id="rt-test",
        input_stream=broken_input(),
        sampling_params_list=[_streaming_params()],
        final_stage_id=0,
        final_output_stage_ids=[0],
        arrival_time=0.0,
    )
    await task

    error = request_state.queue.get_nowait()
    assert error.error == "second carrier render failed"
    assert len(engine.initial) == 1
    assert engine.initial[0]["prompt"] is first_prompt
    assert engine.initial[0]["resumable"] is True
    assert engine.updates == []


# @spec PORT-DEC-009, PORT-REGIME-004
@pytest.mark.asyncio
async def test_explicit_flush_precedes_successful_end_of_input_marker() -> None:
    carrier = {
        "prompt_token_ids": [13089],
        "multi_modal_data": {"audio": np.zeros(8, dtype=np.float32)},
    }
    flush = {"prompt_token_ids": [0]}

    async def complete_input() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(prompt=carrier)
        yield StreamingInput(prompt=flush)

    omni, engine, request_state = _streaming_omni()
    task = await omni._add_streaming_input_request(
        request_id="rt-test",
        input_stream=complete_input(),
        sampling_params_list=[_streaming_params()],
        final_stage_id=0,
        final_output_stage_ids=[0],
        arrival_time=0.0,
    )
    await task

    assert request_state.queue.empty()
    assert len(engine.initial) == 1
    assert engine.initial[0]["prompt"] is carrier
    assert engine.initial[0]["resumable"] is True
    assert engine.updates[0]["prompt"] == flush
    assert engine.updates[0]["resumable"] is True
    assert engine.updates[1]["prompt"]["prompt_token_ids"] == [0]
    assert engine.updates[1]["resumable"] is False
