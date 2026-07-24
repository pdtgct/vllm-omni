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
from vllm.entrypoints.speech_to_text.realtime.protocol import (
    TranscriptionDelta,
    TranscriptionDone,
)
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


class _StreamingTestAsyncOmni(AsyncOmni):
    """Test double: ``AsyncOmni.model_config`` is a read-only ``@property``
    (async_omni.py) that derives its value from ``self.engine``/
    ``self.vllm_config`` — irrelevant plumbing for these input-stream unit
    tests. Override it here with a plain settable property so the fixture
    can stamp a stand-in config directly, instead of reconstructing the
    engine/stage-config chain the real property reads."""

    @property
    def model_config(self):  # type: ignore[override]
        return self._test_model_config

    @model_config.setter
    def model_config(self, value):  # type: ignore[override]
        self._test_model_config = value


def _streaming_omni() -> tuple[AsyncOmni, _RecordingStreamingEngine, Any]:
    omni = object.__new__(_StreamingTestAsyncOmni)
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


class _RealtimeGenerationEngine:
    default_sampling_params_list = [_streaming_params()]

    def __init__(
        self,
        outputs: list[Any],
        *,
        error: BaseException | None = None,
    ) -> None:
        self.outputs = outputs
        self.error = error

    def generate(self, **_kwargs: Any) -> AsyncGenerator[Any, None]:
        async def _outputs() -> AsyncGenerator[Any, None]:
            for output in self.outputs:
                yield output
            if self.error is not None:
                raise self.error

        return _outputs()


def _generation_output(text: str, token_ids: list[int]) -> Any:
    return SimpleNamespace(
        stage_id=0,
        outputs=[
            SimpleNamespace(
                text=text,
                token_ids=token_ids,
            )
        ],
        prompt_token_ids=[1],
        multimodal_output=None,
    )


def _realtime_generation_connection(
    engine: _RealtimeGenerationEngine,
) -> tuple[
    RealtimeConnection,
    list[Any],
    list[dict[str, Any]],
    list[tuple[str, str]],
]:
    connection = RealtimeConnection.__new__(RealtimeConnection)
    connection.connection_id = "test"
    connection.engine = engine
    connection._is_connected = True
    connection.audio_queue = asyncio.Queue()

    sent_events: list[Any] = []
    sent_json: list[dict[str, Any]] = []
    sent_errors: list[tuple[str, str]] = []

    async def _send(event: Any) -> None:
        sent_events.append(event)

    async def _send_json(payload: dict[str, Any]) -> None:
        sent_json.append(payload)

    async def _send_error(message: str, error_type: str) -> None:
        sent_errors.append((message, error_type))

    connection.send = _send  # type: ignore[method-assign]
    connection.send_json = _send_json  # type: ignore[method-assign]
    connection.send_error = _send_error  # type: ignore[method-assign]
    return connection, sent_events, sent_json, sent_errors


async def _empty_streaming_input() -> AsyncGenerator[Any, None]:
    if False:
        yield None


# @spec ING-LIFE-011
@pytest.mark.asyncio
async def test_successful_empty_terminal_output_emits_one_done_after_deltas() -> None:
    engine = _RealtimeGenerationEngine(
        [
            _generation_output("hello", [7]),
            _generation_output("", []),
        ]
    )
    connection, sent_events, _sent_json, sent_errors = _realtime_generation_connection(engine)

    await connection._run_generation(
        _empty_streaming_input(),
        asyncio.Queue(),
    )

    assert [type(event) for event in sent_events] == [
        TranscriptionDelta,
        TranscriptionDone,
    ]
    assert sent_events[0].delta == "hello"
    assert sent_events[1].text == "hello"
    assert sent_errors == []


# @spec ING-LIFE-011
@pytest.mark.asyncio
async def test_generation_error_emits_processing_error_without_done() -> None:
    engine = _RealtimeGenerationEngine(
        [_generation_output("partial", [7])],
        error=RuntimeError("terminal lifecycle divergence"),
    )
    connection, sent_events, _sent_json, sent_errors = _realtime_generation_connection(engine)

    await connection._run_generation(
        _empty_streaming_input(),
        asyncio.Queue(),
    )

    assert [type(event) for event in sent_events] == [TranscriptionDelta]
    assert sent_errors == [
        ("terminal lifecycle divergence", "processing_error"),
    ]


# @spec ING-LIFE-011
@pytest.mark.asyncio
async def test_disconnected_generation_emits_no_terminal_event() -> None:
    engine = _RealtimeGenerationEngine([])
    connection, sent_events, sent_json, sent_errors = _realtime_generation_connection(engine)
    connection._is_connected = False

    await connection._run_generation(
        _empty_streaming_input(),
        asyncio.Queue(),
    )

    assert sent_events == []
    assert sent_json == []
    assert sent_errors == []
