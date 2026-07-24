# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for queue-less streaming terminal disposition at submission time."""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.output_processor import StreamingUpdate

from vllm_omni.engine.orchestrator import OrchestratorRequestState
from vllm_omni.engine.output_processor import (
    MultimodalOutputProcessor,
    OmniRequestState,
)
from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_detokenizer() -> MagicMock:
    return MagicMock(
        output_token_ids=[],
        get_next_output_text=MagicMock(return_value=""),
        num_output_tokens=MagicMock(return_value=0),
        update=MagicMock(return_value=None),
    )


def _make_request_state(request_id: str) -> OmniRequestState:
    return OmniRequestState(
        request_id=request_id,
        external_req_id=request_id,
        parent_req=None,
        request_index=0,
        lora_request=None,
        output_kind=RequestOutputKind.CUMULATIVE,
        prompt=None,
        prompt_token_ids=[1],
        prompt_embeds=None,
        logprobs_processor=MagicMock(
            logprobs=None,
            cumulative_logprob=None,
            prompt_logprobs=None,
        ),
        detokenizer=_make_detokenizer(),
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
        stream_input=True,
    )


def _register(
    processor: MultimodalOutputProcessor,
    request_state: OmniRequestState,
) -> None:
    processor.request_states[request_state.request_id] = request_state
    processor.external_req_ids[request_state.external_req_id].append(request_state.request_id)


def _terminal_request(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        external_req_id=request_id,
        resumable=False,
        prompt_token_ids=[0],
        prompt_embeds=None,
        sampling_params=SamplingParams(
            max_tokens=1,
            detokenize=False,
        ),
        pooling_params=None,
        lora_request=None,
        arrival_time=1.0,
    )


# @spec PORT-INT-008
def test_terminal_disposition_parked_complete_is_synchronous() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("parked")
    request_state.input_chunk_queue = None
    _register(processor, request_state)

    disposition = processor.apply_terminal_update(
        _terminal_request("parked"),
        prompt=None,
    )

    assert disposition.name == "PARKED_COMPLETE"
    assert "parked" not in processor.request_states
    assert "parked" not in processor.external_req_ids


# @spec PORT-INT-008
def test_terminal_disposition_active_request_stays_core_required() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("active")
    assert request_state.input_chunk_queue == deque()
    _register(processor, request_state)

    disposition = processor.apply_terminal_update(
        _terminal_request("active"),
        prompt=None,
    )

    assert disposition.name == "CORE_REQUIRED"
    assert request_state.streaming_input is False
    assert "active" in processor.request_states


# @spec PORT-INT-008
def test_terminal_disposition_queued_request_marks_last_update_final() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("queued")
    assert request_state.input_chunk_queue is not None
    request_state.input_chunk_queue.append(
        StreamingUpdate(
            prompt=None,
            prompt_token_ids=[9],
            arrival_time=0.5,
        )
    )
    _register(processor, request_state)

    disposition = processor.apply_terminal_update(
        _terminal_request("queued"),
        prompt=None,
    )

    assert disposition.name == "CORE_REQUIRED"
    assert request_state.input_chunk_queue[-1].final is True
    assert request_state.streaming_input is True


# @spec PORT-INT-008, PORT-INT-012
def test_terminal_disposition_unknown_does_not_create_request_state() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)

    disposition = processor.apply_terminal_update(
        _terminal_request("unknown"),
        prompt=None,
    )

    assert disposition.name == "UNKNOWN"
    assert processor.request_states == {}
    assert processor.external_req_ids == {}


class _StageClient:
    stage_type = "llm"
    final_output = True
    final_output_type = "text"

    def __init__(
        self,
        *,
        add_error: BaseException | None = None,
        abort_error: BaseException | None = None,
    ) -> None:
        self.add_calls: list[object] = []
        self.abort_calls: list[list[str]] = []
        self.add_error = add_error
        self.abort_error = abort_error

    async def add_request_async(self, request, **_kwargs) -> None:
        self.add_calls.append(request)
        if self.add_error is not None:
            raise self.add_error

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))
        if self.abort_error is not None:
            raise self.abort_error


class _BlockingStageClient(_StageClient):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def add_request_async(self, request, **kwargs) -> None:
        await super().add_request_async(request, **kwargs)
        self.entered.set()
        await self.release.wait()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        await super().abort_requests_async(request_ids)
        self.entered.set()
        await self.release.wait()


def _stage_pool(
    processor: MultimodalOutputProcessor,
    *,
    client: _StageClient | None = None,
) -> tuple[StagePool, _StageClient]:
    client = client or _StageClient()
    pool = StagePool(
        0,
        [client],
        output_processor=processor,
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    return pool, client


def _orchestrator_request_state(request_id: str) -> OrchestratorRequestState:
    state = OrchestratorRequestState(
        request_id=request_id,
        sampling_params_list=[SamplingParams(max_tokens=4)],
        final_stage_id=0,
        final_output_stage_ids={0},
    )
    state.streaming.enabled = True
    return state


# @spec PORT-INT-010
@pytest.mark.asyncio
async def test_stage_pool_parked_terminal_aborts_without_core_add() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("parked")
    request_state.input_chunk_queue = None
    _register(processor, request_state)
    pool, client = _stage_pool(processor)
    pool._request_bindings["parked"] = 0

    result = await pool.submit_update(
        "parked",
        _orchestrator_request_state("parked"),
        _terminal_request("parked"),
    )

    assert result.terminal_disposition.name == "PARKED_COMPLETE"
    assert result.replica_id == 0
    assert client.add_calls == []
    assert client.abort_calls == [["parked"]]


# @spec PORT-INT-009
@pytest.mark.asyncio
async def test_stage_pool_active_terminal_remains_core_ordered() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("active")
    _register(processor, request_state)
    pool, client = _stage_pool(processor)
    pool._request_bindings["active"] = 0

    terminal = _terminal_request("active")
    result = await pool.submit_update(
        "active",
        _orchestrator_request_state("active"),
        terminal,
    )

    assert result.terminal_disposition.name == "CORE_REQUIRED"
    assert result.replica_id == 0
    assert client.add_calls == [terminal]
    assert client.abort_calls == []


# @spec PORT-INT-009
@pytest.mark.asyncio
async def test_stage_pool_core_submission_failure_aborts_without_retry() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("active")
    _register(processor, request_state)
    client = _StageClient(
        add_error=RuntimeError("terminal core submission failed"),
    )
    pool, client = _stage_pool(processor, client=client)
    pool._request_bindings["active"] = 0
    terminal = _terminal_request("active")

    with pytest.raises(RuntimeError, match="terminal core submission failed"):
        await pool.submit_update(
            "active",
            _orchestrator_request_state("active"),
            terminal,
        )

    assert client.add_calls == [terminal]
    assert client.abort_calls == [["active"]]
    assert "active" not in processor.request_states


# @spec PORT-INT-010, PORT-INT-012
@pytest.mark.asyncio
async def test_stage_pool_parked_cleanup_failure_is_causal() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("parked")
    request_state.input_chunk_queue = None
    _register(processor, request_state)
    client = _StageClient(
        abort_error=RuntimeError("terminal cleanup abort failed"),
    )
    pool, client = _stage_pool(processor, client=client)
    pool._request_bindings["parked"] = 0

    with pytest.raises(RuntimeError, match="terminal cleanup abort failed"):
        await pool.submit_update(
            "parked",
            _orchestrator_request_state("parked"),
            _terminal_request("parked"),
        )

    assert client.add_calls == []
    assert client.abort_calls == [["parked"]]


# @spec PORT-INT-010, PORT-INT-011
@pytest.mark.asyncio
async def test_concurrent_parked_terminals_have_one_cleanup_owner() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("parked")
    request_state.input_chunk_queue = None
    _register(processor, request_state)
    client = _BlockingStageClient()
    pool, client = _stage_pool(processor, client=client)
    pool._request_bindings["parked"] = 0
    orchestrator_state = _orchestrator_request_state("parked")

    first = asyncio.create_task(
        pool.submit_update(
            "parked",
            orchestrator_state,
            _terminal_request("parked"),
        )
    )
    await client.entered.wait()
    second = asyncio.create_task(
        pool.submit_update(
            "parked",
            orchestrator_state,
            _terminal_request("parked"),
        )
    )
    await asyncio.sleep(0)
    client.release.set()
    await asyncio.gather(first, second)

    assert client.add_calls == []
    assert client.abort_calls == [["parked"]]


# @spec PORT-INT-009, PORT-INT-011
@pytest.mark.asyncio
async def test_concurrent_active_terminals_submit_once() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("active")
    _register(processor, request_state)
    client = _BlockingStageClient()
    pool, client = _stage_pool(processor, client=client)
    pool._request_bindings["active"] = 0
    orchestrator_state = _orchestrator_request_state("active")

    first = asyncio.create_task(
        pool.submit_update(
            "active",
            orchestrator_state,
            _terminal_request("active"),
        )
    )
    await client.entered.wait()
    second = asyncio.create_task(
        pool.submit_update(
            "active",
            orchestrator_state,
            _terminal_request("active"),
        )
    )
    await asyncio.sleep(0)
    client.release.set()
    await asyncio.gather(first, second)

    assert len(client.add_calls) == 1
    assert client.abort_calls == []


# @spec PORT-INT-012
@pytest.mark.asyncio
async def test_stage_pool_terminal_never_selects_replacement_replica() -> None:
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    request_state = _make_request_state("lost-binding")
    request_state.input_chunk_queue = None
    _register(processor, request_state)
    pool, client = _stage_pool(processor)

    with pytest.raises(RuntimeError, match="lifecycle"):
        await pool.submit_update(
            "lost-binding",
            _orchestrator_request_state("lost-binding"),
            _terminal_request("lost-binding"),
        )

    assert pool.get_bound_replica_id("lost-binding") is None
    assert client.add_calls == []
    assert client.abort_calls == []

    await pool.abort_requests(["lost-binding"])
    assert "lost-binding" not in processor.request_states
