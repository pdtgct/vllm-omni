# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestrator contracts for queue-less streaming terminal completion."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.messages import (
    ErrorMessage,
    OutputMessage,
    StageSubmissionMessage,
)
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    _build_terminal_empty_output,
)
from vllm_omni.engine.output_processor import StreamingTerminalDisposition
from vllm_omni.engine.stage_pool import StageUpdateResult

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RunningCounter:
    def __init__(self, value: int) -> None:
        self.value = value

    def increment(self) -> None:
        self.value += 1

    def decrement(self) -> None:
        self.value -= 1


class _CfgTracker:
    def is_companion(self, _request_id: str) -> bool:
        return False

    def has_companions(self, _request_id: str) -> bool:
        return False

    def cleanup_parent(self, _request_id: str) -> list[str]:
        return []


class _TerminalPool:
    final_output = True
    stage_client = SimpleNamespace(
        final_output_type="text",
        sample_rate=16000,
    )

    def __init__(
        self,
        *,
        disposition: str = "PARKED_COMPLETE",
        error: BaseException | None = None,
        abort_error: BaseException | None = None,
    ) -> None:
        self.disposition = disposition
        self.error = error
        self.abort_error = abort_error
        self.update_calls: list[str] = []
        self.initial_calls: list[str] = []
        self.abort_calls: list[list[str]] = []
        self.release_calls: list[list[str]] = []

    async def submit_update(
        self,
        request_id: str,
        _req_state: OrchestratorRequestState,
        _request: Any,
        *,
        prompt_text: Any = None,
    ) -> StageUpdateResult:
        del prompt_text
        self.update_calls.append(request_id)
        if self.error is not None:
            raise self.error
        disposition = StreamingTerminalDisposition[self.disposition]
        return StageUpdateResult(
            replica_id=0,
            terminal_disposition=disposition,
            owns_completion=(disposition is StreamingTerminalDisposition.PARKED_COMPLETE),
        )

    async def submit_initial(
        self,
        request_id: str,
        _req_state: OrchestratorRequestState,
        _request: Any,
        **_kwargs: Any,
    ) -> int:
        self.initial_calls.append(request_id)
        return 0

    async def abort_requests(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))
        if self.abort_error is not None:
            raise self.abort_error

    def release_bindings(self, request_ids: list[str]) -> None:
        self.release_calls.append(list(request_ids))


def _request_state(request_id: str) -> OrchestratorRequestState:
    state = OrchestratorRequestState(
        request_id=request_id,
        prompt={"prompt_token_ids": [1]},
        sampling_params_list=[SamplingParams(max_tokens=4)],
        final_stage_id=0,
        final_output_stage_ids={0},
        request_timestamp=time.time(),
    )
    state.streaming.enabled = True
    return state


def _orchestrator(
    pool: _TerminalPool,
    *request_ids: str,
) -> tuple[Orchestrator, _RunningCounter]:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = False
    orchestrator.stage_pools = [pool]
    orchestrator.request_states = {request_id: _request_state(request_id) for request_id in request_ids}
    orchestrator.output_async_queue = asyncio.Queue()
    orchestrator._pd_kv_params = {}
    orchestrator._pd_pair = None
    orchestrator._cfg_tracker = _CfgTracker()
    counter = _RunningCounter(len(request_ids))
    orchestrator._running_counter = counter
    return orchestrator, counter


def _terminal_message(request_id: str) -> StageSubmissionMessage:
    prompt = SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[0],
        resumable=False,
    )
    return StageSubmissionMessage(
        type="streaming_update",
        request_id=request_id,
        prompt=prompt,
        original_prompt=prompt,
        output_prompt_text=None,
        sampling_params_list=[SamplingParams(max_tokens=4)],
        final_stage_id=0,
        final_output_stage_ids=[0],
        preprocess_ms=0.0,
        request_timestamp=time.time(),
        enqueue_ts=time.perf_counter(),
    )


# @spec PORT-INT-010, PORT-INT-011
@pytest.mark.asyncio
async def test_parked_completion_routes_one_stop_and_cleans_only_owner() -> None:
    pool = _TerminalPool()
    orchestrator, counter = _orchestrator(pool, "done", "other")
    # The normal terminal marker follows a committed FLUSH park, so the
    # preceding segment boundary is still visible when terminal promotion
    # begins. It must not suppress the request-level completion below.
    orchestrator.request_states["done"].streaming.segment_finished = True

    await orchestrator._handle_streaming_update(
        _terminal_message("done"),
    )

    message = orchestrator.output_async_queue.get_nowait()
    assert isinstance(message, OutputMessage)
    assert message.request_id == "done"
    assert message.finished is True
    assert message.engine_outputs.finished is True
    assert message.engine_outputs.outputs[0].text == ""
    assert message.engine_outputs.outputs[0].token_ids == []
    assert message.engine_outputs.outputs[0].finish_reason == "stop"
    assert "done" not in orchestrator.request_states
    assert "other" in orchestrator.request_states
    assert counter.value == 1
    assert pool.release_calls == [["done"]]


# @spec PORT-INT-009, PORT-INT-010
@pytest.mark.asyncio
async def test_resumable_segment_boundary_does_not_finish_logical_request() -> None:
    pool = _TerminalPool()
    orchestrator, counter = _orchestrator(pool, "streaming")
    req_state = orchestrator.request_states["streaming"]
    req_state.streaming.segment_finished = True
    segment_output = _build_terminal_empty_output(
        "streaming",
        final_output_type="text",
    )

    await orchestrator._route_output(
        0,
        0,
        segment_output,
        req_state,
        None,
    )

    message = orchestrator.output_async_queue.get_nowait()
    assert isinstance(message, OutputMessage)
    assert message.engine_outputs.finished is True
    assert message.finished is False
    assert "streaming" in orchestrator.request_states
    assert counter.value == 1
    assert pool.release_calls == []


# @spec PORT-INT-011, PORT-INT-012
@pytest.mark.asyncio
async def test_unknown_terminal_is_idempotent_not_fresh_request() -> None:
    pool = _TerminalPool()
    orchestrator, counter = _orchestrator(pool)

    await orchestrator._handle_streaming_update(
        _terminal_message("already-clean"),
    )

    assert pool.initial_calls == []
    assert pool.update_calls == []
    assert orchestrator.output_async_queue.empty()
    assert orchestrator.request_states == {}
    assert counter.value == 0


# @spec PORT-INT-011
@pytest.mark.asyncio
async def test_delayed_core_finish_after_cleanup_is_discarded() -> None:
    pool = _TerminalPool()
    orchestrator, counter = _orchestrator(pool)
    delayed_finish = SimpleNamespace(
        request_id="already-clean",
        finished=True,
        outputs=[],
    )

    await orchestrator._handle_processed_outputs(
        0,
        0,
        [delayed_finish],
    )

    assert orchestrator.output_async_queue.empty()
    assert pool.release_calls == []
    assert counter.value == 0


# @spec PORT-INT-011, PORT-INT-012
@pytest.mark.asyncio
async def test_terminal_lifecycle_error_is_request_scoped() -> None:
    pool = _TerminalPool(
        error=RuntimeError("terminal lifecycle divergence"),
    )
    orchestrator, counter = _orchestrator(pool, "broken", "healthy")

    await orchestrator._handle_streaming_update(
        _terminal_message("broken"),
    )

    message = orchestrator.output_async_queue.get_nowait()
    assert isinstance(message, ErrorMessage)
    assert message.request_id == "broken"
    assert "lifecycle divergence" in message.error
    assert "broken" not in orchestrator.request_states
    assert "healthy" in orchestrator.request_states
    assert counter.value == 1
    assert pool.abort_calls == [["broken"]]
    assert pool.release_calls == [["broken"]]


# @spec PORT-INT-012
@pytest.mark.asyncio
async def test_repeated_cleanup_failure_still_releases_logical_request() -> None:
    pool = _TerminalPool(
        error=RuntimeError("terminal cleanup abort failed"),
        abort_error=RuntimeError("terminal cleanup still failing"),
    )
    orchestrator, counter = _orchestrator(pool, "broken", "healthy")

    await orchestrator._handle_streaming_update(
        _terminal_message("broken"),
    )

    message = orchestrator.output_async_queue.get_nowait()
    assert isinstance(message, ErrorMessage)
    assert message.request_id == "broken"
    assert message.error == "terminal cleanup abort failed"
    assert "broken" not in orchestrator.request_states
    assert "healthy" in orchestrator.request_states
    assert counter.value == 1
    assert pool.abort_calls == [["broken"]]
    assert pool.release_calls == [["broken"]]


# @spec PORT-INT-012
@pytest.mark.asyncio
async def test_terminal_cleanup_attempts_every_stage_after_one_abort_fails() -> None:
    first = _TerminalPool(
        abort_error=RuntimeError("first stage abort failed"),
    )
    first.stage_id = 0
    second = _TerminalPool()
    second.stage_id = 1
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [first, second]

    with pytest.raises(RuntimeError, match="first stage abort failed"):
        await orchestrator._abort_request_ids(["broken"])

    assert first.abort_calls == [["broken"]]
    assert second.abort_calls == [["broken"]]


# @spec PORT-INT-011, PORT-INT-012
@pytest.mark.asyncio
async def test_live_request_with_unknown_stage_state_fails_closed() -> None:
    pool = _TerminalPool(disposition="UNKNOWN")
    orchestrator, counter = _orchestrator(pool, "diverged", "healthy")

    await orchestrator._handle_streaming_update(
        _terminal_message("diverged"),
    )

    message = orchestrator.output_async_queue.get_nowait()
    assert isinstance(message, ErrorMessage)
    assert message.request_id == "diverged"
    assert "lifecycle" in message.error
    assert "diverged" not in orchestrator.request_states
    assert "healthy" in orchestrator.request_states
    assert counter.value == 1
    assert pool.abort_calls == [["diverged"]]
    assert pool.release_calls == [["diverged"]]


# @spec PORT-INT-010, PORT-INT-012
@pytest.mark.asyncio
async def test_intermediate_stage_parked_completion_fails_closed() -> None:
    pool = _TerminalPool()
    pool.final_output = False
    orchestrator, counter = _orchestrator(pool, "intermediate")

    await orchestrator._handle_streaming_update(
        _terminal_message("intermediate"),
    )

    message = orchestrator.output_async_queue.get_nowait()
    assert isinstance(message, ErrorMessage)
    assert message.request_id == "intermediate"
    assert "unsupported" in message.error
    assert "intermediate" not in orchestrator.request_states
    assert counter.value == 0
    assert pool.abort_calls == [["intermediate"]]
    assert pool.release_calls == [["intermediate"]]


# @spec PORT-INT-009
@pytest.mark.asyncio
async def test_core_required_terminal_emits_no_early_output() -> None:
    pool = _TerminalPool(disposition="CORE_REQUIRED")
    orchestrator, counter = _orchestrator(pool, "active")

    await orchestrator._handle_streaming_update(
        _terminal_message("active"),
    )

    assert orchestrator.output_async_queue.empty()
    assert "active" in orchestrator.request_states
    assert counter.value == 1
    assert pool.release_calls == []
