# SPDX-License-Identifier: Apache-2.0
"""A scheduler-side error finish must surface as an error event, never as a
successful transcription.done (recover-don't-latch: the connection stays usable)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.entrypoints.session_lifecycle import SessionLifecycleDeadline


class _Engine:
    default_sampling_params_list = [SimpleNamespace(tag="default")]

    def __init__(self, runs: list[list[Any]]) -> None:
        self.runs = runs

    def generate(self, **_kwargs: Any) -> AsyncGenerator[Any, None]:
        outputs = self.runs.pop(0)

        async def _gen() -> AsyncGenerator[Any, None]:
            for output in outputs:
                yield output

        return _gen()


def _output(text: str, finish_reason: str | None = None) -> Any:
    return SimpleNamespace(
        stage_id=0,
        outputs=[SimpleNamespace(text=text, token_ids=[], finish_reason=finish_reason)],
        prompt_token_ids=[1],
        multimodal_output=None,
    )


def _connection(engine: Any) -> tuple[RealtimeConnection, list[Any], list[tuple[str, str | None]]]:
    connection: Any = RealtimeConnection.__new__(RealtimeConnection)
    connection.connection_id = "test"
    connection.engine = engine
    connection._is_connected = True
    connection.audio_queue = asyncio.Queue()
    connection._observer = None
    connection._park_token_id = None

    async def _inert(_kind: str) -> None:
        return None

    connection._session_lifecycle = SessionLifecycleDeadline(
        idle_timeout_s=None, finalization_timeout_s=None, on_expire=_inert
    )
    events: list[Any] = []
    errors: list[tuple[str, str | None]] = []

    async def _send(event: Any) -> None:
        events.append(event)

    async def _send_json(_payload: dict[str, Any]) -> None:
        pass

    async def _send_error(message: str, error_type: str | None = None) -> None:
        errors.append((message, error_type))

    connection.send = _send
    connection.send_json = _send_json
    connection.send_error = _send_error
    return connection, events, errors


async def _empty() -> AsyncGenerator[Any, None]:
    return
    yield None


def _done(events: list[Any]) -> list[Any]:
    return [e for e in events if getattr(e, "type", "") == "transcription.done"]


@pytest.mark.asyncio
async def test_error_finish_emits_error_and_no_done() -> None:
    engine = _Engine([[_output("", finish_reason="error")]])
    connection, events, errors = _connection(engine)
    await connection._run_generation(_empty(), asyncio.Queue(), request_id="r1")
    assert errors == [("generation failed", "processing_error")]
    assert _done(events) == []
    assert connection._streaming_finish_reason == "error"


@pytest.mark.asyncio
async def test_normal_finish_is_unchanged() -> None:
    engine = _Engine([[_output("hello", finish_reason=None), _output("", finish_reason="stop")]])
    connection, events, errors = _connection(engine)
    await connection._run_generation(_empty(), asyncio.Queue(), request_id="r1")
    assert errors == []
    done = _done(events)
    assert len(done) == 1 and done[0].text == "hello"
    assert connection._streaming_finish_reason == "completed"


@pytest.mark.asyncio
async def test_connection_can_generate_again_after_error_finish() -> None:
    engine = _Engine([[_output("", finish_reason="error")], [_output("again")]])
    connection, events, errors = _connection(engine)
    await connection._run_generation(_empty(), asyncio.Queue(), request_id="r1")
    await connection._run_generation(_empty(), asyncio.Queue(), request_id="r2")
    assert len(errors) == 1
    done = _done(events)
    assert len(done) == 1 and done[0].text == "again"
