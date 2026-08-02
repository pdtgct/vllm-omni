# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for the shared resettable streaming lifecycle deadline."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, NoReturn

import pytest


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError


def _deadline_class() -> Any:
    try:
        from vllm_omni.entrypoints.session_lifecycle import (
            SessionLifecycleDeadline,
        )
    except (ImportError, ModuleNotFoundError):
        _fail("PORT-SESS-005 missing shared SessionLifecycleDeadline")
    return SessionLifecycleDeadline


def _callback(
    observed: list[str],
) -> Callable[[str], Awaitable[None]]:
    async def observe(kind: str) -> None:
        observed.append(kind)

    return observe


# @spec PORT-SESS-005
@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_cadence_rearm_creates_no_asyncio_task() -> None:
    deadline_type = _deadline_class()
    observed: list[str] = []
    deadline = deadline_type(
        idle_timeout_s=60.0,
        finalization_timeout_s=40.0,
        on_expire=_callback(observed),
    )
    tasks_before = asyncio.all_tasks()

    deadline.arm("idle")
    for _ in range(100):
        deadline.arm("idle")

    assert asyncio.all_tasks() == tasks_before
    assert observed == []
    await deadline.close()


# @spec PORT-SESS-005
@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_stale_timer_callback_cannot_expire_a_rearmed_deadline() -> None:
    deadline_type = _deadline_class()
    observed: list[str] = []
    deadline = deadline_type(
        idle_timeout_s=60.0,
        finalization_timeout_s=40.0,
        on_expire=_callback(observed),
    )
    deadline.arm("idle")
    stale_generation = deadline.generation

    deadline.arm("idle")
    deadline._deadline_reached(stale_generation, "idle")
    await asyncio.sleep(0)

    assert deadline.expired is False
    assert observed == []
    await deadline.close()


# @spec PORT-SESS-005
@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_expiry_creates_one_callback_task_and_latches_terminal_kind() -> None:
    deadline_type = _deadline_class()
    observed: list[str] = []
    deadline = deadline_type(
        idle_timeout_s=0.001,
        finalization_timeout_s=0.001,
        on_expire=_callback(observed),
    )

    deadline.arm("finalization")
    await asyncio.sleep(0.02)

    assert deadline.expired is True
    assert deadline.expired_kind == "finalization"
    assert observed == ["finalization"]
    with pytest.raises(RuntimeError, match="expired"):
        deadline.arm("idle")
    await deadline.close()


# @spec PORT-SESS-005
@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_close_cancels_the_live_handle_without_running_expiry() -> None:
    deadline_type = _deadline_class()
    observed: list[str] = []
    deadline = deadline_type(
        idle_timeout_s=60.0,
        finalization_timeout_s=40.0,
        on_expire=_callback(observed),
    )
    deadline.arm("idle")

    await deadline.close()
    deadline._deadline_reached(deadline.generation, "idle")
    await asyncio.sleep(0)

    assert observed == []


# @spec PORT-SESS-005, PORT-STATE-014
@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_close_awaits_an_expiry_callback_already_in_progress() -> None:
    deadline_type = _deadline_class()
    started = asyncio.Event()
    proceed = asyncio.Event()
    completed: list[str] = []

    async def expire(kind: str) -> None:
        started.set()
        await proceed.wait()
        completed.append(kind)

    deadline = deadline_type(
        idle_timeout_s=60.0,
        finalization_timeout_s=40.0,
        on_expire=expire,
    )
    deadline.arm("finalization")
    deadline._deadline_reached(deadline.generation, "finalization")
    await started.wait()

    close_task = asyncio.create_task(deadline.close())
    await asyncio.sleep(0)
    assert close_task.done() is False

    proceed.set()
    await close_task
    assert completed == ["finalization"]
