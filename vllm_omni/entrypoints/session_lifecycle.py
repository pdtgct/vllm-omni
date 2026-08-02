# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared resettable lifecycle deadline for streaming sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, Literal

SessionLifecycleKind = Literal["idle", "finalization"]
ExpiryCallback = Callable[
    [SessionLifecycleKind],
    Coroutine[Any, Any, None],
]


class SessionLifecycleDeadline:
    """Own one resettable timer handle and one terminal expiry callback.

    Cadence-rate rearming only replaces an event-loop ``TimerHandle``. An
    ``asyncio.Task`` is created once, and only if the selected deadline
    actually expires. Once that callback starts, ``close`` waits for it
    rather than cancelling terminal cleanup.
    """

    def __init__(
        self,
        *,
        idle_timeout_s: float | None,
        finalization_timeout_s: float | None,
        on_expire: ExpiryCallback,
    ) -> None:
        self._idle_timeout_s = idle_timeout_s
        self._finalization_timeout_s = finalization_timeout_s
        self._on_expire = on_expire
        self._handle: asyncio.TimerHandle | None = None
        self._expiry_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._expired = False
        self._expired_kind: SessionLifecycleKind | None = None
        self._closed = False

    @property
    def generation(self) -> int:
        """Current rearm generation used to fence stale callbacks."""

        return self._generation

    @property
    def expired(self) -> bool:
        """Whether a live deadline has fired."""

        return self._expired

    @property
    def expired_kind(self) -> SessionLifecycleKind | None:
        """The terminal deadline kind, when expired."""

        return self._expired_kind

    def arm(self, kind: SessionLifecycleKind) -> None:
        """Replace the live deadline without creating an asyncio task."""

        if self._expired:
            raise RuntimeError("session lifecycle has expired")
        if self._closed:
            raise RuntimeError("session lifecycle deadline is closed")
        self._generation += 1
        generation = self._generation
        self._cancel_handle()
        timeout_s = (
            self._idle_timeout_s
            if kind == "idle"
            else self._finalization_timeout_s
        )
        if timeout_s is None:
            return
        loop = asyncio.get_running_loop()
        self._handle = loop.call_later(
            float(timeout_s),
            self._deadline_reached,
            generation,
            kind,
        )

    def _deadline_reached(
        self,
        generation: int,
        kind: SessionLifecycleKind,
    ) -> None:
        """Latch one current expiry and schedule its terminal callback."""

        if (
            self._closed
            or self._expired
            or generation != self._generation
        ):
            return
        self._cancel_handle()
        self._expired = True
        self._expired_kind = kind
        self._expiry_task = asyncio.create_task(
            self._on_expire(kind),
            name=f"streaming-session-{kind}-expiry",
        )

    async def close(self) -> None:
        """Cancel a pending handle or await terminal cleanup in progress."""

        if not self._closed:
            self._closed = True
            self._generation += 1
            self._cancel_handle()
        task = self._expiry_task
        if task is None or task is asyncio.current_task():
            return
        await asyncio.shield(task)

    def _cancel_handle(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.cancel()
