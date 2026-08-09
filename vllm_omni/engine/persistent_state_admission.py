# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded pre-audio admission for persistent streaming sessions."""

from __future__ import annotations

import hashlib
import heapq
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from vllm_omni.engine.persistent_state_service import (
    PersistentStateBackpressure,
)

AdmissionAuthority = Literal["OPEN", "UNHANDSHAKED"]
AdmissionOutcome = Literal["admitted", "shed", "unavailable", "cancelled"]


class StaleAdmissionHandle(RuntimeError):  # noqa: N818
    """An admission handle no longer identifies its slab generation."""


class AdmissionQueueFull(PersistentStateBackpressure):
    """The fixed pre-audio waiter slab has no free entry."""

    def __init__(self, *, retry_after_ms: int) -> None:
        super().__init__("persistent-state admission controller is full")
        self.reason = "capacity"
        self.cause = "controller_full"
        self.retry_after_ms = retry_after_ms


@dataclass(frozen=True)
class AdmissionControllerConfig:
    waiter_capacity: int
    max_inflight_reserves: int
    dispatch_budget: int
    aging_threshold_ns: int
    admission_wait_timeout_s: float
    retry_floor_ms: int
    retry_jitter_ms: int
    recovery_backoff_s: tuple[float, ...]
    release_convergence_timeout_s: float
    supported_intervals_ms: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.waiter_capacity <= 0:
            raise ValueError("waiter capacity must be positive")
        if not (
            1
            <= self.dispatch_budget
            <= self.max_inflight_reserves
            <= self.waiter_capacity
        ):
            raise ValueError(
                "dispatch budget, reserve limit, and waiter capacity are ordered"
            )
        if self.aging_threshold_ns <= 0:
            raise ValueError("aging threshold must be positive")
        if self.admission_wait_timeout_s <= 0:
            raise ValueError("admission wait timeout must be positive")
        if self.retry_floor_ms < 0 or self.retry_jitter_ms < 0:
            raise ValueError("retry delay values must be non-negative")
        if (
            not self.recovery_backoff_s
            or any(value <= 0 for value in self.recovery_backoff_s)
            or self.release_convergence_timeout_s <= 0
        ):
            raise ValueError("recovery bounds must be positive")
        if (
            not self.supported_intervals_ms
            or len(set(self.supported_intervals_ms))
            != len(self.supported_intervals_ms)
            or any(value <= 0 for value in self.supported_intervals_ms)
        ):
            raise ValueError("supported intervals must be unique and positive")


@dataclass(frozen=True)
class AdmissionAttempt:
    attempt_id: str
    connection_id: str
    connection_handle: str
    operation_id: str
    service_interval_ms: int
    admission_deadline_ns: int
    unadmitted_deadline_ns: int


@dataclass(frozen=True)
class AdmissionCapacity:
    hard_headroom: int
    nominal_headroom_by_interval: Mapping[int, int]
    authority: str


@dataclass(frozen=True)
class AdmissionHandle:
    slot: int
    generation: int


@dataclass(frozen=True)
class AdmissionDispatch:
    handle: AdmissionHandle
    attempt: AdmissionAttempt


@dataclass(frozen=True)
class AdmissionDisposition:
    attempt_id: str
    outcome: AdmissionOutcome
    terminal_connection: bool = False
    close_code: int | None = None


@dataclass(frozen=True)
class AdmissionControllerSnapshot:
    waiter_count: int
    submitted_count: int
    free_entries: int
    protected_attempt_id: str | None
    authority: AdmissionAuthority
    last_drain_examined_heads: int
    last_drain_launched: int


@dataclass
class _Entry:
    attempt: AdmissionAttempt
    handle: AdmissionHandle
    enqueued_at_ns: int
    sequence: int
    state: Literal["waiting", "submitted"] = "waiting"


class BoundedAdmissionController:
    """Own a preallocated waiter slab and five fixed cadence FIFOs."""

    # @spec PORT-STATE-027, PORT-STATE-028, PORT-PERF-008
    def __init__(
        self,
        *,
        config: AdmissionControllerConfig,
        engine_epoch: str,
        monotonic_ns: Callable[[], int],
        jitter_secret: bytes,
    ) -> None:
        if not engine_epoch or not jitter_secret:
            raise ValueError("engine epoch and jitter secret cannot be empty")
        self._config = config
        self._engine_epoch = engine_epoch
        self._monotonic_ns = monotonic_ns
        self._jitter_secret = jitter_secret
        self._entries: list[_Entry | None] = [None] * config.waiter_capacity
        self._generations = [0] * config.waiter_capacity
        self._free = list(range(config.waiter_capacity))
        heapq.heapify(self._free)
        self._queues = {
            interval: deque[int]() for interval in config.supported_intervals_ms
        }
        self._connections: dict[str, AdmissionHandle] = {}
        self._deadlines: list[tuple[int, int, int, int]] = []
        self._sequence = 0
        self._submitted = 0
        self._protected: AdmissionHandle | None = None
        self._authority: AdmissionAuthority = "OPEN"
        self._last_examined = 0
        self._last_launched = 0

    @property
    def snapshot(self) -> AdmissionControllerSnapshot:
        protected = self._entry_or_none(self._protected)
        return AdmissionControllerSnapshot(
            waiter_count=self._config.waiter_capacity - len(self._free),
            submitted_count=self._submitted,
            free_entries=len(self._free),
            protected_attempt_id=(
                None if protected is None else protected.attempt.attempt_id
            ),
            authority=self._authority,
            last_drain_examined_heads=self._last_examined,
            last_drain_launched=self._last_launched,
        )

    def _entry_or_none(self, handle: AdmissionHandle | None) -> _Entry | None:
        if handle is None or not 0 <= handle.slot < len(self._entries):
            return None
        entry = self._entries[handle.slot]
        if entry is None or entry.handle != handle:
            return None
        return entry

    def _require(self, handle: AdmissionHandle) -> _Entry:
        entry = self._entry_or_none(handle)
        if entry is None:
            raise StaleAdmissionHandle("stale admission slab handle")
        return entry

    def _retry_after_ms(self, attempt_id: str) -> int:
        if self._config.retry_jitter_ms == 0:
            return self._config.retry_floor_ms
        digest = hashlib.blake2s(
            attempt_id.encode(),
            key=self._jitter_secret,
            digest_size=4,
        ).digest()
        jitter = int.from_bytes(digest, "big") % (
            self._config.retry_jitter_ms + 1
        )
        return self._config.retry_floor_ms + jitter

    def enqueue(self, attempt: AdmissionAttempt) -> AdmissionHandle:
        if attempt.service_interval_ms not in self._queues:
            raise ValueError(
                f"unsupported service interval {attempt.service_interval_ms}"
            )
        existing = self._connections.get(attempt.connection_id)
        if existing is not None and self._entry_or_none(existing) is not None:
            return existing
        if not self._free:
            raise AdmissionQueueFull(
                retry_after_ms=self._retry_after_ms(attempt.attempt_id)
            )
        slot = heapq.heappop(self._free)
        self._generations[slot] += 1
        handle = AdmissionHandle(slot, self._generations[slot])
        self._sequence += 1
        entry = _Entry(
            attempt=attempt,
            handle=handle,
            enqueued_at_ns=self._monotonic_ns(),
            sequence=self._sequence,
        )
        self._entries[slot] = entry
        self._connections[attempt.connection_id] = handle
        self._queues[attempt.service_interval_ms].append(slot)
        heapq.heappush(
            self._deadlines,
            (attempt.admission_deadline_ns, 0, slot, handle.generation),
        )
        heapq.heappush(
            self._deadlines,
            (attempt.unadmitted_deadline_ns, 1, slot, handle.generation),
        )
        return handle

    def _release_slot(self, entry: _Entry) -> None:
        if entry.state == "submitted":
            self._submitted -= 1
        self._entries[entry.handle.slot] = None
        self._connections.pop(entry.attempt.connection_id, None)
        if self._protected == entry.handle:
            self._protected = None
        heapq.heappush(self._free, entry.handle.slot)

    def _disposition(
        self,
        entry: _Entry,
        outcome: AdmissionOutcome,
        *,
        terminal_connection: bool = False,
    ) -> AdmissionDisposition:
        disposition = AdmissionDisposition(
            attempt_id=entry.attempt.attempt_id,
            outcome=outcome,
            terminal_connection=terminal_connection,
            close_code=1013 if terminal_connection else None,
        )
        self._release_slot(entry)
        return disposition

    def cancel(self, handle: AdmissionHandle) -> AdmissionDisposition:
        return self._disposition(self._require(handle), "cancelled")

    def expire_due(self) -> tuple[AdmissionDisposition, ...]:
        now = self._monotonic_ns()
        due: dict[AdmissionHandle, bool] = {}
        while self._deadlines and self._deadlines[0][0] <= now:
            _, kind, slot, generation = heapq.heappop(self._deadlines)
            handle = AdmissionHandle(slot, generation)
            if self._entry_or_none(handle) is not None:
                due[handle] = due.get(handle, False) or kind == 1
        dispositions: list[AdmissionDisposition] = []
        for handle, terminal_connection in due.items():
            entry = self._entry_or_none(handle)
            if entry is None:
                continue
            outcome: AdmissionOutcome = (
                "shed" if self._authority == "OPEN" else "unavailable"
            )
            dispositions.append(
                self._disposition(
                    entry,
                    outcome,
                    terminal_connection=terminal_connection,
                )
            )
        return tuple(dispositions)

    def _head(self, interval: int) -> _Entry | None:
        queue = self._queues[interval]
        while queue:
            entry = self._entries[queue[0]]
            if entry is not None and entry.state == "waiting":
                return entry
            queue.popleft()
        return None

    def _fits(self, entry: _Entry, capacity: AdmissionCapacity) -> bool:
        return (
            capacity.authority == "OPEN"
            and capacity.hard_headroom > 0
            and capacity.nominal_headroom_by_interval.get(
                entry.attempt.service_interval_ms,
                0,
            )
            > 0
        )

    def _launch(self, entry: _Entry) -> AdmissionDispatch:
        queue = self._queues[entry.attempt.service_interval_ms]
        while queue and queue[0] != entry.handle.slot:
            queue.popleft()
        if not queue:
            raise RuntimeError("admission FIFO lost its selected head")
        queue.popleft()
        entry.state = "submitted"
        self._submitted += 1
        if self._protected == entry.handle:
            self._protected = None
        return AdmissionDispatch(entry.handle, entry.attempt)

    def drain(
        self,
        capacity: AdmissionCapacity,
    ) -> tuple[AdmissionDispatch, ...]:
        self._last_examined = 0
        self._last_launched = 0
        if self._authority != "OPEN" or capacity.authority != "OPEN":
            return ()
        available = min(
            self._config.dispatch_budget,
            self._config.max_inflight_reserves - self._submitted,
            capacity.hard_headroom,
        )
        if available <= 0:
            return ()
        now = self._monotonic_ns()
        heads = tuple(
            head
            for interval in self._config.supported_intervals_ms
            if (head := self._head(interval)) is not None
        )
        self._last_examined = len(heads)
        protected = self._entry_or_none(self._protected)
        if protected is None:
            aged = tuple(
                entry
                for entry in heads
                if now - entry.enqueued_at_ns >= self._config.aging_threshold_ns
            )
            if aged:
                protected = min(aged, key=lambda entry: entry.sequence)
                self._protected = protected.handle
        if protected is not None:
            if not self._fits(protected, capacity):
                return ()
            launched = (self._launch(protected),)
            self._last_launched = 1
            return launched

        remaining_nominal = dict(capacity.nominal_headroom_by_interval)
        launched_items: list[AdmissionDispatch] = []
        while len(launched_items) < available:
            heads = tuple(
                head
                for interval in self._config.supported_intervals_ms
                if (head := self._head(interval)) is not None
            )
            feasible = tuple(
                entry
                for entry in heads
                if remaining_nominal.get(entry.attempt.service_interval_ms, 0)
                > 0
            )
            if not feasible:
                break
            selected = min(feasible, key=lambda entry: entry.sequence)
            launched_items.append(self._launch(selected))
            interval = selected.attempt.service_interval_ms
            remaining_nominal[interval] -= 1
        self._last_launched = len(launched_items)
        return tuple(launched_items)

    def complete_no_lease(self, handle: AdmissionHandle) -> None:
        self._release_slot(self._require(handle))

    def complete_admitted(
        self,
        handle: AdmissionHandle,
        *,
        lease: object,
        attached: bool,
        handoff: Callable[[AdmissionAttempt, object], None],
    ) -> None:
        del attached
        entry = self._require(handle)
        handoff(entry.attempt, lease)
        self._release_slot(entry)

    def authority_lost(self, *, engine_epoch: str) -> None:
        if engine_epoch != self._engine_epoch:
            raise ValueError("authority loss named the wrong engine epoch")
        self._authority = "UNHANDSHAKED"

    def authority_recovered(self, *, engine_epoch: str) -> None:
        if engine_epoch != self._engine_epoch:
            raise ValueError("authority recovery named the wrong engine epoch")
        self._authority = "OPEN"

    def engine_epoch_changed(
        self,
        engine_epoch: str,
    ) -> tuple[AdmissionDisposition, ...]:
        if not engine_epoch or engine_epoch == self._engine_epoch:
            raise ValueError("engine epoch change requires a fresh epoch")
        entries = tuple(entry for entry in self._entries if entry is not None)
        dispositions = tuple(
            self._disposition(entry, "unavailable") for entry in entries
        )
        self._engine_epoch = engine_epoch
        self._authority = "UNHANDSHAKED"
        return dispositions
