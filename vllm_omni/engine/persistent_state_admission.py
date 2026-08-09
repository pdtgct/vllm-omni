# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded pre-audio admission for persistent streaming sessions."""

from __future__ import annotations

import hashlib
import heapq
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
            len(self.supported_intervals_ms) != 5
            or len(set(self.supported_intervals_ms))
            != len(self.supported_intervals_ms)
            or any(value <= 0 for value in self.supported_intervals_ms)
        ):
            raise ValueError(
                "exactly five supported intervals must be unique and positive"
            )


@dataclass(frozen=True)
class AdmissionAttempt:
    attempt_id: str
    connection_id: str
    connection_handle: str
    operation_id: str
    service_interval_ms: int
    admission_deadline_ns: int
    unadmitted_deadline_ns: int
    resource_key: str = ""
    schema_id: str = ""
    profile_id: str = ""


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
    handle: AdmissionHandle | None = None
    service_interval_ms: int | None = None
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
    """One mutable, startup-preallocated waiter slab entry."""

    attempt: AdmissionAttempt | None = None
    handle: AdmissionHandle | None = None
    enqueued_at_ns: int = 0
    sequence: int = 0
    state: Literal["free", "waiting", "submitted"] = "free"
    queue_previous: int = -1
    queue_next: int = -1
    deadline_ns: int = 0
    deadline_closes_connection: bool = False


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
        # The waiter nodes, intrusive queue links, and deadline heap storage
        # are all allocated once. Enqueue mutates one free slab entry; it does
        # not allocate a deque node or retain a lazy deadline tombstone.
        self._entries = [_Entry() for _ in range(config.waiter_capacity)]
        self._generations = [0] * config.waiter_capacity
        self._free = list(range(config.waiter_capacity))
        heapq.heapify(self._free)
        self._queue_heads = dict.fromkeys(config.supported_intervals_ms, -1)
        self._queue_tails = dict.fromkeys(config.supported_intervals_ms, -1)
        self._waiting_counts = dict.fromkeys(config.supported_intervals_ms, 0)
        self._submitted_counts = dict.fromkeys(config.supported_intervals_ms, 0)
        self._connections: dict[str, AdmissionHandle] = {}
        self._deadline_heap = [-1] * config.waiter_capacity
        self._deadline_positions = [-1] * config.waiter_capacity
        self._deadline_size = 0
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
                None
                if protected is None or protected.attempt is None
                else protected.attempt.attempt_id
            ),
            authority=self._authority,
            last_drain_examined_heads=self._last_examined,
            last_drain_launched=self._last_launched,
        )

    def _entry_or_none(self, handle: AdmissionHandle | None) -> _Entry | None:
        if handle is None or not 0 <= handle.slot < len(self._entries):
            return None
        entry = self._entries[handle.slot]
        if entry.state == "free" or entry.handle != handle:
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

    @property
    def next_deadline_ns(self) -> int | None:
        """Return the earliest live waiter deadline without allocating."""

        if self._deadline_size == 0:
            return None
        slot = self._deadline_heap[0]
        return self._entries[slot].deadline_ns

    @property
    def pending_counts(self) -> Mapping[int, Mapping[str, int]]:
        """Project five fixed counters without scanning waiter entries."""

        return {
            interval: {
                "waiting": self._waiting_counts[interval],
                "submitted": self._submitted_counts[interval],
                "reconciling": 0,
            }
            for interval in self._config.supported_intervals_ms
        }

    def _deadline_less(self, left_slot: int, right_slot: int) -> bool:
        left = self._entries[left_slot]
        right = self._entries[right_slot]
        return (left.deadline_ns, left.sequence) < (
            right.deadline_ns,
            right.sequence,
        )

    def _deadline_swap(self, left: int, right: int) -> None:
        left_slot = self._deadline_heap[left]
        right_slot = self._deadline_heap[right]
        self._deadline_heap[left], self._deadline_heap[right] = (
            right_slot,
            left_slot,
        )
        self._deadline_positions[left_slot] = right
        self._deadline_positions[right_slot] = left

    def _deadline_sift_up(self, position: int) -> int:
        while position > 0:
            parent = (position - 1) // 2
            if not self._deadline_less(
                self._deadline_heap[position],
                self._deadline_heap[parent],
            ):
                break
            self._deadline_swap(position, parent)
            position = parent
        return position

    def _deadline_sift_down(self, position: int) -> None:
        while True:
            left = 2 * position + 1
            if left >= self._deadline_size:
                return
            right = left + 1
            child = left
            if right < self._deadline_size and self._deadline_less(
                self._deadline_heap[right],
                self._deadline_heap[left],
            ):
                child = right
            if not self._deadline_less(
                self._deadline_heap[child],
                self._deadline_heap[position],
            ):
                return
            self._deadline_swap(position, child)
            position = child

    def _deadline_insert(self, slot: int) -> None:
        if self._deadline_positions[slot] != -1:
            raise RuntimeError("admission deadline already indexed")
        position = self._deadline_size
        self._deadline_size += 1
        self._deadline_heap[position] = slot
        self._deadline_positions[slot] = position
        self._deadline_sift_up(position)

    def _deadline_remove(self, slot: int) -> None:
        position = self._deadline_positions[slot]
        if position == -1:
            return
        self._deadline_size -= 1
        last_slot = self._deadline_heap[self._deadline_size]
        self._deadline_heap[self._deadline_size] = -1
        self._deadline_positions[slot] = -1
        if position == self._deadline_size:
            return
        self._deadline_heap[position] = last_slot
        self._deadline_positions[last_slot] = position
        position = self._deadline_sift_up(position)
        self._deadline_sift_down(position)

    def _queue_append(self, entry: _Entry) -> None:
        attempt = entry.attempt
        handle = entry.handle
        if attempt is None or handle is None:
            raise RuntimeError("cannot queue an empty admission entry")
        interval = attempt.service_interval_ms
        tail = self._queue_tails[interval]
        entry.queue_previous = tail
        entry.queue_next = -1
        if tail == -1:
            self._queue_heads[interval] = handle.slot
        else:
            self._entries[tail].queue_next = handle.slot
        self._queue_tails[interval] = handle.slot

    def _queue_remove(self, entry: _Entry) -> None:
        attempt = entry.attempt
        handle = entry.handle
        if attempt is None or handle is None:
            raise RuntimeError("cannot unlink an empty admission entry")
        interval = attempt.service_interval_ms
        previous = entry.queue_previous
        following = entry.queue_next
        if previous == -1:
            self._queue_heads[interval] = following
        else:
            self._entries[previous].queue_next = following
        if following == -1:
            self._queue_tails[interval] = previous
        else:
            self._entries[following].queue_previous = previous
        entry.queue_previous = -1
        entry.queue_next = -1

    def enqueue(self, attempt: AdmissionAttempt) -> AdmissionHandle:
        if attempt.service_interval_ms not in self._queue_heads:
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
        entry = self._entries[slot]
        entry.attempt = attempt
        entry.handle = handle
        entry.enqueued_at_ns = self._monotonic_ns()
        entry.sequence = self._sequence
        entry.state = "waiting"
        entry.deadline_ns = min(
            attempt.admission_deadline_ns,
            attempt.unadmitted_deadline_ns,
        )
        entry.deadline_closes_connection = (
            attempt.unadmitted_deadline_ns <= attempt.admission_deadline_ns
        )
        self._connections[attempt.connection_id] = handle
        self._queue_append(entry)
        self._waiting_counts[attempt.service_interval_ms] += 1
        self._deadline_insert(slot)
        return handle

    def _release_slot(self, entry: _Entry) -> None:
        attempt = entry.attempt
        handle = entry.handle
        if attempt is None or handle is None:
            raise RuntimeError("cannot release an empty admission entry")
        if entry.state == "waiting":
            self._queue_remove(entry)
            self._waiting_counts[attempt.service_interval_ms] -= 1
        if entry.state == "submitted":
            self._submitted -= 1
            self._submitted_counts[attempt.service_interval_ms] -= 1
        self._deadline_remove(handle.slot)
        self._connections.pop(attempt.connection_id, None)
        if self._protected == handle:
            self._protected = None
        slot = handle.slot
        entry.attempt = None
        entry.handle = None
        entry.enqueued_at_ns = 0
        entry.sequence = 0
        entry.state = "free"
        entry.queue_previous = -1
        entry.queue_next = -1
        entry.deadline_ns = 0
        entry.deadline_closes_connection = False
        heapq.heappush(self._free, slot)

    def _disposition(
        self,
        entry: _Entry,
        outcome: AdmissionOutcome,
        *,
        terminal_connection: bool = False,
    ) -> AdmissionDisposition:
        if entry.attempt is None:
            raise RuntimeError("cannot disposition an empty admission entry")
        disposition = AdmissionDisposition(
            attempt_id=entry.attempt.attempt_id,
            outcome=outcome,
            handle=entry.handle,
            service_interval_ms=entry.attempt.service_interval_ms,
            terminal_connection=terminal_connection,
            close_code=1013 if terminal_connection else None,
        )
        self._release_slot(entry)
        return disposition

    def cancel(self, handle: AdmissionHandle) -> AdmissionDisposition:
        return self._disposition(self._require(handle), "cancelled")

    def detach(self, handle: AdmissionHandle) -> AdmissionDisposition | None:
        """Cancel waiting work, but retain submitted resource authority."""

        entry = self._require(handle)
        if entry.state == "submitted":
            return None
        return self._disposition(entry, "cancelled")

    def expire_due(self) -> tuple[AdmissionDisposition, ...]:
        now = self._monotonic_ns()
        dispositions: list[AdmissionDisposition] = []
        while self._deadline_size:
            slot = self._deadline_heap[0]
            entry = self._entries[slot]
            if entry.deadline_ns > now:
                break
            terminal_connection = entry.deadline_closes_connection
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
        slot = self._queue_heads[interval]
        return None if slot == -1 else self._entries[slot]

    def _fits(self, entry: _Entry, capacity: AdmissionCapacity) -> bool:
        if entry.attempt is None:
            return False
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
        attempt = entry.attempt
        handle = entry.handle
        if attempt is None or handle is None:
            raise RuntimeError("cannot launch an empty admission entry")
        if self._queue_heads[attempt.service_interval_ms] != handle.slot:
            raise RuntimeError("admission FIFO lost its selected head")
        self._queue_remove(entry)
        self._waiting_counts[attempt.service_interval_ms] -= 1
        # The admission-wait deadline ends at submission. Resource
        # terminality, not a stale timer, owns this slab entry thereafter.
        self._deadline_remove(handle.slot)
        entry.state = "submitted"
        self._submitted += 1
        self._submitted_counts[attempt.service_interval_ms] += 1
        if self._protected == handle:
            self._protected = None
        return AdmissionDispatch(handle, attempt)

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
        protected = self._entry_or_none(self._protected)
        if protected is None:
            for interval in self._config.supported_intervals_ms:
                entry = self._head(interval)
                if entry is None:
                    continue
                self._last_examined += 1
                if (
                    now - entry.enqueued_at_ns
                    >= self._config.aging_threshold_ns
                    and (
                        protected is None
                        or entry.sequence < protected.sequence
                    )
                ):
                    protected = entry
            if protected is not None:
                assert protected.handle is not None
                self._protected = protected.handle
        if protected is not None:
            if not self._fits(protected, capacity):
                return ()
            launched = (self._launch(protected),)
            self._last_launched = 1
            return launched

        remaining_nominal = [
            capacity.nominal_headroom_by_interval.get(interval, 0)
            for interval in self._config.supported_intervals_ms
        ]
        launched_items: list[AdmissionDispatch] = []
        while len(launched_items) < available:
            selected: _Entry | None = None
            selected_index = -1
            for index, interval in enumerate(
                self._config.supported_intervals_ms
            ):
                if remaining_nominal[index] <= 0:
                    continue
                entry = self._head(interval)
                if entry is not None and (
                    selected is None or entry.sequence < selected.sequence
                ):
                    selected = entry
                    selected_index = index
            if selected is None:
                break
            launched_items.append(self._launch(selected))
            remaining_nominal[selected_index] -= 1
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
        if entry.attempt is None:
            raise RuntimeError("cannot hand off an empty admission entry")
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
        entries = tuple(
            entry for entry in self._entries if entry.state != "free"
        )
        dispositions = tuple(
            self._disposition(entry, "unavailable") for entry in entries
        )
        self._engine_epoch = engine_epoch
        self._authority = "UNHANDSHAKED"
        return dispositions
