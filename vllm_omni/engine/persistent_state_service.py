# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API-process ownership for model-defined persistent-state leases.

The service deliberately uses EngineCore's existing named utility boundary.
It does not introduce a new wire protocol, expose physical slots, or make
the API process authoritative for resident state.  The engine-core manager
commits every transition; this object serializes admission and projects the
committed result back into the serving process.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

StateLocation = Literal["resident", "offloaded", "absent"]

logger = logging.getLogger(__name__)


class PersistentStateServiceError(RuntimeError):
    """Base failure raised by the persistent-state admission service."""


class PersistentStateCapacityExhausted(PersistentStateServiceError):  # noqa: N818
    """The manager has no admissible resident slot."""


class PersistentStateServiceUnavailable(PersistentStateServiceError):  # noqa: N818
    """The service cannot currently establish authoritative state."""


class PersistentStateIndeterminate(PersistentStateServiceError):  # noqa: N818
    """A submitted operation could not be reconciled authoritatively."""


@dataclass(frozen=True)
class StateLocationEvent:
    """Content-free projection of one committed location transition."""

    engine_epoch: str
    session_key: str
    generation: int
    location: StateLocation
    transition: str


@dataclass(frozen=True)
class StateLease:
    """Opaque logical binding; physical slot identity never crosses RPC."""

    engine_epoch: str
    session_key: str
    generation: int
    schema_id: str
    profile_id: str
    location: StateLocation
    binding_token: str


@dataclass(frozen=True)
class StateReserveResult:
    """Post-commit reserve projection returned by EngineCore."""

    operation_id: str
    manager_revision: int
    resident_count: int
    lease: StateLease
    location_event: StateLocationEvent


@dataclass(frozen=True)
class StateReleaseResult:
    """Post-commit release projection returned by EngineCore."""

    operation_id: str
    manager_revision: int
    resident_count: int
    location_event: StateLocationEvent


@dataclass(frozen=True)
class _ReserveCommand:
    operation_id: str
    session_key: str
    schema_id: str
    profile_id: str
    future: asyncio.Future[StateReserveResult]


@dataclass(frozen=True)
class _ReleaseCommand:
    operation_id: str
    lease: StateLease
    reason: str
    future: asyncio.Future[StateReleaseResult]


@dataclass(frozen=True)
class _BeginCleanupCommand:
    lease: StateLease
    future: asyncio.Future[bool]


class PersistentStateService:
    """Bounded, cleanup-priority bridge to one stage's state manager.

    Admission starts closed.  The first operation performs a capability
    inventory handshake through ``persistent_state_snapshot`` and only then
    opens admission.  Cleanup has its own queue and is always checked before
    reserve work, so a saturated admission lane cannot starve slot returns.
    Duplicate operation ids coalesce onto one future and therefore cannot
    commit parallel reserve or release attempts.
    """

    def __init__(
        self,
        stage_client: Any,
        *,
        reserve_queue_capacity: int = 64,
        cleanup_queue_capacity: int = 256,
        operation_timeout_s: float = 10.0,
        reconciliation_timeout_s: float = 30.0,
        tombstone_ttl_s: float = 3600.0,
        max_tombstones: int = 4096,
        pending_claim_timeout_s: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if reserve_queue_capacity <= 0 or cleanup_queue_capacity <= 0:
            raise ValueError("persistent-state queue capacities must be positive")
        if (
            operation_timeout_s <= 0
            or reconciliation_timeout_s <= 0
            or pending_claim_timeout_s <= 0
        ):
            raise ValueError("persistent-state timeouts must be positive")
        if tombstone_ttl_s <= 0:
            raise ValueError("persistent-state tombstone TTL must be positive")
        if max_tombstones <= 0:
            raise ValueError("persistent-state tombstone count must be positive")
        self._stage_client = stage_client
        self.reserve_queue: asyncio.Queue[_ReserveCommand] = asyncio.Queue(
            maxsize=reserve_queue_capacity
        )
        self.cleanup_queue: asyncio.Queue[
            _ReleaseCommand | _BeginCleanupCommand
        ] = asyncio.Queue(
            maxsize=cleanup_queue_capacity
        )
        self._operation_timeout_s = operation_timeout_s
        self._reconciliation_timeout_s = reconciliation_timeout_s
        self._tombstone_ttl_s = tombstone_ttl_s
        self._max_tombstones = max_tombstones
        self._pending_claim_timeout_s = pending_claim_timeout_s
        self._monotonic = monotonic
        self._operations: dict[str, asyncio.Future[Any]] = {}
        self._operation_expires_at: dict[str, float] = {}
        self._pending_cleanup_claims: dict[str, asyncio.Future[bool]] = {}
        self._queue_event = asyncio.Event()
        self._startup_lock: asyncio.Lock | None = None
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._ready = False
        self._admission_open = False
        self._tombstone_admission_blocked = False
        self._engine_epoch: str | None = None
        self._inventory: dict[str, Any] | None = None
        self._metrics_sink: Any | None = None
        self._fatal_error: BaseException | None = None

    @property
    def ready(self) -> bool:
        """Whether capability inventory is known and admission is open."""
        return (
            self._ready
            and self._admission_open
            and not self._tombstone_admission_blocked
            and self._fatal_error is None
        )

    @property
    def inventory(self) -> dict[str, Any] | None:
        """Return a copy of the current content-free capability inventory."""
        return None if self._inventory is None else dict(self._inventory)

    @property
    def pending_claim_timeout_s(self) -> float:
        """Return the fingerprinted reserved-before-claim recovery bound."""

        return self._pending_claim_timeout_s

    def close_admission(self) -> None:
        """Close new admission without discarding cleanup authority."""
        self._admission_open = False

    def _prune_operation_tombstones(self) -> None:
        """Forget only completed operations whose retry horizon expired."""

        now = self._monotonic()
        expired = [
            operation_id
            for operation_id, expires_at in self._operation_expires_at.items()
            if expires_at <= now
        ]
        for operation_id in expired:
            future = self._operations.get(operation_id)
            if future is not None and future.done():
                self._operations.pop(operation_id, None)
                self._operation_expires_at.pop(operation_id, None)

    def _refresh_tombstone_admission(self) -> None:
        """Reserve release-tombstone headroom for every resident lease."""

        self._prune_operation_tombstones()
        resident_count = 0
        if self._inventory is not None:
            resident_count = int(self._inventory.get("resident_count", 0))
        # A new session consumes one reserve tombstone immediately and one
        # release tombstone eventually. Existing resident leases each retain
        # one cleanup slot. This invariant lets release remain available even
        # when new admission is closed by the operation horizon.
        self._tombstone_admission_blocked = (
            len(self._operation_expires_at) + resident_count + 2
            > self._max_tombstones
        )

    def _track_operation(
        self,
        operation_id: str,
        future: asyncio.Future[Any],
        *,
        retain_error: bool,
    ) -> None:
        """Retain one completed future as the exact retry tombstone."""

        self._operations[operation_id] = future

        def completed(_future: asyncio.Future[Any]) -> None:
            if (
                _future.cancelled()
                or (
                    not retain_error
                    and _future.exception() is not None
                )
            ):
                if self._operations.get(operation_id) is _future:
                    self._operations.pop(operation_id, None)
                    self._operation_expires_at.pop(operation_id, None)
                return
            self._operation_expires_at[operation_id] = (
                self._monotonic() + self._tombstone_ttl_s
            )
            self._refresh_tombstone_admission()

        future.add_done_callback(completed)

    def install_metrics(self, metrics_sink: Any) -> None:
        """Attach the app-owned non-authoritative metric sink exactly once."""
        if self._metrics_sink is not None:
            raise RuntimeError("persistent-state metrics already installed")
        self._metrics_sink = metrics_sink
        self._project_inventory()

    def _observe_metrics(self, method_name: str, *args: Any) -> None:
        sink = self._metrics_sink
        if sink is None:
            return
        try:
            getattr(sink, method_name)(*args)
        except Exception:
            logger.exception(
                "persistent-state metric observation failed; serving is unaffected"
            )

    def _project_inventory(self) -> None:
        inventory = self._inventory
        if inventory is None or self._metrics_sink is None:
            return
        self._observe_metrics(
            "observe_persistent_state_slots",
            str(inventory["stage"]),
            str(inventory["replica"]),
            {
                "resident": int(inventory["resident_count"]),
                "safety_reserve": int(inventory["safety_reserve"]),
                "physical_capacity": int(inventory["physical_capacity"]),
                "configured_limit": int(inventory["configured_limit"]),
                "effective_capacity": int(inventory["effective_capacity"]),
            },
        )

    def _observe_admission_rejection(self, reason: str) -> None:
        self._observe_metrics("inc_admission_rejection", reason)

    def _update_projection(self, *, manager_revision: int, resident_count: int) -> None:
        if self._inventory is None:
            return
        self._inventory["manager_revision"] = manager_revision
        self._inventory["resident_count"] = resident_count
        self._project_inventory()

    def shutdown(self) -> None:
        """Close admission and cancel the API-owned dispatcher task."""

        self.close_admission()
        task = self._dispatcher_task
        self._dispatcher_task = None
        if task is not None and not task.done():
            task.cancel()

    def engine_epoch_changed(self, engine_epoch: str) -> None:
        """Fail closed when a new engine epoch invalidates every live lease."""
        self.close_admission()
        self._ready = False
        self._fatal_error = PersistentStateServiceUnavailable(
            "persistent-state engine epoch changed; live leases are invalid"
        )
        self._engine_epoch = engine_epoch

    async def _ensure_started(self) -> None:
        if self._startup_lock is None:
            self._startup_lock = asyncio.Lock()
        async with self._startup_lock:
            if self._fatal_error is not None:
                raise PersistentStateServiceUnavailable(str(self._fatal_error))
            if not self._ready:
                snapshot = await self._stage_client.call_utility_async(
                    "persistent_state_snapshot"
                )
                capabilities = set(snapshot.get("capabilities", ()))
                if "resident" not in capabilities:
                    raise PersistentStateServiceUnavailable(
                        "persistent-state capability inventory lacks resident state"
                    )
                required_inventory = {
                    "manager_revision",
                    "resident_count",
                    "physical_capacity",
                    "safety_reserve",
                    "configured_limit",
                    "effective_capacity",
                    "stage",
                    "replica",
                    "schema_id",
                    "profile_id",
                }
                missing_inventory = required_inventory.difference(snapshot)
                if missing_inventory:
                    raise PersistentStateServiceUnavailable(
                        "persistent-state capability inventory is incomplete: "
                        f"missing {sorted(missing_inventory)}"
                    )
                engine_epoch = str(snapshot["engine_epoch"])
                if self._engine_epoch is not None and engine_epoch != self._engine_epoch:
                    self.engine_epoch_changed(engine_epoch)
                    raise PersistentStateServiceUnavailable(
                        "persistent-state engine epoch changed during snapshot"
                    )
                self._engine_epoch = engine_epoch
                self._inventory = dict(snapshot)
                if self._max_tombstones < 2 * int(snapshot["effective_capacity"]):
                    raise PersistentStateServiceUnavailable(
                        "persistent-state tombstone count cannot reserve cleanup "
                        "headroom for effective capacity"
                    )
                advertised_ttl = snapshot.get(
                    "persistent_state_tombstone_ttl_s"
                )
                advertised_max = snapshot.get(
                    "persistent_state_max_tombstones"
                )
                if advertised_ttl is not None and float(advertised_ttl) != self._tombstone_ttl_s:
                    raise PersistentStateServiceUnavailable(
                        "persistent-state tombstone TTL disagrees across processes"
                    )
                if advertised_max is not None and int(advertised_max) != self._max_tombstones:
                    raise PersistentStateServiceUnavailable(
                        "persistent-state tombstone count disagrees across processes"
                    )
                self._ready = True
                self._admission_open = True
                self._refresh_tombstone_admission()
                self._project_inventory()
            if self._dispatcher_task is None or self._dispatcher_task.done():
                self._dispatcher_task = asyncio.create_task(
                    self._dispatch(), name="persistent-state-dispatch"
                )

    async def check_health(self) -> None:
        """Require a completed inventory handshake and open admission."""
        await self._ensure_started()
        self._refresh_tombstone_admission()
        if not self.ready:
            raise PersistentStateServiceUnavailable(
                "persistent-state admission is closed"
            )

    async def check_admission(self) -> None:
        """Check health for one admission attempt and classify its denial."""
        try:
            await self.check_health()
        except PersistentStateServiceUnavailable:
            self._observe_admission_rejection("unavailable")
            raise

    def _coalesced_future(
        self, operation_id: str
    ) -> asyncio.Future[Any] | None:
        self._prune_operation_tombstones()
        return self._operations.get(operation_id)

    async def reserve(
        self,
        *,
        operation_id: str,
        session_key: str,
        schema_id: str,
        profile_id: str,
    ) -> StateLease:
        """Reserve once, reconciling the same operation after timeout."""
        try:
            await self._ensure_started()
        except PersistentStateServiceUnavailable:
            self._observe_admission_rejection("unavailable")
            raise
        shared = self._coalesced_future(operation_id)
        if shared is None:
            self._refresh_tombstone_admission()
            if not self.ready:
                self._observe_admission_rejection("unavailable")
                reason = (
                    "persistent-state operation tombstone horizon is full"
                    if self._tombstone_admission_blocked
                    else "persistent-state admission is closed"
                )
                raise PersistentStateServiceUnavailable(reason)
            future: asyncio.Future[StateReserveResult] = (
                asyncio.get_running_loop().create_future()
            )
            self._track_operation(
                operation_id,
                future,
                retain_error=True,
            )
            try:
                self.reserve_queue.put_nowait(
                    _ReserveCommand(
                        operation_id,
                        session_key,
                        schema_id,
                        profile_id,
                        future,
                    )
                )
            except asyncio.QueueFull as error:
                self._operations.pop(operation_id, None)
                self._operation_expires_at.pop(operation_id, None)
                self._observe_admission_rejection("unavailable")
                raise PersistentStateServiceUnavailable(
                    "persistent-state reserve queue is full"
                ) from error
            self._queue_event.set()
            shared = future
        try:
            result = await asyncio.wait_for(
                asyncio.shield(shared), timeout=self._operation_timeout_s
            )
        except asyncio.TimeoutError:
            # Submission may already be committed.  Close admission and
            # reconcile this same operation id; never mint a parallel attempt.
            self.close_admission()
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(shared),
                    timeout=self._reconciliation_timeout_s,
                )
            except asyncio.TimeoutError as error:
                raise PersistentStateIndeterminate(
                    "persistent-state reserve reconciliation remained indeterminate"
                ) from error
            self._admission_open = True
        if not isinstance(result, StateReserveResult):
            raise PersistentStateServiceUnavailable(
                "persistent-state reserve returned an invalid result"
            )
        self._update_projection(
            manager_revision=result.manager_revision,
            resident_count=result.resident_count,
        )
        return result.lease

    async def release(
        self,
        *,
        operation_id: str,
        lease: StateLease,
        reason: str,
    ) -> StateReleaseResult:
        """Return a lease once through the cleanup-priority lane."""
        await self._ensure_started()
        shared = self._coalesced_future(operation_id)
        if shared is None:
            future: asyncio.Future[StateReleaseResult] = (
                asyncio.get_running_loop().create_future()
            )
            self._track_operation(
                operation_id,
                future,
                retain_error=False,
            )
            try:
                self.cleanup_queue.put_nowait(
                    _ReleaseCommand(operation_id, lease, reason, future)
                )
            except asyncio.QueueFull as error:
                self._operations.pop(operation_id, None)
                self._operation_expires_at.pop(operation_id, None)
                self.close_admission()
                raise PersistentStateServiceUnavailable(
                    "persistent-state cleanup queue is full"
                ) from error
            self._queue_event.set()
            shared = future
        try:
            result = await asyncio.wait_for(
                asyncio.shield(shared), timeout=self._reconciliation_timeout_s
            )
        except asyncio.TimeoutError as error:
            self.close_admission()
            raise PersistentStateIndeterminate(
                "persistent-state release reconciliation remained indeterminate"
            ) from error
        if not isinstance(result, StateReleaseResult):
            raise PersistentStateServiceUnavailable(
                "persistent-state release returned an invalid result"
            )
        self._update_projection(
            manager_revision=result.manager_revision,
            resident_count=result.resident_count,
        )
        return result

    async def claim_pending_cleanup(self, lease: StateLease) -> bool:
        """Atomically claim API cleanup only while a lease is unclaimed."""

        await self._ensure_started()
        binding_token = lease.binding_token
        shared = self._pending_cleanup_claims.get(binding_token)
        if shared is None:
            future: asyncio.Future[bool] = (
                asyncio.get_running_loop().create_future()
            )
            self._pending_cleanup_claims[binding_token] = future

            def completed(done: asyncio.Future[bool]) -> None:
                if self._pending_cleanup_claims.get(binding_token) is done:
                    self._pending_cleanup_claims.pop(binding_token, None)

            future.add_done_callback(completed)
            try:
                self.cleanup_queue.put_nowait(
                    _BeginCleanupCommand(lease, future)
                )
            except asyncio.QueueFull as error:
                self._pending_cleanup_claims.pop(binding_token, None)
                self.close_admission()
                raise PersistentStateServiceUnavailable(
                    "persistent-state cleanup queue is full"
                ) from error
            self._queue_event.set()
            shared = future
        try:
            return await asyncio.wait_for(
                asyncio.shield(shared),
                timeout=self._reconciliation_timeout_s,
            )
        except asyncio.TimeoutError as error:
            self.close_admission()
            raise PersistentStateIndeterminate(
                "persistent-state pending cleanup claim remained indeterminate"
            ) from error

    async def _dispatch(self) -> None:
        """Drain cleanup_queue before reserve_queue on every turn."""
        while True:
            await self._queue_event.wait()
            self._queue_event.clear()
            while True:
                command: (
                    _ReserveCommand
                    | _ReleaseCommand
                    | _BeginCleanupCommand
                    | None
                ) = None
                try:
                    command = self.cleanup_queue.get_nowait()
                except asyncio.QueueEmpty:
                    try:
                        command = self.reserve_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if isinstance(command, _ReleaseCommand):
                    await self._execute_release(command)
                    self.cleanup_queue.task_done()
                elif isinstance(command, _BeginCleanupCommand):
                    await self._execute_begin_cleanup(command)
                    self.cleanup_queue.task_done()
                else:
                    await self._execute_reserve(command)
                    self.reserve_queue.task_done()
                if not self.cleanup_queue.empty() or not self.reserve_queue.empty():
                    self._queue_event.set()

    async def _execute_reserve(self, command: _ReserveCommand) -> None:
        try:
            raw = await self._stage_client.call_utility_async(
                "persistent_state_reserve",
                command.operation_id,
                command.session_key,
                command.schema_id,
                command.profile_id,
            )
            result = self._decode_reserve(raw)
            if result.lease.engine_epoch != self._engine_epoch:
                self.engine_epoch_changed(result.lease.engine_epoch)
                raise PersistentStateServiceUnavailable(
                    "persistent-state reserve crossed an engine epoch"
                )
        except Exception as error:
            mapped = self._map_error(error)
            self._observe_admission_rejection(
                "capacity"
                if isinstance(mapped, PersistentStateCapacityExhausted)
                else "unavailable"
            )
            if not command.future.done():
                command.future.set_exception(mapped)
        else:
            if not command.future.done():
                command.future.set_result(result)

    async def _execute_release(self, command: _ReleaseCommand) -> None:
        try:
            raw = await self._stage_client.call_utility_async(
                "persistent_state_release",
                command.operation_id,
                self._lease_payload(command.lease),
                command.reason,
            )
            result = self._decode_release(raw)
            if result.location_event.engine_epoch != self._engine_epoch:
                self.engine_epoch_changed(result.location_event.engine_epoch)
                raise PersistentStateServiceUnavailable(
                    "persistent-state release crossed an engine epoch"
                )
        except Exception as error:
            mapped = self._map_error(error)
            if not command.future.done():
                command.future.set_exception(mapped)
        else:
            if not command.future.done():
                command.future.set_result(result)

    async def _execute_begin_cleanup(
        self,
        command: _BeginCleanupCommand,
    ) -> None:
        try:
            won = await self._stage_client.call_utility_async(
                "persistent_state_begin_pending_cleanup",
                self._lease_payload(command.lease),
            )
            if not isinstance(won, bool):
                raise PersistentStateServiceUnavailable(
                    "persistent-state pending cleanup returned an invalid result"
                )
        except Exception as error:
            mapped = self._map_error(error)
            if not command.future.done():
                command.future.set_exception(mapped)
        else:
            if not command.future.done():
                command.future.set_result(won)

    @staticmethod
    def _map_error(error: BaseException) -> PersistentStateServiceError:
        if isinstance(error, PersistentStateServiceError):
            return error
        message = str(error)
        if "capacity" in message.lower():
            return PersistentStateCapacityExhausted(message)
        return PersistentStateServiceUnavailable(message)

    @staticmethod
    def _lease_payload(lease: StateLease) -> dict[str, Any]:
        return {
            "engine_epoch": lease.engine_epoch,
            "session_key": lease.session_key,
            "generation": lease.generation,
            "schema_id": lease.schema_id,
            "profile_id": lease.profile_id,
            "location": lease.location,
            "binding_token": lease.binding_token,
        }

    @staticmethod
    def _decode_location(raw: dict[str, Any]) -> StateLocationEvent:
        return StateLocationEvent(
            engine_epoch=str(raw["engine_epoch"]),
            session_key=str(raw["session_key"]),
            generation=int(raw["generation"]),
            location=raw["location"],
            transition=str(raw["transition"]),
        )

    @classmethod
    def _decode_reserve(cls, raw: dict[str, Any]) -> StateReserveResult:
        lease_raw = raw["lease"]
        lease = StateLease(
            engine_epoch=str(lease_raw["engine_epoch"]),
            session_key=str(lease_raw["session_key"]),
            generation=int(lease_raw["generation"]),
            schema_id=str(lease_raw["schema_id"]),
            profile_id=str(lease_raw["profile_id"]),
            location=lease_raw["location"],
            binding_token=str(lease_raw["binding_token"]),
        )
        return StateReserveResult(
            operation_id=str(raw["operation_id"]),
            manager_revision=int(raw["manager_revision"]),
            resident_count=int(raw["resident_count"]),
            lease=lease,
            location_event=cls._decode_location(raw["location_event"]),
        )

    @classmethod
    def _decode_release(cls, raw: dict[str, Any]) -> StateReleaseResult:
        return StateReleaseResult(
            operation_id=str(raw["operation_id"]),
            manager_revision=int(raw["manager_revision"]),
            resident_count=int(raw["resident_count"]),
            location_event=cls._decode_location(raw["location_event"]),
        )
