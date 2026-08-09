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
from fractions import Fraction
from typing import Any, Literal

StateLocation = Literal["resident", "offloaded", "absent"]

logger = logging.getLogger(__name__)


class PersistentStateServiceError(RuntimeError):
    """Base failure raised by the persistent-state admission service."""

    retryable = False


class PersistentStateBackpressure(PersistentStateServiceError):  # noqa: N818
    """Retryable per-operation shed; admission remains open."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after_ms: int | None = None,
        cause: str = "capacity",
        binding_authority: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_ms = retry_after_ms
        self.cause = cause
        self.binding_authority = binding_authority

    @property
    def telemetry_fields(self) -> dict[str, int | str]:
        fields: dict[str, int | str] = {"cause": self.cause}
        if self.binding_authority is not None:
            fields["binding_authority"] = self.binding_authority
        if self.retry_after_ms is not None:
            fields["retry_after_ms"] = self.retry_after_ms
        return fields


class PersistentStateCapacityExhausted(PersistentStateBackpressure):  # noqa: N818
    """The installed hard authorities cannot accept another lease."""


class PersistentStateServiceUnavailable(PersistentStateServiceError):  # noqa: N818
    """The service cannot currently establish authoritative state."""


class PersistentStateIndeterminate(PersistentStateServiceError):  # noqa: N818
    """A submitted operation could not be reconciled authoritatively."""


class PersistentStateUnsupportedServiceInterval(  # noqa: N818
    PersistentStateServiceError
):
    """The requested interval cannot be served by an empty pool."""

    retry_after_ms = None

    def __init__(
        self,
        *,
        requested_interval_ms: int,
        resolved_envelope: str,
    ) -> None:
        self.requested_interval_ms = requested_interval_ms
        self.resolved_envelope = resolved_envelope
        super().__init__(
            "persistent-state service interval "
            f"{requested_interval_ms}ms is unsupported by {resolved_envelope}"
        )


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
    service_interval_ms: int
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
        runtime_config: Any | None = None,
        admission_config: Any | None = None,
        host_fatal_callback: Callable[[BaseException], None] | None = None,
        compiled_service_profile: Any | None = None,
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
        self._runtime_config = runtime_config
        self._admission_config = admission_config
        self._host_fatal_callback = host_fatal_callback
        self._compiled_service_profile = compiled_service_profile
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
        # Live leases this mint issued and has not seen released - the
        # authority the handshake's orphan reconciliation compares engine
        # bindings against (PORT-STATE-023).
        self._live_leases: dict[str, StateLease] = {}
        self._pending_service_intervals: dict[str, int] = {}
        self._service_intervals: dict[str, int] = {}
        self._failed_releases: dict[str, _ReleaseCommand] = {}
        self._failed_release_started_at: float | None = None
        self._engine_epoch: str | None = None
        self._inventory: dict[str, Any] | None = None
        self._metrics_sink: Any | None = None
        self._fatal_error: BaseException | None = None
        self._host_fatal_reported = False
        self._recovery_task: asyncio.Task[None] | None = None

    @property
    def ready(self) -> bool:
        """Whether capability inventory is known and admission is open."""
        return (
            self._ready
            and self._admission_open
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

    @property
    def runtime_config(self) -> Any | None:
        """Return the resolved process envelope used by serving adapters."""

        return self._runtime_config

    @property
    def compiled_service_profile(self) -> Any | None:
        """Return the immutable cold-start service authority."""

        return self._compiled_service_profile

    def close_admission(self) -> None:
        """Close new admission without discarding cleanup authority."""
        self._admission_open = False

    def _demote(self) -> None:
        """Close admission AND require a fresh handshake to reopen.

        Recovery reuses the startup path (PORT-STATE-022): the next health
        probe re-runs the capability/inventory handshake and reopens
        admission only against a consistent engine snapshot. Retained
        tombstones survive, so exact-retry semantics hold across recovery.
        Only ``engine_epoch_changed`` latches terminally.
        """
        self.close_admission()
        self._ready = False
        self._arm_recovery()

    def _arm_recovery(self) -> None:
        """Start the one service-owned same-epoch recovery driver."""

        if self._admission_config is None or self._fatal_error is not None:
            return
        task = self._recovery_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._recovery_task = loop.create_task(
            self._recover(),
            name="persistent-state-recovery",
        )

    def _report_host_fatal(self, error: BaseException) -> None:
        if self._host_fatal_reported:
            return
        self._host_fatal_reported = True
        self._fatal_error = error
        self.close_admission()
        callback = self._host_fatal_callback
        if callback is not None:
            callback(error)

    async def _recover(self) -> None:
        """Re-run startup authority without remeasuring service capacity."""

        assert self._admission_config is not None
        backoffs = tuple(self._admission_config.recovery_backoff_s)
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        attempt = 0
        while self._fatal_error is None and not self.ready:
            try:
                await self._ensure_started()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if (
                    self._failed_releases
                    and loop.time() - started_at
                    >= self._admission_config.release_convergence_timeout_s
                ):
                    self._report_host_fatal(
                        PersistentStateServiceUnavailable(
                            "persistent-state release recovery did not converge: "
                            f"{error}"
                        )
                    )
                    return
                delay = backoffs[min(attempt, len(backoffs) - 1)]
                attempt += 1
                await asyncio.sleep(delay)
            else:
                return

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

    def _sync_service_projection(self) -> None:
        inventory = self._inventory
        if inventory is None:
            return
        capacity = max(1, int(inventory.get("effective_capacity", 1)))
        charged_count = len(self._pending_service_intervals) + len(
            self._service_intervals
        )
        inventory["execution_claims"] = charged_count
        inventory["charged_demand"] = float(Fraction(charged_count, capacity))
        profile = self._compiled_service_profile
        if profile is not None:
            receipt_hash = getattr(profile, "receipt_sha256", None)
            if receipt_hash is not None:
                inventory["service_profile_receipt_sha256"] = str(receipt_hash)
        self._project_inventory()

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
        recovery = self._recovery_task
        self._recovery_task = None
        if recovery is not None and not recovery.done():
            recovery.cancel()

    def engine_epoch_changed(self, engine_epoch: str) -> None:
        """Fail closed when a new engine epoch invalidates every live lease."""
        self.close_admission()
        self._ready = False
        self._live_leases.clear()
        self._fatal_error = PersistentStateServiceUnavailable(
            "persistent-state engine epoch changed; live leases are invalid"
        )
        self._engine_epoch = engine_epoch

    async def _reconcile_orphan_bindings(
        self, snapshot: dict[str, Any]
    ) -> None:
        """Release engine bindings no live lease owns (PORT-STATE-023).

        Runs inside the handshake, before admission opens. An orphan is a
        binding whose token this mint never issued or has already seen
        released, and whose claim horizon has expired - a live lease or
        an unexpired pending claim is never reclaimed. A claimed binding
        the engine still runs is skipped (its abort path owns it); any
        other release failure fails the handshake, so admission stays
        closed until the engine answers consistently.
        """
        for binding in snapshot.get("bindings", ()):
            token = str(binding["binding_token"])
            if token in self._live_leases:
                continue
            expires = binding.get("claim_expires_at")
            if expires is None or float(expires) > self._monotonic():
                continue
            payload = {
                "engine_epoch": str(binding["engine_epoch"]),
                "session_key": str(binding["session_key"]),
                "generation": int(binding["generation"]),
                "schema_id": str(binding["schema_id"]),
                "profile_id": str(binding["profile_id"]),
                "location": "resident",
                "binding_token": token,
            }
            try:
                raw = await self._stage_client.call_utility_async(
                    "persistent_state_release",
                    f"orphan-{token}",
                    payload,
                    "orphan_reconciliation",
                )
            except Exception as error:
                if "still running" in str(error):
                    continue
                raise PersistentStateServiceUnavailable(
                    f"orphan reconciliation failed for one binding: {error}"
                ) from error
            self._inventory["manager_revision"] = raw.get(
                "manager_revision",
                self._inventory.get("manager_revision"),
            )
            self._inventory["resident_count"] = raw.get(
                "resident_count",
                self._inventory.get("resident_count"),
            )

    # @spec PORT-STATE-014, PORT-STATE-022, PORT-STATE-023
    async def _reconcile_failed_releases(
        self,
        snapshot: dict[str, Any],
    ) -> None:
        """Exact-retry retained releases after engine terminality is proven."""

        inventory = self._inventory
        if inventory is None:
            raise PersistentStateServiceUnavailable(
                "persistent-state inventory is unavailable during release recovery"
            )
        bindings = {
            str(binding["binding_token"]): binding
            for binding in snapshot.get("bindings", ())
        }
        for token, command in tuple(self._failed_releases.items()):
            binding = bindings.get(token)
            if binding is not None and not bool(binding.get("terminal", False)):
                raise PersistentStateServiceUnavailable(
                    "persistent-state failed release is not terminally "
                    f"reconcilable for binding {token}"
                )
            try:
                raw = await self._stage_client.call_utility_async(
                    "persistent_state_release",
                    command.operation_id,
                    self._lease_payload(command.lease),
                    command.reason,
                )
                result = self._decode_release(raw)
            except Exception as error:
                raise PersistentStateServiceUnavailable(
                    f"persistent-state failed release did not converge: {error}"
                ) from error
            inventory["manager_revision"] = result.manager_revision
            inventory["resident_count"] = result.resident_count
            retry = asyncio.get_running_loop().create_future()
            retry.set_result(result)
            self._track_operation(
                command.operation_id,
                retry,
                retain_error=False,
            )
            self._failed_releases.pop(token, None)
            self._live_leases.pop(token, None)
            self._service_intervals.pop(token, None)
            self._sync_service_projection()
            snapshot["bindings"] = [
                candidate
                for candidate in snapshot.get("bindings", ())
                if str(candidate["binding_token"]) != token
            ]
        if not self._failed_releases:
            self._failed_release_started_at = None

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
                self._sync_service_projection()
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
                await self._reconcile_failed_releases(snapshot)
                await self._reconcile_orphan_bindings(snapshot)
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
        recovery = self._recovery_task
        if (
            recovery is not None
            and not recovery.done()
            and recovery is not asyncio.current_task()
        ):
            await asyncio.shield(recovery)
        else:
            try:
                await self._ensure_started()
            except PersistentStateServiceUnavailable:
                self._demote()
                raise
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
        service_interval_ms: int = 560,
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
            if self._tombstone_admission_blocked:
                self._observe_admission_rejection("capacity")
                raise PersistentStateBackpressure(
                    "persistent-state operation tombstone horizon is full",
                    cause="tombstone_horizon",
                )
            if not self.ready:
                self._observe_admission_rejection("unavailable")
                raise PersistentStateServiceUnavailable(
                    "persistent-state admission is closed"
                )
            self._pending_service_intervals[operation_id] = service_interval_ms
            self._sync_service_projection()
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
                        service_interval_ms,
                        future,
                    )
                )
            except asyncio.QueueFull as error:
                self._operations.pop(operation_id, None)
                self._operation_expires_at.pop(operation_id, None)
                self._pending_service_intervals.pop(operation_id, None)
                self._sync_service_projection()
                self._observe_admission_rejection("capacity")
                raise PersistentStateBackpressure(
                    "persistent-state reserve queue is full",
                    cause="bridge_full",
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
                self._demote()
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
                self._demote()
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
            self._demote()
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
                self._demote()
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
            self._demote()
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
            self._pending_service_intervals.pop(command.operation_id, None)
            self._sync_service_projection()
            mapped = self._map_error(
                error,
                service_interval_ms=command.service_interval_ms,
            )
            if isinstance(mapped, PersistentStateUnsupportedServiceInterval):
                rejection = "unsupported"
            elif isinstance(mapped, PersistentStateBackpressure):
                rejection = "capacity"
            else:
                rejection = "unavailable"
            self._observe_admission_rejection(rejection)
            if not command.future.done():
                command.future.set_exception(mapped)
        else:
            self._live_leases[result.lease.binding_token] = result.lease
            interval = self._pending_service_intervals.pop(
                command.operation_id,
                command.service_interval_ms,
            )
            self._service_intervals[result.lease.binding_token] = interval
            self._sync_service_projection()
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
            self._failed_releases[command.lease.binding_token] = command
            if self._failed_release_started_at is None:
                self._failed_release_started_at = asyncio.get_running_loop().time()
            self._demote()
            if not command.future.done():
                command.future.set_exception(mapped)
        else:
            self._live_leases.pop(command.lease.binding_token, None)
            self._failed_releases.pop(command.lease.binding_token, None)
            self._service_intervals.pop(command.lease.binding_token, None)
            self._sync_service_projection()
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

    def _map_error(
        self,
        error: BaseException,
        *,
        service_interval_ms: int | None = None,
    ) -> PersistentStateServiceError:
        if isinstance(error, PersistentStateServiceError):
            return error
        message = str(error)
        if "persistent_state_unsupported_service_interval" in message:
            return PersistentStateUnsupportedServiceInterval(
                requested_interval_ms=(
                    0 if service_interval_ms is None else service_interval_ms
                ),
                resolved_envelope=str(
                    (self._inventory or {}).get("profile_id", "installed profile")
                ),
            )
        if "persistent_state_horizon_exhausted" in message:
            return PersistentStateBackpressure(
                message,
                cause="tombstone_horizon",
            )
        if "capacity" in message.lower():
            # A fail-closed error that does not name its remedy costs an
            # operator a source dive. The resident limit is a deployment
            # choice whose default suits functional qualification, not
            # load, so say what it is and what raises it.
            inventory = self._inventory or {}
            effective = inventory.get("effective_capacity")
            configured = inventory.get("configured_limit")
            if effective is not None:
                message = (
                    f"{message} (effective capacity {effective} resident "
                    f"sessions, configured limit {configured}; raise "
                    "max_resident_sessions in --additional-config to admit "
                    "more concurrent sessions)"
                )
            return PersistentStateCapacityExhausted(
                message,
                cause="hard_pressure",
            )
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
