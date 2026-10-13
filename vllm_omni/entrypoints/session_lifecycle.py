# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared resettable lifecycle deadline for streaming sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
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
        finalization_stall_timeout_s: float | None = None,
    ) -> None:
        self._idle_timeout_s = idle_timeout_s
        self._finalization_timeout_s = finalization_timeout_s
        self._stall_timeout_s = finalization_stall_timeout_s
        self.absolute_deadline: float | None = None
        self._stall_deadline: float | None = None
        self._high_water = (-1, -1, -1)
        self.expired_clock: str | None = None
        self.receipt: dict[str, Any] = {}
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
        timeout_s = self._idle_timeout_s if kind == "idle" else self._finalization_timeout_s
        if timeout_s is None:
            return
        loop = asyncio.get_running_loop()
        if kind == "finalization":
            if self.absolute_deadline is None:
                self.absolute_deadline = loop.time() + float(timeout_s)
                self._stall_deadline = loop.time() + float(self._stall_timeout_s or timeout_s)
            assert self._stall_deadline is not None
            timeout_s = min(self.absolute_deadline, self._stall_deadline) - loop.time()
        self._handle = loop.call_later(
            max(0.0, float(timeout_s)),
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

        if self._closed or self._expired or generation != self._generation:
            return
        self._cancel_handle()
        self._expired = True
        self._expired_kind = kind
        if kind == "finalization":
            self.expired_clock = (
                "absolute"
                if self.absolute_deadline is not None
                and self.absolute_deadline <= (self._stall_deadline or self.absolute_deadline)
                else "stall"
            )
            self.receipt["clock"] = self.expired_clock
        self._expiry_task = asyncio.create_task(
            self._on_expire(kind),
            name=f"streaming-session-{kind}-expiry",
        )

    # @spec PORT-SESS-016, PORT-SESS-017
    def observe_progress(self, key: tuple[int, int, int]) -> bool:
        """Consume only a model-authorized, monotonically increasing key."""
        if self._closed or self._expired or key <= self._high_water:
            return False
        self._high_water = key
        if self.absolute_deadline is not None:
            timeout = self._stall_timeout_s or self._finalization_timeout_s
            assert timeout is not None
            self._stall_deadline = asyncio.get_running_loop().time() + float(timeout)
            self.arm("finalization")
        return True

    def configure_finalization(self, bounds: FinalizationBounds) -> None:
        if self.absolute_deadline is not None:
            return
        self._finalization_timeout_s = bounds.absolute_s
        self._stall_timeout_s = bounds.stall_s
        self.receipt.update(vars(bounds))

    def fail_finalization(self) -> None:
        """Checked-arithmetic failure takes the same terminal cleanup path."""
        self._generation += 1
        self._stall_deadline = self.absolute_deadline = asyncio.get_running_loop().time()
        self._deadline_reached(self._generation, "finalization")

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


@dataclass(frozen=True)
class FinalizationBounds:
    stall_s: float
    absolute_s: float
    stall_winner: str
    absolute_winner: str


# @spec PORT-SESS-016, PORT-SESS-017
def derive_finalization_bounds(
    *,
    regular_samples: int,
    chunk_samples: int,
    cadence_s: float,
    largest_eligibility_gap_s: float,
    response_bound_s: float,
    initial_eligibility_delay_s: float,
    final_tail_then_flush_s: float,
    adapter_margin_s: float,
    operator_stall_s: float,
    operator_absolute_s: float,
) -> FinalizationBounds:
    """Pure checked drain arithmetic; the residual belongs only to the tail."""
    if regular_samples < 0 or chunk_samples <= 0:
        raise ValueError("finalization_timeout: invalid sample bounds")
    positive = (cadence_s, response_bound_s, final_tail_then_flush_s, operator_stall_s, operator_absolute_s)
    nonnegative = (largest_eligibility_gap_s, initial_eligibility_delay_s, adapter_margin_s)
    if any(not math.isfinite(x) or x <= 0 for x in positive) or any(not math.isfinite(x) or x < 0 for x in nonnegative):
        raise ValueError("finalization_timeout: invalid clock operand")
    units = (regular_samples + chunk_samples - 1) // chunk_samples
    stall = max(largest_eligibility_gap_s + response_bound_s, final_tail_then_flush_s) + adapter_margin_s
    absolute = (
        units * max(cadence_s, response_bound_s)
        + initial_eligibility_delay_s
        + final_tail_then_flush_s
        + adapter_margin_s
    )
    if not math.isfinite(stall) or not math.isfinite(absolute):
        raise OverflowError("finalization_timeout: clock arithmetic overflow")
    return FinalizationBounds(
        max(operator_stall_s, stall),
        max(operator_absolute_s, absolute),
        "operator" if operator_stall_s >= stall else "derived",
        "operator" if operator_absolute_s >= absolute else "derived",
    )


@dataclass(frozen=True)
class FinalizationResponseBounds:
    response_s: float
    tail_s: float
    source: str
    unqualified: bool


# @spec PORT-SESS-016, PORT-SESS-017
def finalization_response_bounds(
    profile: Any,
    *,
    service_interval_ms: int,
    configured_s: float,
) -> FinalizationResponseBounds:
    """Resolve a per-session upper at the maximum admissible population."""
    from vllm_omni.engine.persistent_state_capacity import conservative_service_lower_bound

    if not math.isfinite(configured_s) or configured_s <= 0:
        raise ValueError("session_finalization_response_bound_s must be finite and positive")
    fallback = FinalizationResponseBounds(configured_s, 2 * configured_s, "configured", True)
    if profile is None:
        return fallback
    rate = conservative_service_lower_bound(profile, service_interval_ms=service_interval_ms)
    if rate is None:
        return fallback
    receipt = profile.receipt
    population = min(
        receipt.allocated_pool,
        receipt.execution_claim_ceiling,
        receipt.count_cap if receipt.count_cap is not None else receipt.allocated_pool,
    )
    if receipt.admission_policy == "profile":
        population = min(population, receipt.maximum_charged_population)
    tails = [
        sample.elapsed_ns
        for sample in profile.observations
        if sample.scenario_id == "final_tail_then_flush"
        and sample.post_jit
        and sample.active_population >= population
        and not getattr(sample, "censored", False)
    ]
    if not tails:
        return fallback
    qualified = profile.qualified and receipt.admission_policy == "profile"
    response = service_interval_ms / 1000 if qualified else float(service_interval_ms / 1000 / rate)
    tail = float(max(tails) / profile.derating_factor / 1_000_000_000)
    return FinalizationResponseBounds(
        response, tail, "profile" if receipt.admission_policy == "profile" else "advisory", not qualified
    )


# @spec PORT-SESS-016, PORT-SESS-017
def arm_session_finalization(deadline: SessionLifecycleDeadline, session: Any, service: Any) -> None:
    """Shared adapter integration; snapshot regular work once at acceptance."""
    if deadline.absolute_deadline is not None:
        return
    authority = session.accepted_audio
    runtime = getattr(service, "runtime_config", None)
    try:
        cadence_s = authority.chunk_samples / 16000
        response = finalization_response_bounds(
            getattr(service, "compiled_service_profile", None),
            service_interval_ms=round(cadence_s * 1000),
            configured_s=getattr(runtime, "session_finalization_response_bound_s", 10.0),
        )
        regular_samples, initial_delay_s = authority.finalization_work()
        bounds = derive_finalization_bounds(
            regular_samples=regular_samples,
            chunk_samples=authority.chunk_samples,
            cadence_s=cadence_s,
            largest_eligibility_gap_s=1.12,
            response_bound_s=response.response_s,
            initial_eligibility_delay_s=initial_delay_s,
            final_tail_then_flush_s=response.tail_s,
            adapter_margin_s=0.1,
            operator_stall_s=getattr(runtime, "session_finalization_stall_timeout", 10.0),
            operator_absolute_s=deadline._finalization_timeout_s or 40.0,
        )
        deadline.configure_finalization(bounds)
        deadline.receipt.update(vars(response))
        deadline.receipt["adapter_margin_s"] = 0.1
        deadline.arm("finalization")
        logging.getLogger(__name__).info("Finalization receipt %s", json.dumps(deadline.receipt, sort_keys=True))
    except (ValueError, OverflowError, ArithmeticError):
        deadline.receipt["derivation_error"] = True
        deadline.fail_finalization()


# @spec PORT-SESS-016
def consume_session_progress(deadline: SessionLifecycleDeadline, session: Any, output: Any) -> None:
    """Both adapters read the mandatory, status-clean model receipt."""
    authority = getattr(session, "accepted_audio", None)
    if authority is None:
        return
    for completion in getattr(output, "outputs", ()) or ():
        receipt = getattr(completion, "finalization_progress", None)
        if receipt is not None:
            request_id, generation, raw = receipt
            key = authority.observe_model_progress(request_id, generation, tuple(raw))
            if key is not None:
                deadline.observe_progress(key)


def claim_session_terminal(session: Any, cause: str, lease: Any) -> bool:
    authority = getattr(session, "accepted_audio", None)
    if authority is None:
        return True
    return bool(
        authority.claim_terminal(cause, lease_generation=getattr(lease, "generation", authority.lease_generation))
    )


def record_finalization_expiry(deadline: SessionLifecycleDeadline, observer: Any) -> None:
    clock = deadline.expired_clock or "absolute"
    deadline.receipt["clock"] = clock
    logging.getLogger(__name__).info("Finalization receipt %s", json.dumps(deadline.receipt, sort_keys=True))
    if observer is not None:
        from vllm_omni.metrics.streaming_transport import observe_safely

        callback = getattr(observer, "overflow", None)
        if callable(callback):
            observe_safely(callback, kind=f"finalization_{clock}")
