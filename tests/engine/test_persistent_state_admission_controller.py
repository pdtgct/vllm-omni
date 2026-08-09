# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the bounded pre-audio admission controller."""

from __future__ import annotations

import dataclasses
import importlib
from types import ModuleType
from typing import Any, NoReturn

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "vllm_omni.engine.persistent_state_admission"
_INTERVALS = (80, 160, 320, 560, 1120)


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _module(spec_id: str = "PORT-STATE-027") -> ModuleType:
    try:
        return importlib.import_module(_MODULE)
    except ModuleNotFoundError:
        _fail(f"{spec_id} missing {_MODULE}")


def _symbol(name: str, spec_id: str = "PORT-STATE-027") -> Any:
    try:
        return getattr(_module(spec_id), name)
    except AttributeError:
        _fail(f"{spec_id} missing {_MODULE}.{name}")


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns


def _config(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "waiter_capacity": 8,
        "max_inflight_reserves": 2,
        "dispatch_budget": 2,
        "aging_threshold_ns": 10,
        "admission_wait_timeout_s": 1.0,
        "retry_floor_ms": 25,
        "retry_jitter_ms": 5,
        "recovery_backoff_s": (0.01, 0.02),
        "release_convergence_timeout_s": 1.0,
        "supported_intervals_ms": _INTERVALS,
    }
    values.update(overrides)
    return _symbol("AdmissionControllerConfig")(**values)


def _controller(clock: _Clock, **overrides: Any) -> Any:
    return _symbol("BoundedAdmissionController")(
        config=_config(**overrides),
        engine_epoch="epoch-a",
        monotonic_ns=clock,
        jitter_secret=b"phase5-deterministic",
    )


def _attempt(
    n: int,
    *,
    interval_ms: int = 560,
    connection_id: str | None = None,
    admission_deadline_ns: int = 1_000,
    unadmitted_deadline_ns: int = 2_000,
) -> Any:
    return _symbol("AdmissionAttempt")(
        attempt_id=f"attempt-{n}",
        connection_id=connection_id or f"connection-{n}",
        connection_handle=f"connection-handle-{n}",
        operation_id=f"operation-{n}",
        service_interval_ms=interval_ms,
        admission_deadline_ns=admission_deadline_ns,
        unadmitted_deadline_ns=unadmitted_deadline_ns,
    )


def _capacity(
    *,
    hard: int = 8,
    nominal: dict[int, int] | None = None,
    authority: str = "OPEN",
) -> Any:
    if nominal is None:
        nominal = dict.fromkeys(_INTERVALS, 8)
    return _symbol("AdmissionCapacity")(
        hard_headroom=hard,
        nominal_headroom_by_interval=nominal,
        authority=authority,
    )


def _ids(dispatches: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(item.attempt.attempt_id for item in dispatches)


def test_waiter_entry_is_fixed_shape_and_carries_no_audio_or_session() -> None:
    """@spec PORT-STATE-027: K bounds fixed metadata, never payload state."""

    field_names = {field.name for field in dataclasses.fields(_attempt(1))}

    assert {
        "attempt_id",
        "connection_id",
        "connection_handle",
        "operation_id",
        "service_interval_ms",
        "admission_deadline_ns",
        "unadmitted_deadline_ns",
    } <= field_names
    assert not field_names.intersection(
        {"audio", "audio_bytes", "decoded_audio", "session", "error_text"}
    )


def test_all_arrivals_enter_the_queue_and_duplicate_connection_coalesces() -> None:
    """@spec PORT-STATE-027 / PORT-STATE-028: no fast-path bypass."""

    clock = _Clock()
    controller = _controller(clock)
    first = controller.enqueue(
        _attempt(
            1,
            connection_id="connection-a",
            admission_deadline_ns=10,
            unadmitted_deadline_ns=20,
        )
    )
    duplicate = controller.enqueue(
        _attempt(
            2,
            connection_id="connection-a",
            admission_deadline_ns=100,
            unadmitted_deadline_ns=200,
        )
    )

    assert duplicate == first
    assert controller.snapshot.waiter_count == 1
    assert controller.snapshot.free_entries == 7
    clock.now_ns = 11
    dispositions = controller.expire_due()
    assert [(item.attempt_id, item.outcome) for item in dispositions] == [
        ("attempt-1", "shed")
    ]


def test_full_waiter_slab_is_typed_shed_and_never_dispatches_directly() -> None:
    """@spec PORT-STATE-024 / PORT-STATE-027: bounded burst waiting."""

    clock = _Clock()
    controller = _controller(
        clock,
        waiter_capacity=1,
        max_inflight_reserves=1,
        dispatch_budget=1,
    )
    controller.enqueue(_attempt(1))
    queue_full = _symbol("AdmissionQueueFull", "PORT-STATE-024")

    with pytest.raises(queue_full) as info:
        controller.enqueue(_attempt(2))

    from vllm_omni.engine.persistent_state_service import (
        PersistentStateBackpressure,
    )

    assert isinstance(info.value, PersistentStateBackpressure)
    assert info.value.retryable is True
    assert info.value.reason == "capacity"
    assert info.value.cause == "controller_full"
    assert 25 <= info.value.retry_after_ms <= 30
    assert controller.snapshot.waiter_count == 1


def test_reused_slab_slot_fences_stale_handle_and_deadline() -> None:
    """@spec PORT-STATE-027 / PORT-STATE-028: indexed timers never ghost."""

    clock = _Clock()
    controller = _controller(
        clock,
        waiter_capacity=1,
        max_inflight_reserves=1,
        dispatch_budget=1,
    )
    stale = controller.enqueue(
        _attempt(1, admission_deadline_ns=5, unadmitted_deadline_ns=50)
    )
    controller.cancel(stale)
    current = controller.enqueue(
        _attempt(2, admission_deadline_ns=20, unadmitted_deadline_ns=50)
    )

    assert current != stale
    with pytest.raises(_symbol("StaleAdmissionHandle")):
        controller.cancel(stale)
    clock.now_ns = 6
    assert controller.expire_due() == ()
    assert controller.snapshot.waiter_count == 1
    clock.now_ns = 21
    dispositions = controller.expire_due()
    assert [(item.attempt_id, item.outcome) for item in dispositions] == [
        ("attempt-2", "shed")
    ]


def test_one_drain_turn_is_bounded_by_five_heads_and_dispatch_budget() -> None:
    """@spec PORT-STATE-027 / PORT-PERF-008: fixed work per reactor turn."""

    clock = _Clock()
    controller = _controller(
        clock,
        waiter_capacity=100,
        max_inflight_reserves=100,
        dispatch_budget=2,
    )
    for n in range(75):
        controller.enqueue(_attempt(n, interval_ms=_INTERVALS[n % 5]))

    dispatches = controller.drain(_capacity(hard=100))

    assert len(dispatches) == 2
    assert controller.snapshot.last_drain_examined_heads <= 5
    assert controller.snapshot.last_drain_launched == 2
    assert controller.snapshot.waiter_count == 75
    assert controller.snapshot.submitted_count == 2


def test_aging_protects_the_old_expensive_head_release_window() -> None:
    """@spec PORT-STATE-027: later cheap heads cannot steal every release."""

    clock = _Clock()
    controller = _controller(clock, max_inflight_reserves=1, dispatch_budget=1)
    expensive = controller.enqueue(_attempt(1, interval_ms=80))
    cheap = controller.enqueue(_attempt(2, interval_ms=1120))
    cheap_only = _capacity(
        hard=1,
        nominal={80: 0, 160: 0, 320: 0, 560: 0, 1120: 1},
    )

    assert _ids(controller.drain(cheap_only)) == ("attempt-2",)
    controller.complete_no_lease(cheap)
    clock.now_ns = 11
    controller.enqueue(_attempt(3, interval_ms=1120))

    assert controller.drain(cheap_only) == ()
    assert controller.snapshot.protected_attempt_id == "attempt-1"
    both_fit = _capacity(hard=1, nominal=dict.fromkeys(_INTERVALS, 1))
    assert _ids(controller.drain(both_fit)) == ("attempt-1",)
    assert controller.snapshot.protected_attempt_id is None
    assert expensive != cheap


@pytest.mark.parametrize("terminal", ["cancel", "expire"])
def test_cancel_or_expiry_releases_the_protected_head(
    terminal: str,
) -> None:
    """@spec PORT-STATE-027 / PORT-STATE-028: protection is bounded."""

    clock = _Clock()
    controller = _controller(clock, max_inflight_reserves=1, dispatch_budget=1)
    protected = controller.enqueue(
        _attempt(1, interval_ms=80, admission_deadline_ns=20)
    )
    controller.enqueue(_attempt(2, interval_ms=1120))
    clock.now_ns = 11
    cheap_only = _capacity(
        hard=1,
        nominal={80: 0, 160: 0, 320: 0, 560: 0, 1120: 1},
    )
    assert controller.drain(cheap_only) == ()
    assert controller.snapshot.protected_attempt_id == "attempt-1"

    if terminal == "cancel":
        disposition = controller.cancel(protected)
        assert disposition.outcome == "cancelled"
    else:
        clock.now_ns = 21
        dispositions = controller.expire_due()
        assert [(d.attempt_id, d.outcome) for d in dispositions] == [
            ("attempt-1", "shed")
        ]

    assert controller.snapshot.protected_attempt_id is None
    assert _ids(controller.drain(cheap_only)) == ("attempt-2",)


def test_authority_loss_stops_dispatch_and_same_epoch_recovery_resumes() -> None:
    """@spec PORT-STATE-022 / PORT-STATE-028: same-epoch waiters survive."""

    clock = _Clock()
    controller = _controller(clock)
    controller.enqueue(_attempt(1))

    controller.authority_lost(engine_epoch="epoch-a")
    assert controller.drain(_capacity()) == ()
    assert controller.snapshot.authority == "UNHANDSHAKED"
    assert controller.snapshot.waiter_count == 1

    controller.authority_recovered(engine_epoch="epoch-a")
    assert controller.snapshot.authority == "OPEN"
    assert _ids(controller.drain(_capacity())) == ("attempt-1",)


def test_waiter_expiring_without_authority_is_unavailable_not_shed() -> None:
    """@spec PORT-STATE-022 / PORT-STATE-024 / PORT-STATE-028."""

    clock = _Clock()
    controller = _controller(clock)
    controller.enqueue(
        _attempt(1, admission_deadline_ns=10, unadmitted_deadline_ns=100)
    )
    controller.authority_lost(engine_epoch="epoch-a")
    clock.now_ns = 11

    dispositions = controller.expire_due()

    assert [(item.attempt_id, item.outcome) for item in dispositions] == [
        ("attempt-1", "unavailable")
    ]


def test_total_connection_deadline_is_terminal_even_before_wait_deadline() -> None:
    """@spec PORT-STATE-024 / PORT-SESS-012."""

    clock = _Clock()
    controller = _controller(clock)
    controller.enqueue(
        _attempt(1, admission_deadline_ns=100, unadmitted_deadline_ns=10)
    )
    clock.now_ns = 11

    dispositions = controller.expire_due()

    assert len(dispositions) == 1
    assert dispositions[0].attempt_id == "attempt-1"
    assert dispositions[0].outcome == "shed"
    assert dispositions[0].terminal_connection is True
    assert dispositions[0].close_code == 1013


def test_epoch_change_flushes_old_waiters_and_submissions_as_unavailable() -> None:
    """@spec PORT-STATE-016 / PORT-STATE-022 / PORT-STATE-028."""

    clock = _Clock()
    controller = _controller(clock, dispatch_budget=1)
    controller.enqueue(_attempt(1))
    controller.enqueue(_attempt(2))
    assert _ids(controller.drain(_capacity())) == ("attempt-1",)

    dispositions = controller.engine_epoch_changed("epoch-b")

    assert sorted((item.attempt_id, item.outcome) for item in dispositions) == [
        ("attempt-1", "unavailable"),
        ("attempt-2", "unavailable"),
    ]
    assert controller.snapshot.waiter_count == 0
    assert controller.snapshot.submitted_count == 0
    assert controller.snapshot.authority == "UNHANDSHAKED"


@pytest.mark.parametrize("attached", [True, False])
def test_admission_or_cleanup_handoff_precedes_waiter_slot_reuse(
    attached: bool,
) -> None:
    """@spec PORT-STATE-013 / PORT-STATE-028: no forgotten lease window."""

    clock = _Clock()
    controller = _controller(
        clock,
        waiter_capacity=1,
        max_inflight_reserves=1,
        dispatch_budget=1,
    )
    handle = controller.enqueue(_attempt(1))
    controller.drain(_capacity())[0]
    lease = object()
    handoff_calls = 0
    cleanup_inventory: list[object] = []

    def handoff(_attempt: Any, lease: object) -> None:
        nonlocal handoff_calls
        handoff_calls += 1
        assert lease is committed_lease
        assert controller.snapshot.waiter_count == 1
        assert controller.snapshot.free_entries == 0
        if not attached:
            cleanup_inventory.append(lease)

    committed_lease = lease
    controller.complete_admitted(
        handle,
        lease=committed_lease,
        attached=attached,
        handoff=handoff,
    )

    assert handoff_calls == 1
    assert controller.snapshot.waiter_count == 0
    assert controller.snapshot.free_entries == 1
    assert cleanup_inventory == ([] if attached else [committed_lease])
    with pytest.raises(_symbol("StaleAdmissionHandle")):
        controller.complete_admitted(
            handle,
            lease=committed_lease,
            attached=attached,
            handoff=handoff,
        )
    assert handoff_calls == 1
