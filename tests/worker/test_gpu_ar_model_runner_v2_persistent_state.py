# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the model-neutral Omni MRv2 state bridge."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace
from typing import Any, cast

import pytest
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runner_cls() -> type[Any]:
    try:
        module = importlib.import_module("vllm_omni.worker.gpu_ar_model_runner_v2")
    except ModuleNotFoundError:
        pytest.fail(
            "PORT-MIG-005 missing GPUARModelRunnerV2 module",
            pytrace=False,
        )
    return cast(type[Any], module.GPUARModelRunnerV2)


class _ProjectionState:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def begin_omni_projection(
        self,
        scheduler_output: object,
        *,
        dummy_run: bool,
        is_profile: bool,
    ) -> None:
        self.events.append(("begin", (scheduler_output, dummy_run, is_profile)))

    def end_omni_projection(self) -> None:
        self.events.append(("end", None))

    def reconcile_omni_lifecycle(
        self,
        *,
        finished_req_ids: frozenset[str],
        preempted_req_ids: frozenset[str],
        resident_req_ids: frozenset[str],
    ) -> None:
        self.events.append(
            (
                "reconcile",
                (finished_req_ids, preempted_req_ids, resident_req_ids),
            )
        )


def _runner(state: object | None = None) -> Any:
    runner = object.__new__(_runner_cls())
    runner.model_state = state if state is not None else object()
    runner.req_states = SimpleNamespace(req_id_to_index={})
    return runner


def _scheduler_output(
    *,
    finished: set[str] | None = None,
    preempted: set[str] | None = None,
    total_tokens: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        finished_req_ids=frozenset(finished or set()),
        preempted_req_ids=frozenset(preempted or set()),
        total_num_scheduled_tokens=total_tokens,
    )


def test_core_call_order_guard_matches_the_selected_v025_pin() -> None:
    # @spec PORT-MIG-005 / PORT-MIG-006
    source = inspect.getsource(GPUModelRunner.execute_model)
    ordered = (
        "self.finish_requests(scheduler_output)",
        "self.free_states(scheduler_output)",
        "self.add_requests(scheduler_output)",
        "self.update_requests(scheduler_output)",
        "self.block_tables.apply_staged_writes()",
        "if scheduler_output.total_num_scheduled_tokens == 0:",
    )
    offsets = [source.index(fragment) for fragment in ordered]
    assert offsets == sorted(offsets)


def test_bridge_is_a_direct_thin_core_runner_subclass() -> None:
    # @spec PORT-MIG-005 / PORT-MIG-006
    runner_cls = _runner_cls()
    source = inspect.getsource(runner_cls.execute_model)

    assert issubclass(runner_cls, GPUModelRunner)
    assert "super().execute_model" in source
    assert "prepare_inputs(" not in source
    assert "prepare_attn(" not in source
    assert "sample_tokens(" not in source


def test_finish_requests_keeps_finished_and_preempted_authority_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-STATE-005 / PORT-MIG-005
    calls: list[str] = []
    monkeypatch.setattr(
        GPUModelRunner,
        "finish_requests",
        lambda self, output: calls.append("super-finish"),
    )
    runner = _runner()
    output = _scheduler_output(finished={"finished"}, preempted={"preempted"})

    type(runner).finish_requests(runner, output)

    assert calls == ["super-finish"]
    assert runner._omni_finished_req_ids == frozenset({"finished"})
    assert runner._omni_preempted_req_ids == frozenset({"preempted"})


def test_update_requests_reconciles_after_super_with_complete_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005
    state = _ProjectionState()
    runner = _runner(state)
    runner._omni_finished_req_ids = frozenset({"finished"})
    runner._omni_preempted_req_ids = frozenset({"preempted"})

    def update(self: Any, output: object) -> None:
        state.events.append(("super-update", None))
        self.req_states.req_id_to_index = {"resident": 0}

    monkeypatch.setattr(GPUModelRunner, "update_requests", update)

    type(runner).update_requests(runner, _scheduler_output())

    assert state.events == [
        ("super-update", None),
        (
            "reconcile",
            (
                frozenset({"finished"}),
                frozenset({"preempted"}),
                frozenset({"resident"}),
            ),
        ),
    ]


def test_finish_only_zero_token_update_reconciles_in_the_same_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005
    state = _ProjectionState()
    runner = _runner(state)
    monkeypatch.setattr(GPUModelRunner, "finish_requests", lambda self, output: None)
    monkeypatch.setattr(GPUModelRunner, "update_requests", lambda self, output: None)
    output = _scheduler_output(finished={"last"}, total_tokens=0)

    type(runner).finish_requests(runner, output)
    type(runner).update_requests(runner, output)

    assert state.events[-1] == (
        "reconcile",
        (frozenset({"last"}), frozenset(), frozenset()),
    )


def test_execute_model_brackets_super_with_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005
    state = _ProjectionState()
    runner = _runner(state)
    output = _scheduler_output()
    result = object()

    def execute(self: Any, value: object, **kwargs: object) -> object:
        state.events.append(("super-execute", None))
        return result

    monkeypatch.setattr(GPUModelRunner, "execute_model", execute)

    actual = type(runner).execute_model(
        runner,
        output,
        dummy_run=False,
        is_profile=False,
    )

    assert actual is result
    assert state.events == [
        ("begin", (output, False, False)),
        ("super-execute", None),
        ("end", None),
    ]


def test_execute_model_ends_projection_and_clears_authority_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005
    state = _ProjectionState()
    runner = _runner(state)
    runner._omni_finished_req_ids = frozenset({"stale-finished"})
    runner._omni_preempted_req_ids = frozenset({"stale-preempted"})

    def execute(self: Any, output: object, **kwargs: object) -> object:
        raise RuntimeError("super failed")

    monkeypatch.setattr(GPUModelRunner, "execute_model", execute)

    with pytest.raises(RuntimeError, match="super failed"):
        type(runner).execute_model(runner, _scheduler_output())

    assert state.events[-1] == ("end", None)
    assert runner._omni_finished_req_ids == frozenset()
    assert runner._omni_preempted_req_ids == frozenset()


def test_dummy_profile_projection_is_marked_but_not_reconciled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005
    state = _ProjectionState()
    runner = _runner(state)
    output = _scheduler_output()
    monkeypatch.setattr(
        GPUModelRunner,
        "execute_model",
        lambda self, value, **kwargs: None,
    )

    type(runner).execute_model(
        runner,
        output,
        dummy_run=True,
        is_profile=True,
    )

    assert state.events == [
        ("begin", (output, True, True)),
        ("end", None),
    ]


def test_profile_flag_without_dummy_is_rejected_before_core_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005
    state = _ProjectionState()
    runner = _runner(state)
    core_calls: list[str] = []
    monkeypatch.setattr(
        GPUModelRunner,
        "execute_model",
        lambda self, value, **kwargs: core_calls.append("execute"),
    )

    with pytest.raises(ValueError, match="profile.*dummy|dummy.*profile"):
        type(runner).execute_model(
            runner,
            _scheduler_output(),
            dummy_run=False,
            is_profile=True,
        )

    assert core_calls == []
    assert state.events == []


def test_failed_dummy_is_fully_closed_before_the_next_real_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005
    state = _ProjectionState()
    runner = _runner(state)
    output = _scheduler_output()
    attempts = 0

    def execute(self: Any, value: object, **kwargs: object) -> object:
        nonlocal attempts
        del self, value, kwargs
        attempts += 1
        if attempts == 1:
            raise RuntimeError("profile failed")
        state.events.append(("real-execute", None))
        return "real-result"

    monkeypatch.setattr(GPUModelRunner, "execute_model", execute)

    with pytest.raises(RuntimeError, match="profile failed"):
        type(runner).execute_model(
            runner,
            output,
            dummy_run=True,
            is_profile=True,
        )
    actual = type(runner).execute_model(
        runner,
        output,
        dummy_run=False,
        is_profile=False,
    )

    assert actual == "real-result"
    assert state.events == [
        ("begin", (output, True, True)),
        ("end", None),
        ("begin", (output, False, False)),
        ("real-execute", None),
        ("end", None),
    ]


@pytest.mark.parametrize("enabled", [True, False])
def test_v2_prefix_cache_guard_runs_before_initial_profile(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    # @spec PORT-INT-007 / PORT-STATE-002
    runner_cls = _runner_cls()
    if "profile_run" not in runner_cls.__dict__:
        pytest.fail(
            "PORT-INT-007 missing V2 pre-profile persistent-state guard",
            pytrace=False,
        )
    runner = object.__new__(runner_cls)
    runner.vllm_config = object()
    runner.cache_config = SimpleNamespace(enable_prefix_caching=enabled)
    profile_calls: list[str] = []
    module = importlib.import_module("vllm_omni.worker.gpu_ar_model_runner_v2")
    monkeypatch.setattr(
        module,
        "discover_persistent_state_specs",
        lambda config: {"persistent": object()},
    )
    monkeypatch.setattr(
        GPUModelRunner,
        "profile_run",
        lambda self: profile_calls.append("profile"),
    )

    if enabled:
        with pytest.raises(ValueError, match="prefix caching"):
            runner_cls.profile_run(runner)
        assert profile_calls == []
    else:
        runner_cls.profile_run(runner)
        assert profile_calls == ["profile"]


def test_v2_cache_discovery_and_initialization_are_thin_pin_guarded_overrides() -> None:
    # @spec PORT-STATE-002 / PORT-MIG-006
    runner_cls = _runner_cls()
    discovery = inspect.getsource(runner_cls.get_kv_cache_spec)
    initialization = inspect.getsource(runner_cls.initialize_kv_cache)

    assert "super().get_kv_cache_spec" in discovery
    assert "PersistentStateSpec" in discovery
    assert "super().initialize_kv_cache" in initialization
    assert "PersistentStateSpec" in initialization
    assert "MambaSpec" not in discovery + initialization


def test_worker_selects_v2_without_mutating_the_requested_runner() -> None:
    # @spec PORT-MIG-005 / PORT-MIG-006
    from vllm_omni.worker.gpu_ar_worker import GPUARWorker

    source = inspect.getsource(GPUARWorker.init_device)
    assert "self.use_v2_model_runner = False" not in source
    assert "GPUARModelRunnerV2" in source
