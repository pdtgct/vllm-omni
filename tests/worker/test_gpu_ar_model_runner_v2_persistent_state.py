# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the model-neutral Omni MRv2 state bridge."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as CoreGPUWorker

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


def _worker_cls() -> type[Any]:
    from vllm_omni.worker.gpu_ar_worker import GPUARWorker

    if "compile_or_warm_up_model" not in GPUARWorker.__dict__:
        pytest.fail(
            "PORT-ADV-003 missing persistent-only worker warmup",
            pytrace=False,
        )
    return GPUARWorker


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


def test_persistent_only_block_tables_cross_the_real_core_staged_write_step() -> None:
    # @spec PORT-MIG-005 / PORT-STATE-002
    runner_cls = _runner_cls()
    if "_install_persistent_only_block_tables" not in runner_cls.__dict__:
        pytest.fail(
            "PORT-MIG-005 missing persistent-only block-table adapter",
            pytrace=False,
        )

    class RawEmptyBlockTables:
        num_kv_cache_groups = 0
        fused_writer = None

        def apply_staged_writes(self) -> None:
            BlockTables.apply_staged_writes(self)  # type: ignore[arg-type]

    state = _ProjectionState()
    runner = _runner(state)
    runner.block_tables = RawEmptyBlockTables()
    runner.update_pp_decode_requests = lambda: None
    runner.finish_requests = lambda output: None
    runner.free_states = lambda output: None
    runner.add_requests = lambda output: None
    runner.update_requests = lambda output: None
    no_forward = object()
    runner.kv_connector = SimpleNamespace(
        no_forward=lambda output: no_forward,
    )
    output = _scheduler_output(total_tokens=0)

    runner_cls._install_persistent_only_block_tables(runner)
    actual = runner_cls.execute_model(runner, output)

    assert actual is no_forward
    assert runner.block_tables.num_kv_cache_groups == 0


def test_persistent_only_block_tables_expose_empty_attention_views() -> None:
    # @spec PORT-MIG-005 / PORT-STATE-002
    runner_cls = _runner_cls()

    class RawEmptyBlockTables:
        num_kv_cache_groups = 0
        slot_mappings = torch.empty((0, 16), dtype=torch.int64)

    runner = _runner()
    runner.block_tables = RawEmptyBlockTables()
    runner_cls._install_persistent_only_block_tables(runner)

    runner.block_tables.apply_staged_writes()
    runner.block_tables.append_block_ids(0, (), overwrite=True)
    assert runner.block_tables.gather_block_tables(object(), 4) == ()
    assert runner.block_tables.get_dummy_block_tables(4) == ()
    slot_mappings = runner.block_tables.compute_slot_mappings(
        object(), object(), object(), 7
    )
    assert slot_mappings.shape == (0, 7)
    assert runner.block_tables.get_dummy_slot_mappings(5).shape == (0, 5)
    with pytest.raises(RuntimeError, match="ordinary block ids"):
        runner.block_tables.append_block_ids(0, ([1],), overwrite=True)


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


@pytest.mark.parametrize("async_wrapped", [False, True])
def test_sample_boundary_consumes_and_forwards_model_transaction_status(
    monkeypatch: pytest.MonkeyPatch,
    async_wrapped: bool,
) -> None:
    # @spec PORT-ADV-004 / PORT-HOOK-001 / PORT-MIG-005
    events: list[str] = []

    class StatusModel:
        def collect_commit_status(self) -> tuple[dict[str, int], set[str]]:
            events.append("collect")
            return {"healthy": 0, "failed": 512}, {"failed"}

    runner = _runner()
    runner.model = StatusModel()
    model_runner_output = SimpleNamespace()
    core_output = (
        SimpleNamespace(model_runner_output=model_runner_output)
        if async_wrapped
        else model_runner_output
    )

    def sample(self: Any, grammar_output: object) -> object:
        del self, grammar_output
        events.append("sample")
        return core_output

    monkeypatch.setattr(GPUModelRunner, "sample_tokens", sample)

    actual = type(runner).sample_tokens(runner, grammar_output=None)

    assert actual is core_output
    assert events == ["sample", "collect"]
    connector = model_runner_output.omni_connector_output
    assert connector.model_status == {"healthy": 0, "failed": 512}
    assert connector.model_failed_req_ids == {"failed"}


def test_sample_boundary_consumes_clean_status_before_the_next_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-004 / PORT-HOOK-001 / PORT-MIG-005
    staged = True

    class StatusModel:
        def collect_commit_status(self) -> tuple[dict[str, int], set[str]]:
            nonlocal staged
            assert staged
            staged = False
            return {"request": 0}, set()

    runner = _runner()
    runner.model = StatusModel()
    model_runner_output = SimpleNamespace()
    monkeypatch.setattr(
        GPUModelRunner,
        "sample_tokens",
        lambda self, grammar_output: model_runner_output,
    )

    type(runner).sample_tokens(runner, grammar_output=None)

    assert not staged
    assert model_runner_output.omni_connector_output.model_status == {
        "request": 0
    }


@pytest.mark.parametrize("async_wrapped", [False, True])
def test_sample_boundary_drains_streaming_batch_stats_under_mrv2(
    monkeypatch: pytest.MonkeyPatch,
    async_wrapped: bool,
) -> None:
    # @spec PORT-OBS-008, PORT-OBS-009
    class BatchStatsModel:
        def __init__(self) -> None:
            self.consume_calls = 0

        def consume_batch_stats(self) -> list[tuple[str, int]]:
            self.consume_calls += 1
            return [("1120", 4)]

        def collect_commit_status(self) -> tuple[dict[str, int], set[str]]:
            return {}, set()

    runner = _runner()
    model = BatchStatsModel()
    runner.model = model
    model_runner_output = SimpleNamespace()
    core_output = (
        SimpleNamespace(model_runner_output=model_runner_output)
        if async_wrapped
        else model_runner_output
    )
    monkeypatch.setattr(
        GPUModelRunner,
        "sample_tokens",
        lambda self, grammar_output: core_output,
    )

    actual = type(runner).sample_tokens(runner, grammar_output=None)

    assert actual is core_output
    assert model.consume_calls == 1
    assert model_runner_output.streaming_chunk_batch_stats == [("1120", 4)]


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


def _worker(*, persistent: bool, ordinary_groups: int) -> Any:
    worker = object.__new__(_worker_cls())
    worker.model_config = SimpleNamespace(enforce_eager=True)
    worker.model_runner = SimpleNamespace(
        _persistent_state_storage=object() if persistent else None,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[object()] * ordinary_groups,
        ),
        model=SimpleNamespace(),
    )
    return worker


def test_persistent_only_worker_skips_core_text_cache_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-003 / PORT-MIG-005
    worker_cls = _worker_cls()
    worker = _worker(persistent=True, ordinary_groups=0)
    events: list[str] = []
    marker = object()
    monkeypatch.setattr(
        CoreGPUWorker,
        "compile_or_warm_up_model",
        lambda self: pytest.fail(
            "PORT-ADV-003 persistent-only warmup entered core text warmup",
            pytrace=False,
        ),
    )
    monkeypatch.setattr(
        worker_cls,
        "_compile_or_warm_up_persistent_only_model",
        lambda self: events.append("persistent-only") or marker,
    )
    actual = worker_cls.compile_or_warm_up_model(worker)

    assert actual is marker
    assert events == ["persistent-only"]


def test_persistent_only_worker_logs_actual_execution_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005 / PORT-MIG-006 / PORT-STATE-002
    module = importlib.import_module("vllm_omni.worker.gpu_ar_worker")
    worker_cls = _worker_cls()
    worker = _worker(persistent=True, ordinary_groups=0)

    class LoadedModel:
        is_hybrid = False

    class LoadedModelState:
        pass

    class LoadedPersistentStateSpec:
        pass

    class LoadedPersistentStateStorage:
        spec = LoadedPersistentStateSpec()

    worker.model_runner.model = LoadedModel()
    worker.model_runner.model_state = LoadedModelState()
    worker.model_runner._persistent_state_storage = (
        LoadedPersistentStateStorage()
    )
    worker.model_config = SimpleNamespace(enforce_eager=True)
    marker = object()
    monkeypatch.setattr(
        worker_cls,
        "_compile_or_warm_up_persistent_only_model",
        lambda self: marker,
    )
    logged: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        module.logger,
        "info",
        lambda message, *args: logged.append((message, *args)),
    )

    assert worker_cls.compile_or_warm_up_model(worker) is marker
    assert logged == [
        (
            "Persistent-state execution fingerprint: worker=%s "
            "runner=%s model=%s model_state=%s state_spec=%s "
            "is_hybrid=%s eager=%s",
            "GPUARWorker",
            "SimpleNamespace",
            "LoadedModel",
            "LoadedModelState",
            "LoadedPersistentStateSpec",
            False,
            True,
        )
    ]


def test_nonpersistent_worker_preserves_core_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-003 / PORT-MIG-006
    worker_cls = _worker_cls()
    worker = _worker(persistent=False, ordinary_groups=1)
    events: list[str] = []
    marker = object()
    monkeypatch.setattr(
        CoreGPUWorker,
        "compile_or_warm_up_model",
        lambda self: events.append("core") or marker,
    )

    actual = worker_cls.compile_or_warm_up_model(worker)

    assert actual is marker
    assert events == ["core"]


def test_mixed_persistent_and_token_cache_warmup_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-003 / PORT-MIG-006
    worker_cls = _worker_cls()
    worker = _worker(persistent=True, ordinary_groups=1)
    monkeypatch.setattr(
        CoreGPUWorker,
        "compile_or_warm_up_model",
        lambda self: pytest.fail(
            "PORT-ADV-003 mixed persistent cache silently entered core warmup",
            pytrace=False,
        ),
    )

    with pytest.raises(RuntimeError, match="mixed.*not qualified"):
        worker_cls.compile_or_warm_up_model(worker)


def test_persistent_only_warmup_preserves_the_core_operational_postamble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-003 / PORT-MIG-005 / PORT-MIG-006 / PORT-PERF-009
    from vllm.config.compilation import CompilationMode

    module = importlib.import_module("vllm_omni.worker.gpu_ar_worker")
    worker_cls = _worker_cls()
    if "_compile_or_warm_up_persistent_only_model" not in worker_cls.__dict__:
        pytest.fail(
            "PORT-ADV-003 missing persistent-only worker warmup lifecycle",
            pytrace=False,
        )
    worker = _worker(persistent=True, ordinary_groups=0)
    events: list[str] = []
    worker.model_runner.maybe_remove_all_loras = (
        lambda config: events.append("remove-loras")
    )
    worker.model_runner.lora_config = None
    worker.model_runner.model.warmup_resident_state = lambda: events.append(
        f"resident-state:inference={torch.is_inference_mode_enabled()}"
    )
    worker.model_config = SimpleNamespace(enforce_eager=True, seed=17)
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            backend="",
            compilation_time=1.5,
            encoder_compilation_time=2.5,
        ),
        observability_config=SimpleNamespace(
            jit_monitor_mode="off",
            jit_monitor_verbose=False,
        ),
    )
    worker.compilation_config = worker.vllm_config.compilation_config
    worker.observability_config = worker.vllm_config.observability_config
    monkeypatch.setattr(
        module,
        "kernel_warmup",
        lambda value: events.append(
            f"kernel-warmup:inference={torch.is_inference_mode_enabled()}"
        ),
    )
    monkeypatch.setattr(
        module,
        "set_random_seed",
        lambda seed: events.append(f"seed:{seed}"),
    )
    monkeypatch.setattr(
        module,
        "freeze_gc_heap",
        lambda: events.append("freeze-gc"),
    )
    monkeypatch.setattr(
        module,
        "maybe_attach_gc_debug_callback",
        lambda: events.append("gc-debug"),
    )
    monkeypatch.setattr(
        module,
        "enable_gpu_sync_check",
        lambda: events.append("gpu-sync-check"),
    )
    jit_monitor = importlib.import_module("vllm.utils.jit_monitor")
    monkeypatch.setattr(
        jit_monitor,
        "activate",
        lambda **kwargs: events.append("jit-monitor"),
    )

    warmup_attestation = getattr(
        worker_cls,
        "persistent_state_warmup_attestation",
        None,
    )
    if not callable(warmup_attestation):
        pytest.fail(
            "PORT-ADV-003 missing worker warmup attestation",
            pytrace=False,
        )
    assert warmup_attestation(worker) is False
    result = worker_cls._compile_or_warm_up_persistent_only_model(worker)

    assert result.language_model == 1.5
    assert result.encoder == 2.5
    assert warmup_attestation(worker) is True
    assert events == [
        "remove-loras",
        "kernel-warmup:inference=True",
        "resident-state:inference=True",
        "seed:17",
        "jit-monitor",
        "freeze-gc",
        "gc-debug",
        "gpu-sync-check",
    ]


def test_failed_resident_warmup_never_attests_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-ADV-003 / ENV-MIG-012."""
    from vllm.config.compilation import CompilationMode

    module = importlib.import_module("vllm_omni.worker.gpu_ar_worker")
    worker_cls = _worker_cls()
    worker = _worker(persistent=True, ordinary_groups=0)
    worker.model_runner.maybe_remove_all_loras = lambda config: None
    worker.model_runner.lora_config = None
    worker.model_runner.model.warmup_resident_state = lambda: (_ for _ in ()).throw(
        RuntimeError("scatter warmup failed")
    )
    worker.model_config = SimpleNamespace(enforce_eager=True, seed=17)
    worker.compilation_config = SimpleNamespace(
        mode=CompilationMode.NONE,
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker.observability_config = SimpleNamespace(
        jit_monitor_mode="off",
        jit_monitor_verbose=False,
    )
    monkeypatch.setattr(module, "kernel_warmup", lambda value: None)

    with pytest.raises(RuntimeError, match="scatter warmup failed"):
        worker_cls._compile_or_warm_up_persistent_only_model(worker)

    warmup_attestation = getattr(
        worker_cls,
        "persistent_state_warmup_attestation",
        None,
    )
    if not callable(warmup_attestation):
        pytest.fail(
            "PORT-ADV-003 missing worker warmup attestation",
            pytrace=False,
        )
    assert warmup_attestation(worker) is False


def test_core_warmup_pin_guard_keeps_the_persistent_only_divergence_narrow() -> None:
    # @spec PORT-MIG-006
    source = inspect.getsource(CoreGPUWorker.compile_or_warm_up_model)
    ordered = (
        "self.model_runner.maybe_remove_all_loras",
        "kernel_warmup(self)",
        "self.model_runner.capture_model()",
        "warmup_kernels(self.model_runner",
        "set_random_seed(self.model_config.seed)",
        "activate_jit_monitor(",
        "freeze_gc_heap()",
        "enable_gpu_sync_check()",
        "return CompilationTimes(",
    )
    offsets = [source.index(fragment) for fragment in ordered]
    assert offsets == sorted(offsets)
