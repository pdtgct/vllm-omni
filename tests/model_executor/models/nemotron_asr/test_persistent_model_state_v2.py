# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for Nemotron's direct MRv2 ``ModelState``."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace
from typing import Any, cast

import pytest
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _module() -> Any:
    try:
        return importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.model_state_v2")
    except ModuleNotFoundError:
        pytest.fail(
            "PORT-MIG-004 missing direct NemotronASRModelState module",
            pytrace=False,
        )


def _model_class() -> type[Any]:
    try:
        module = importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.nemotron_asr")
    except ModuleNotFoundError:
        pytest.fail(
            "PORT-MIG-004 missing Nemotron ASR model module",
            pytrace=False,
        )
    return cast(type[Any], module.NemotronASRForRNNT)


def test_model_selects_a_direct_model_state_not_a_mamba_state() -> None:
    # @spec PORT-MIG-004 / PORT-MIG-006
    state_cls = _module().NemotronASRModelState

    assert issubclass(state_cls, ModelState)
    assert not issubclass(state_cls, MambaHybridModelState)
    assert _model_class().get_model_state_cls() is state_cls


def test_model_state_accounts_for_the_complete_v025_hook_surface() -> None:
    # @spec PORT-MIG-004
    state_cls = _module().NemotronASRModelState
    projected_hooks = {
        "add_request",
        "remove_request",
        "get_mm_embeddings",
        "prepare_inputs",
        "prepare_dummy_inputs",
        "prepare_attn",
        "begin_omni_projection",
        "end_omni_projection",
        "reconcile_omni_lifecycle",
    }
    inherited_hooks = {
        "get_supported_generation_tasks",
        "apply_staged_writes",
        "dummy_inputs_embeds",
        "gather_mm_embeddings",
        "preprocess_state",
        "postprocess_state",
        "custom_sampler",
    }

    assert projected_hooks <= state_cls.__dict__.keys()
    assert inherited_hooks.isdisjoint(state_cls.__dict__)
    assert state_cls.num_new_sampled_tokens_per_step == 1


def test_core_task_policy_is_generate_plus_realtime_only() -> None:
    # @spec PORT-MIG-004
    state_cls = _module().NemotronASRModelState
    state = object.__new__(state_cls)
    state.model = object.__new__(_model_class())

    assert state.get_supported_generation_tasks() == ("generate", "realtime")
    assert "transcription" not in state.get_supported_generation_tasks()


def test_stock_one_token_sampler_is_unchanged() -> None:
    # @spec PORT-MIG-004
    state_cls = _module().NemotronASRModelState
    state = object.__new__(state_cls)

    assert state.num_new_sampled_tokens_per_step == 1
    assert state.custom_sampler(object()) is None


def test_cuda_graph_dummy_inputs_fail_closed() -> None:
    # @spec PORT-MIG-004 / PORT-MIG-005
    state_cls = _module().NemotronASRModelState
    state = object.__new__(state_cls)

    with pytest.raises(RuntimeError, match="CUDA graph|graph capture"):
        state.prepare_dummy_inputs(num_reqs=2, num_tokens=2)


def test_lifecycle_hooks_cannot_become_physical_state_authority() -> None:
    # @spec PORT-MIG-004 / PORT-STATE-014
    state_cls = _module().NemotronASRModelState
    forbidden = {
        "reserve",
        "release",
        "free_slot",
        "allocate_slot",
        "PersistentStateService",
        "PersistentStateManager",
    }

    for method_name in ("add_request", "remove_request"):
        source = inspect.getsource(getattr(state_cls, method_name))
        assert not any(fragment in source for fragment in forbidden)


def test_projection_join_rejects_incomplete_and_cross_epoch_views() -> None:
    # @spec PORT-MIG-005 / PORT-STATE-007
    join = _module().ProjectionJoin()
    first = join.begin(dummy_run=False, is_profile=False)
    join.record_mm(
        first,
        req_ids=("r0",),
        scheduled_encoder_inputs={"r0": [0]},
        encoder_features=object(),
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        join.complete(
            first,
            input_batch=SimpleNamespace(req_ids=["r0"]),
            req_states=object(),
            model_kwargs={},
        )

    join.end(first)
    second = join.begin(dummy_run=False, is_profile=False)
    with pytest.raises(RuntimeError, match="epoch"):
        join.record_bindings(
            first,
            req_ids=("r0",),
            bindings=(object(),),
            block_ids=((7,),),
        )
    join.end(second)


def test_projection_join_requires_one_consistent_request_order() -> None:
    # @spec PORT-MIG-005 / PORT-STATE-007
    join = _module().ProjectionJoin()
    epoch = join.begin(dummy_run=False, is_profile=False)
    join.record_mm(
        epoch,
        req_ids=("left", "right"),
        scheduled_encoder_inputs={"left": [0], "right": [0]},
        encoder_features=object(),
    )
    join.record_bindings(
        epoch,
        req_ids=("right", "left"),
        bindings=(object(), object()),
        block_ids=((8,), (9,)),
    )

    with pytest.raises(ValueError, match="request order"):
        join.complete(
            epoch,
            input_batch=SimpleNamespace(req_ids=["left", "right"]),
            req_states=object(),
            model_kwargs={},
        )

    join.end(epoch)


def test_projection_join_preserves_binding_and_runner_view_identity() -> None:
    # @spec PORT-MIG-005 / PORT-STATE-007
    join = _module().ProjectionJoin()
    epoch = join.begin(dummy_run=False, is_profile=False)
    scheduled = {"r0": [3]}
    features = object()
    binding = object()
    block_ids = ((11,),)
    input_batch = SimpleNamespace(req_ids=["r0"])
    req_states = object()
    kwargs = {"positions": object()}
    join.record_mm(
        epoch,
        req_ids=("r0",),
        scheduled_encoder_inputs=scheduled,
        encoder_features=features,
    )
    join.record_bindings(
        epoch,
        req_ids=("r0",),
        bindings=(binding,),
        block_ids=block_ids,
    )

    snapshot = join.complete(
        epoch,
        input_batch=input_batch,
        req_states=req_states,
        model_kwargs=kwargs,
    )

    assert snapshot.epoch == epoch
    assert snapshot.req_ids == ("r0",)
    assert snapshot.scheduled_encoder_inputs is scheduled
    assert snapshot.encoder_features is features
    assert snapshot.bindings == (binding,)
    assert snapshot.block_ids is block_ids
    assert snapshot.input_batch is input_batch
    assert snapshot.req_states is req_states
    assert snapshot.model_kwargs is kwargs
    join.end(epoch)


def test_projection_join_dummy_marker_is_distinct_nonresumable_and_no_io() -> None:
    # @spec PORT-MIG-005
    join = _module().ProjectionJoin()
    epoch = join.begin(dummy_run=True, is_profile=True)

    bypass = join.complete_dummy(
        epoch,
        input_batch=SimpleNamespace(req_ids=["dummy"]),
        req_states=object(),
        model_kwargs={},
    )

    assert bypass.dummy_run is True
    assert bypass.is_profile is True
    assert bypass.no_page_io is True
    assert bypass.bindings == ()
    join.end(epoch)
    with pytest.raises(RuntimeError, match="epoch"):
        join.complete_dummy(
            epoch,
            input_batch=SimpleNamespace(req_ids=["dummy"]),
            req_states=object(),
            model_kwargs={},
        )


def _projection_state() -> Any:
    state_cls = _module().NemotronASRModelState
    state = object.__new__(state_cls)
    state._projection = _module().ProjectionJoin()
    state._projection_epoch = None
    state._scheduler_output = None
    state._projected_binding_keys = set()
    state._initialized_binding_keys = set()
    state._request_metadata = SimpleNamespace(
        snapshot=lambda req_ids: pytest.fail(
            f"PORT-MIG-005 dummy read request metadata for {req_ids!r}",
            pytrace=False,
        )
    )
    return state


def test_prepare_inputs_completes_the_marked_dummy_without_real_views() -> None:
    # @spec PORT-MIG-005 / PORT-STATE-003 / PORT-STATE-007
    state = _projection_state()
    scheduler_output = SimpleNamespace(persistent_state_bindings={})
    input_batch = SimpleNamespace(req_ids=["arbitrary-client-visible-id"])
    req_states = object()
    state.begin_omni_projection(
        scheduler_output,
        dummy_run=True,
        is_profile=True,
    )

    prepared = state.prepare_inputs(input_batch, req_states)
    snapshot = prepared["persistent_state_projection"]

    assert snapshot.req_ids == ("arbitrary-client-visible-id",)
    assert snapshot.input_batch is input_batch
    assert snapshot.req_states is req_states
    assert snapshot.dummy_run is True
    assert snapshot.is_profile is True
    assert snapshot.no_page_io is True
    assert snapshot.bindings == ()
    assert snapshot.request_metadata is None
    state.end_omni_projection()


def test_profile_marker_without_dummy_fails_before_projection_state_exists() -> None:
    # @spec PORT-MIG-005
    state = _projection_state()

    with pytest.raises(ValueError, match="profile.*dummy|dummy.*profile"):
        state.begin_omni_projection(
            object(),
            dummy_run=False,
            is_profile=True,
        )

    assert state._projection_epoch is None
    assert state._scheduler_output is None


def test_dummy_selection_never_depends_on_the_request_id_spelling() -> None:
    # @spec PORT-MIG-005
    state = _projection_state()
    state.begin_omni_projection(
        SimpleNamespace(persistent_state_bindings={}),
        dummy_run=False,
        is_profile=False,
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        state.prepare_inputs(
            SimpleNamespace(req_ids=["_dummy_req_0"]),
            object(),
        )

    state.end_omni_projection()


def test_failed_dummy_cleanup_allows_an_immediate_real_projection() -> None:
    # @spec PORT-MIG-005
    state = _projection_state()
    state.begin_omni_projection(
        SimpleNamespace(persistent_state_bindings={}),
        dummy_run=True,
        is_profile=True,
    )

    def fail_profile(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("profile failed")

    state._projection.complete_dummy = fail_profile

    with pytest.raises(RuntimeError, match="profile failed"):
        state.prepare_inputs(SimpleNamespace(req_ids=["dummy"]), object())
    state.end_omni_projection()

    state.begin_omni_projection(
        SimpleNamespace(persistent_state_bindings={}),
        dummy_run=False,
        is_profile=False,
    )
    epoch = state._projection_epoch
    assert epoch is not None
    features = object()
    binding = object()
    metadata = object()
    input_batch = SimpleNamespace(req_ids=["real"])
    req_states = object()
    state._projection.record_mm(
        epoch,
        req_ids=("real",),
        scheduled_encoder_inputs={"real": [0]},
        encoder_features=features,
    )
    state._projection.record_bindings(
        epoch,
        req_ids=("real",),
        bindings=(binding,),
        block_ids=((1,),),
    )
    state._request_metadata = SimpleNamespace(snapshot=lambda req_ids: {req_ids[0]: metadata})

    prepared = state.prepare_inputs(input_batch, req_states)
    snapshot = prepared["persistent_state_projection"]

    assert snapshot.dummy_run is False
    assert snapshot.is_profile is False
    assert snapshot.no_page_io is False
    assert snapshot.encoder_features is features
    assert snapshot.bindings == (binding,)
    assert snapshot.request_metadata == {"real": metadata}
    state.end_omni_projection()


def test_terminal_reconciliation_prunes_only_finished_absent_metadata() -> None:
    # @spec PORT-MIG-004 / PORT-MIG-005
    metadata = SimpleNamespace(pruned=[])
    metadata.prune_finished = metadata.pruned.extend
    state_cls = _module().NemotronASRModelState
    state = object.__new__(state_cls)
    state._request_metadata = metadata

    state.reconcile_omni_lifecycle(
        finished_req_ids=frozenset({"done", "still-resident"}),
        preempted_req_ids=frozenset(),
        resident_req_ids=frozenset({"still-resident", "unscheduled"}),
    )

    assert metadata.pruned == ["done"]


def test_preemption_is_named_nonterminal_failure_and_never_prunes() -> None:
    # @spec PORT-STATE-005 / PORT-MIG-005
    metadata = SimpleNamespace(pruned=[])
    metadata.prune_finished = metadata.pruned.extend
    state_cls = _module().NemotronASRModelState
    state = object.__new__(state_cls)
    state._request_metadata = metadata

    with pytest.raises(RuntimeError, match="no-recompute.*preempt"):
        state.reconcile_omni_lifecycle(
            finished_req_ids=frozenset(),
            preempted_req_ids=frozenset({"parked"}),
            resident_req_ids=frozenset(),
        )

    assert metadata.pruned == []
