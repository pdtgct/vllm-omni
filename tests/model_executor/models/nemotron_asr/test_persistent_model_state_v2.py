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
        return importlib.import_module(
            "vllm_omni.model_executor.models.nemotron_asr.model_state_v2"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "PORT-MIG-004 missing direct NemotronASRModelState module",
            pytrace=False,
        )


def _model_class() -> type[Any]:
    try:
        module = importlib.import_module(
            "vllm_omni.model_executor.models.nemotron_asr.nemotron_asr"
        )
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
