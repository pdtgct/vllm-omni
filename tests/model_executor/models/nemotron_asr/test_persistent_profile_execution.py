# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for honest no-resident-state startup profiling."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _profile_module() -> Any:
    try:
        return importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.profile_execution")
    except ModuleNotFoundError:
        pytest.fail(
            "PORT-MIG-005 missing persistent-state profile execution",
            pytrace=False,
        )


def _model_module() -> Any:
    return importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.nemotron_asr")


def _model_class() -> type[Any]:
    return cast(type[Any], _model_module().NemotronASRForRNNT)


def _config() -> Any:
    config_module = importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr")
    return config_module.NemotronASRConfig(
        vocab_size=13_092,
        num_asr_labels=13_087,
        eos_token_id=13_088,
        audio_chunk_token_id=13_089,
        eou_token_id=13_090,
        flush_token_id=13_091,
        prompt_dictionary={"en-US": 0},
        decode_dispatch_arm="dense-eager",
    )


def _projection(*, is_profile: bool, num_rows: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        dummy_run=True,
        is_profile=is_profile,
        no_page_io=True,
        req_ids=tuple(f"opaque-{index}" for index in range(num_rows)),
        bindings=(),
        request_metadata=None,
    )


def _no_state_model() -> Any:
    model = object.__new__(_model_class())
    object.__setattr__(model, "config", _config())
    object.__setattr__(model, "core", object())
    object.__setattr__(model, "_emission_adapter", object())
    object.__setattr__(model, "_decode_resolver", object())
    return model


def test_profile_invocation_uses_largest_geometry_and_maximum_live_rows() -> None:
    # @spec PORT-MIG-005 / PORT-STATE-003 / PORT-STATE-007
    profile = _profile_module()
    config = _config()

    invocation = profile.build_profile_invocation(
        config,
        num_rows=3,
        device=torch.device("cpu"),
    )

    assert invocation.geometry_label == "1120ms"
    assert invocation.geometry_id == 4
    assert invocation.num_rows == 3
    assert invocation.plan.num_decodes == 0
    assert invocation.plan.num_prefills == 3
    assert invocation.plan.num_pool_blocks == 4
    assert invocation.plan.null_block_id == 0
    assert invocation.plan.state_indices_p.tolist() == [1, 2, 3]
    assert invocation.plan.live_block_ids.tolist() == [1, 2, 3]
    assert invocation.plan.has_initial_states_p.tolist() == [False] * 3
    assert invocation.plan.is_chunk.tolist() == [True] * 3
    assert invocation.plan.geometry_id.tolist() == [4] * 3


def test_profile_invocation_uses_ephemeral_manifest_storage_and_valid_carriers() -> None:
    # @spec PORT-MIG-005 / PORT-INT-004 / PORT-STATE-009
    profile = _profile_module()
    manifests = importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.manifests")
    config = _config()

    invocation = profile.build_profile_invocation(
        config,
        num_rows=2,
        device=torch.device("cpu"),
    )

    assert invocation.storage.raw.numel() == (invocation.storage.spec.page_size_bytes * 3)
    assert invocation.input_ids.tolist() == [config.audio_chunk_token_id] * 2
    assert invocation.inputs_embeds.shape == (2, config.hidden_size)
    valid_index = manifests.ENVELOPE_HEADER_FIELDS.index("valid_samples")
    geometry_index = manifests.ENVELOPE_HEADER_FIELDS.index("geometry_id")
    assert invocation.inputs_embeds[:, valid_index].tolist() == [17_920.0] * 2
    assert invocation.inputs_embeds[:, geometry_index].tolist() == [4.0] * 2


def test_profile_execution_invokes_the_canonical_transaction_and_drains_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001 / PORT-MIG-003 / PORT-MIG-005
    profile = _profile_module()
    invocation = SimpleNamespace(
        input_ids=torch.zeros(2, dtype=torch.long),
        inputs_embeds=torch.zeros(2, 7),
        plan=object(),
        pools=SimpleNamespace(
            channel=(),
            convolution=(),
            valid_length=(),
            predictor_h=object(),
            predictor_c=object(),
            replay_queue=object(),
            replay_book=object(),
            frontend_raw=object(),
            frontend_mel=object(),
            frontend_counters=object(),
            endpoint_history=object(),
            endpoint_book=object(),
        ),
    )
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        profile.logger,
        "info",
        lambda *args: events.append(("log", args)),
    )
    monkeypatch.setattr(
        profile,
        "build_profile_invocation",
        lambda config, *, num_rows, device: events.append(("build", (config, num_rows, device))) or invocation,
    )
    monkeypatch.setattr(
        profile,
        "advance_model_rows",
        lambda *args, **kwargs: events.append(("advance", (args, kwargs))),
    )
    monkeypatch.setattr(
        profile,
        "consume_batch_stats",
        lambda: events.append(("drain", None)),
    )
    model = _no_state_model()

    profile.run_persistent_state_profile(
        model,
        num_rows=2,
        device=torch.device("cpu"),
    )

    assert [name for name, _ in events] == ["log", "build", "advance", "drain"]
    assert "persistent-state profile execution" in events[0][1][0]
    advance_args, advance_kwargs = events[2][1]
    assert advance_args[:4] == (
        model.core,
        invocation.input_ids,
        invocation.inputs_embeds,
        invocation.plan,
    )
    assert advance_kwargs["capture"] is False
    assert advance_kwargs["commit_sink"] is None


def test_profile_execution_drains_stats_when_the_transition_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-003 / PORT-MIG-005
    profile = _profile_module()
    invocation = SimpleNamespace(
        input_ids=torch.zeros(1, dtype=torch.long),
        inputs_embeds=torch.zeros(1, 7),
        plan=object(),
        pools=SimpleNamespace(
            channel=(),
            convolution=(),
            valid_length=(),
            predictor_h=object(),
            predictor_c=object(),
            replay_queue=object(),
            replay_book=object(),
            frontend_raw=object(),
            frontend_mel=object(),
            frontend_counters=object(),
            endpoint_history=object(),
            endpoint_book=object(),
        ),
    )
    drained: list[bool] = []
    monkeypatch.setattr(
        profile,
        "build_profile_invocation",
        lambda *args, **kwargs: invocation,
    )
    monkeypatch.setattr(
        profile,
        "advance_model_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    monkeypatch.setattr(
        profile,
        "consume_batch_stats",
        lambda: drained.append(True),
    )

    with pytest.raises(RuntimeError, match="failed"):
        profile.run_persistent_state_profile(
            _no_state_model(),
            num_rows=1,
            device=torch.device("cpu"),
        )

    assert drained == [True]


def test_profile_execution_drains_stats_on_empty_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-003 / PORT-MIG-005
    profile = _profile_module()
    drained: list[bool] = []
    monkeypatch.setattr(
        profile,
        "consume_batch_stats",
        lambda: drained.append(True),
    )

    with pytest.raises(ValueError, match="row count must be positive"):
        profile.run_persistent_state_profile(
            _no_state_model(),
            num_rows=0,
            device=torch.device("cpu"),
        )

    assert drained == [True]


@pytest.mark.parametrize("is_profile", [True, False])
def test_forward_consumes_no_state_projection_before_resident_authorities(
    monkeypatch: pytest.MonkeyPatch,
    is_profile: bool,
) -> None:
    # @spec PORT-MIG-005 / PORT-STATE-003 / PORT-STATE-007
    profile = _profile_module()
    model_cls = _model_class()
    model = _no_state_model()
    calls: list[tuple[int, torch.device]] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail(
            "PORT-MIG-005 dummy touched resident projection authority",
            pytrace=False,
        )

    monkeypatch.setattr(model_cls, "_stage_v2_projection", forbidden)
    monkeypatch.setattr(model_cls, "_state_pools", forbidden)
    monkeypatch.setattr(
        profile,
        "run_persistent_state_profile",
        lambda value, *, num_rows, device: calls.append((num_rows, device)),
    )
    input_ids = torch.zeros(7, dtype=torch.long)
    inputs_embeds = torch.zeros(7, 11, dtype=torch.float32)

    output = model.forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        persistent_state_projection=_projection(is_profile=is_profile),
    )

    assert output.shape == inputs_embeds.shape
    assert output.dtype == inputs_embeds.dtype
    assert output.device == inputs_embeds.device
    assert calls == ([(2, torch.device("cpu"))] if is_profile else [])


def test_profile_forward_accepts_mrv2_embedding_only_dummy_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005 / PORT-STATE-007
    profile = _profile_module()
    model_cls = _model_class()
    model = _no_state_model()
    calls: list[tuple[int, torch.device]] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail(
            "PORT-MIG-005 MRv2 profile touched resident authority",
            pytrace=False,
        )

    monkeypatch.setattr(model_cls, "_stage_v2_projection", forbidden)
    monkeypatch.setattr(model_cls, "_state_pools", forbidden)
    monkeypatch.setattr(
        profile,
        "run_persistent_state_profile",
        lambda value, *, num_rows, device: calls.append((num_rows, device)),
    )
    inputs_embeds = torch.zeros(7, 11, dtype=torch.float32)

    output = model.forward(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        persistent_state_projection=_projection(is_profile=True),
    )

    assert output.shape == inputs_embeds.shape
    assert output.dtype == inputs_embeds.dtype
    assert output.device == inputs_embeds.device
    assert calls == [(2, torch.device("cpu"))]


def test_prepare_inputs_snapshot_flows_directly_into_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005 / PORT-STATE-003 / PORT-STATE-007
    profile = _profile_module()
    state_module = importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.model_state_v2")
    state = object.__new__(state_module.NemotronASRModelState)
    state._projection = state_module.ProjectionJoin()
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
    state.begin_omni_projection(
        SimpleNamespace(persistent_state_bindings={}),
        dummy_run=True,
        is_profile=True,
    )

    model_cls = _model_class()
    model = _no_state_model()
    calls: list[tuple[int, torch.device]] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail(
            "PORT-MIG-005 composed dummy touched resident projection authority",
            pytrace=False,
        )

    monkeypatch.setattr(model_cls, "_stage_v2_projection", forbidden)
    monkeypatch.setattr(model_cls, "_state_pools", forbidden)
    monkeypatch.setattr(
        profile,
        "run_persistent_state_profile",
        lambda value, *, num_rows, device: calls.append((num_rows, device)),
    )
    input_ids = torch.zeros(7, dtype=torch.long)
    inputs_embeds = torch.zeros(7, 11, dtype=torch.float32)

    try:
        prepared = state.prepare_inputs(
            SimpleNamespace(req_ids=["opaque-dummy"]),
            object(),
        )
        output = model.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            persistent_state_projection=prepared["persistent_state_projection"],
        )
    finally:
        state.end_omni_projection()

    assert output.shape == inputs_embeds.shape
    assert output.dtype == inputs_embeds.dtype
    assert output.device == inputs_embeds.device
    assert calls == [(1, torch.device("cpu"))]
