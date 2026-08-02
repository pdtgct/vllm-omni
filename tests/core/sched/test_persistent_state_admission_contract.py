# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for state-aware scheduler admission and claiming."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.output import OmniSchedulerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[3]


def _module() -> Any:
    try:
        return importlib.import_module(
            "vllm_omni.model_executor.models.nemotron_asr.scheduler"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "PORT-STATE-019 missing NemotronASRScheduler module",
            pytrace=False,
        )


def test_model_scheduler_is_selected_explicitly_by_stage_configuration() -> None:
    # @spec PORT-STATE-004 / PORT-STATE-019
    scheduler_cls = _module().NemotronASRScheduler
    stage_config = (
        _ROOT
        / "vllm_omni"
        / "model_executor"
        / "stage_configs"
        / "nemotron_asr.yaml"
    ).read_text()

    assert issubclass(scheduler_cls, OmniARScheduler)
    assert (
        "scheduler_cls: "
        "vllm_omni.model_executor.models.nemotron_asr.scheduler."
        "NemotronASRScheduler"
    ) in stage_config


def test_initial_add_claims_exact_pending_binding_before_base_schedule() -> None:
    # @spec PORT-STATE-019
    scheduler_cls = _module().NemotronASRScheduler
    source = inspect.getsource(scheduler_cls.schedule)

    claim_offset = source.index("claim_pending_lease")
    schedule_offset = source.index("super().schedule")
    assert claim_offset < schedule_offset
    for field in (
        "engine_epoch",
        "session_key",
        "generation",
        "schema_id",
        "profile_id",
    ):
        assert field in source[:schedule_offset]


def test_claim_failure_is_lifecycle_error_not_capacity_or_allocation() -> None:
    # @spec PORT-STATE-019
    scheduler_cls = _module().NemotronASRScheduler
    claim_region = inspect.getsource(scheduler_cls._claim_initial_request)

    assert "FINISHED_ERROR" in claim_region
    assert "lifecycle" in claim_region.lower() or "invariant" in claim_region.lower()
    assert "capacity_exhausted" not in claim_region
    assert "allocate" not in claim_region


def test_streaming_readd_verifies_without_claiming_or_allocating_again() -> None:
    # @spec PORT-STATE-010 / PORT-STATE-019
    scheduler_cls = _module().NemotronASRScheduler
    source = inspect.getsource(scheduler_cls._verify_streaming_readd)

    assert "generation" in source
    assert "block" in source
    assert "claim_pending_lease" not in source
    assert "allocate" not in source
    assert "reserve" not in source


def test_scheduler_output_carries_a_typed_binding_sidecar() -> None:
    # @spec PORT-STATE-007 / PORT-STATE-019
    fields = getattr(OmniSchedulerOutput, "__struct_fields__", ())
    annotations = getattr(OmniSchedulerOutput, "__annotations__", {})

    assert "persistent_state_bindings" in set(fields) | set(annotations)
    assert annotations.get("persistent_state_bindings") not in (None, Any)


def test_scheduler_finish_never_physically_releases_a_state_lease() -> None:
    # @spec PORT-STATE-014
    scheduler_cls = _module().NemotronASRScheduler
    source = inspect.getsource(scheduler_cls.update_from_output)

    assert "mark_terminal" in source
    assert "persistent_state_release" not in source
    assert ".release(" not in source
    assert "free_slot" not in source
    assert "dropped" not in source


def test_recompute_preemption_is_converted_to_named_terminal_error() -> None:
    # @spec PORT-STATE-005
    scheduler_cls = _module().NemotronASRScheduler
    source = inspect.getsource(scheduler_cls)

    assert "no-recompute" in source or "no_recompute" in source
    assert "PREEMPTED" in source
    assert "FINISHED_ERROR" in source
    assert "num_computed_tokens = 0" not in source


def test_scheduler_never_owns_service_or_api_cleanup() -> None:
    # @spec PORT-STATE-004 / PORT-STATE-014
    source = inspect.getsource(_module().NemotronASRScheduler)

    assert "PersistentStateService" not in source
    assert "session_finished" not in source
    assert "streaming_sessions_active" not in source


@dataclass(frozen=True)
class _MMPosition:
    offset: int
    length: int


def _streaming_session() -> Request:
    session = Request(
        request_id="bounded-history-session",
        prompt_token_ids=[101],
        sampling_params=SamplingParams(max_tokens=142),
        pooling_params=None,
        block_hasher=None,
        resumable=True,
    )
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.additional_information = {
        "persistent_state_binding": {
            "engine_epoch": "epoch-a",
            "generation": 17,
            "slot_id": 3,
            "binding_token": "binding-a",
        }
    }
    return session


def _streaming_update(token_id: int) -> StreamingUpdate:
    feature = SimpleNamespace(
        modality="audio",
        mm_position=_MMPosition(offset=0, length=1),
        data=SimpleNamespace(kwargs={"chunk_sequence": token_id}),
    )
    return StreamingUpdate(
        mm_features=[feature],
        prompt_token_ids=[token_id],
        max_tokens=142,
        arrival_time=float(token_id),
        sampling_params=SamplingParams(max_tokens=142),
        additional_information={
            "persistent_state_binding": {
                "engine_epoch": "epoch-a",
                "generation": 17,
                "slot_id": 3,
                "binding_token": "binding-a",
            }
        },
    )


def _streaming_scheduler() -> Any:
    scheduler_cls = _module().NemotronASRScheduler
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler._new_prompt_len_snapshot = {}
    scheduler.num_waiting_for_streaming_input = 1
    scheduler.log_stats = False
    return scheduler


def test_long_stream_replaces_scheduler_envelope_without_growing_history() -> None:
    # @spec PORT-STATE-020 / PORT-STATE-021
    scheduler = _streaming_scheduler()
    session = _streaming_session()
    original_binding = session.additional_information["persistent_state_binding"]

    # 128 updates exceed ten turnovers at the widest shipped 14-frame
    # geometry (10 * ceil(56 / 14) = 40). A history-appending
    # implementation cannot pass accidentally.
    for sequence in range(128):
        session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        scheduler.num_waiting_for_streaming_input = 1
        session.append_output_token_ids([10_000 + sequence, 20_000 + sequence])
        session.num_computed_tokens = session.num_tokens
        session.mm_features.append(
            SimpleNamespace(
                modality="audio",
                mm_position=_MMPosition(offset=99, length=1),
                data=object(),
            )
        )
        session.block_hashes.append(object())

        scheduler._update_request_as_session(
            session,
            _streaming_update(30_000 + sequence),
        )

        assert session.prompt_token_ids == [30_000 + sequence]
        assert list(session.all_token_ids) == [30_000 + sequence]
        assert list(session.output_token_ids) == []
        assert session.num_prompt_tokens == 1
        assert session.num_computed_tokens == 0
        assert len(session.mm_features) == 1
        assert session.mm_features[0].mm_position == _MMPosition(0, 1)
        assert session.block_hashes == []
        assert session.additional_information["persistent_state_binding"] is original_binding


def test_replacement_resets_only_transient_compute_state() -> None:
    # @spec PORT-STATE-005 / PORT-STATE-020
    scheduler = _streaming_scheduler()
    session = _streaming_session()
    binding = session.additional_information["persistent_state_binding"]
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = session.num_tokens
    session.num_output_placeholders = 1
    session.spec_token_ids = [-1]

    scheduler._update_request_as_session(session, _streaming_update(44))

    assert session.async_tokens_to_discard == 1
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    assert session.num_computed_tokens == 0
    assert session.additional_information["persistent_state_binding"] is binding


def test_replacement_rejects_feature_position_from_accumulated_history() -> None:
    # @spec PORT-STATE-020
    scheduler = _streaming_scheduler()
    session = _streaming_session()
    update = _streaming_update(44)
    update.mm_features[0].mm_position = _MMPosition(offset=9, length=1)

    with pytest.raises(ValueError, match="current prompt|replacement prompt"):
        scheduler._update_request_as_session(session, update)


def test_replacement_requires_the_parked_streaming_update_state() -> None:
    # @spec PORT-STATE-020
    scheduler = _streaming_scheduler()
    session = _streaming_session()
    session.status = RequestStatus.RUNNING
    before = list(session.all_token_ids)

    with pytest.raises((AssertionError, RuntimeError, ValueError), match="park|streaming|waiting"):
        scheduler._update_request_as_session(session, _streaming_update(44))

    assert list(session.all_token_ids) == before


def test_generic_omni_stage_zero_history_policy_is_not_changed_globally() -> None:
    # @spec PORT-STATE-020
    generic_source = inspect.getsource(OmniARScheduler._update_request_as_session)
    model_source = inspect.getsource(_module().NemotronASRScheduler._update_request_as_session)

    assert "session.prompt_token_ids.extend" in generic_source
    assert "super()._update_request_as_session(session, update)" not in model_source


def test_model_length_is_a_per_transaction_bound_not_a_session_clock() -> None:
    # @spec PORT-INT-002 / PORT-SESS-006 / PORT-STATE-020
    validate = _module().NemotronASRScheduler._validate_streaming_model_len

    # Shipped widest geometry: 14 frames * 10 labels + EOU + park.
    with pytest.raises(ValueError, match="one current streaming transaction"):
        validate(141)

    validate(142)
    validate(1_000_000)
