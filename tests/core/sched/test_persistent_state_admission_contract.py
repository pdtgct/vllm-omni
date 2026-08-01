# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for state-aware scheduler admission and claiming."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

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
