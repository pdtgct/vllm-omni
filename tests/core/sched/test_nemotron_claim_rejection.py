# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rejected leases must leave the real scheduler queues before allocation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

from tests.engine.test_persistent_state_service_contract import _state_core_for_cleanup_tests
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.model_executor.models.nemotron_asr.scheduler import NemotronASRScheduler
from vllm_omni.model_executor.persistent_state.manager import StateBinding

if TYPE_CHECKING:

    class _LeaseRequest(Request):
        additional_information: dict[str, dict[str, object]]


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(request_id: str) -> _LeaseRequest:
    # This fixture attaches the exact dict carrier below; queue cleanup still
    # executes on the real pinned Request instance, without a behavioral fake.
    request = cast(
        "_LeaseRequest",
        Request(
            request_id=request_id,
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=None,
        ),
    )
    request.additional_information = {
        "persistent_state_binding": {
            "engine_epoch": "epoch",
            "session_key": request_id,
            "generation": 1,
            "schema_id": "schema",
            "profile_id": "profile",
            "binding_token": request_id,
        }
    }
    return request


@pytest.mark.parametrize(
    "fault",
    ["stale", "missing_payload", "missing_registry", "invalid_binding"]
    + [
        f"missing_{field}"
        for field in ("engine_epoch", "session_key", "generation", "schema_id", "profile_id", "binding_token")
    ],
)
def test_rejected_claim_is_removed_before_allocation_and_next_request_runs(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    # @spec PORT-STATE-019 / PORT-STATE-014
    scheduler = object.__new__(NemotronASRScheduler)
    bad = _request("released-session")
    scheduler.requests = {bad.request_id: bad}
    scheduler.waiting = FCFSRequestQueue()
    scheduler.waiting.add_request(bad)
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.running = []
    scheduler._claimed_state_bindings = {}
    scheduler._rejected_state_claims = set()
    scheduler.finished_req_ids = set()
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    registry = Mock()
    registry.claim_pending_lease.side_effect = ValueError("persistent_state pending claim mismatch")
    monkeypatch.setattr(scheduler, "persistent_state_registry", registry, raising=False)
    manager = None
    if fault == "stale":
        core, manager, spec = _state_core_for_cleanup_tests()
        reserved = core.persistent_state_reserve("reserve", bad.request_id, spec.schema_id, "default")
        lease = reserved["lease"]
        assert core.persistent_state_begin_pending_cleanup(lease)
        core.persistent_state_release("release", lease, "pending_claim_timeout")
        assert manager.get_state_binding(bad.request_id) is None
        bad.additional_information["persistent_state_binding"] = lease
        registry.claim_pending_lease.side_effect = core.claim_pending_lease
    elif fault == "missing_payload":
        bad.additional_information = {}
    elif fault == "missing_registry":
        monkeypatch.setattr(scheduler, "persistent_state_registry", None)
    elif fault == "invalid_binding":
        registry.claim_pending_lease.side_effect = None
        registry.claim_pending_lease.return_value = object()
    elif fault.startswith("missing_"):
        bad.additional_information["persistent_state_binding"].pop(fault.removeprefix("missing_"))

    freed = []

    def free_request(request: Request, delay_free_blocks: bool = False) -> None:
        assert request.status == RequestStatus.FINISHED_ERROR
        freed.append(request.request_id)
        scheduler.requests.pop(request.request_id)

    # Keep the actual Omni + pinned vLLM finish_requests methods: their
    # is_finished guard and queue removal are the regression boundary.
    monkeypatch.setattr(scheduler, "_free_request", free_request)
    allocations = []

    def base_schedule(self: NemotronASRScheduler, throttle_prefills: bool = False) -> SimpleNamespace:
        for request in self.waiting:
            allocations.append(request.request_id)
            if request.request_id == bad.request_id:
                if manager is not None:
                    manager.allocate_new_blocks(request.request_id, 1, 1)
                raise ValueError("persistent-state request identity cannot be reused")
        return SimpleNamespace(num_scheduled_tokens={r: 1 for r in allocations}, persistent_state_bindings={})

    monkeypatch.setattr(OmniARScheduler, "schedule", base_schedule)
    scheduler.schedule()
    assert freed == [bad.request_id]
    assert not scheduler.waiting
    assert not scheduler.skipped_waiting
    assert not scheduler.requests
    assert allocations == []
    registry.mark_terminal.assert_not_called()
    # The inherited Omni path emits external finishes as ABORT. The model
    # scheduler must retain the lifecycle error at that output boundary.
    outputs = {
        0: EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(bad.request_id, [], finish_reason=FinishReason.ABORT),
                EngineCoreOutput("ordinary-abort", [], finish_reason=FinishReason.ABORT),
            ]
        )
    }
    monkeypatch.setattr(OmniARScheduler, "update_from_output", lambda *args: outputs)
    delivered = scheduler.update_from_output(None, None)
    assert delivered[0].outputs[0].finish_reason == FinishReason.ERROR
    assert delivered[0].outputs[1].finish_reason == FinishReason.ABORT
    assert not scheduler._rejected_state_claims

    healthy = _request("new-session")
    binding = StateBinding(
        request_id=healthy.request_id,
        slot_id=1,
        generation=2,
        schema_id="schema",
        profile_id="profile",
        stage=0,
        replica=0,
        fresh=True,
        engine_epoch="epoch",
    )
    registry.claim_pending_lease.side_effect = None
    registry.claim_pending_lease.return_value = binding
    monkeypatch.setattr(scheduler, "persistent_state_registry", registry, raising=False)
    scheduler.requests[healthy.request_id] = healthy
    scheduler.waiting.add_request(healthy)
    output = scheduler.schedule()
    assert allocations == [healthy.request_id]
    assert output.persistent_state_bindings == {healthy.request_id: binding}
